from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyaudio
from resemblyzer import normalize_volume
from resemblyzer.hparams import audio_norm_target_dBFS

from src.llm.client import AsyncLLMClient
from src.vad.async_vad import AsyncVAD

TARGET_SR = 16000


@dataclass
class UserTurn:
    text: str
    audio_bytes: bytes | None = None
    audio_array: np.ndarray | None = None


class InputProcessor(Protocol):
    # Not a coroutine: returns an async iterator you can "async for" over
    def stream(self) -> AsyncIterator[UserTurn]: ...
    # Optional: allow "async for turn in processor" directly
    def __aiter__(self) -> AsyncIterator[UserTurn]: ...


def _segment_to_wav_bytes(segment: np.ndarray, sr: int = TARGET_SR) -> bytes:
    seg: np.ndarray = np.asarray(segment, dtype=np.float32).ravel()
    seg = np.clip(seg, -1.0, 1.0)
    pcm16: np.ndarray = (seg * 32767.0).astype('<i2', copy=False)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class WavFileInputProcessor:
    """Produces transcribed user turns from an audio file via VAD."""

    def __init__(self, audio_file: str, vad: AsyncVAD, llm_client: AsyncLLMClient):
        self.audio_file = audio_file
        self.vad = vad
        self.llm_client = llm_client
        self._aiter: AsyncIterator[np.ndarray] | None = None

    async def stream(self) -> AsyncIterator[UserTurn]:
        if self._aiter is None:
            self._aiter = self.vad.detect_from_file(self.audio_file).__aiter__()
        while True:
            try:
                segment: np.ndarray = await self._aiter.__anext__()
            except StopAsyncIteration:
                break

            # Normalize and package
            segment = normalize_volume(segment, audio_norm_target_dBFS, increase_only=True)
            audio_bytes = _segment_to_wav_bytes(segment)
            text = await self.llm_client.transcribe_bytes(audio_bytes)

            yield UserTurn(text=text, audio_bytes=audio_bytes, audio_array=segment)


class TextInputProcessor:
    """Produces user turns from stdin or a script file."""

    def __init__(self, script_file: str | None = None):
        self.script_file = script_file

    async def stream(self) -> AsyncIterator[UserTurn]:
        if self.script_file:
            path = Path(self.script_file)
            lines = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines()]
            for ln in lines:
                if ln and not ln.lstrip().startswith('#'):
                    yield UserTurn(text=ln)
                    await asyncio.sleep(0)  # let loop breathe
        else:
            # interactive stdin
            while True:
                try:
                    line = await asyncio.to_thread(input, 'You: ')
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                yield UserTurn(text=line)


class MicrophoneInputProcessor:
    """Stream live audio from microphone (PulseAudio via parec or PyAudio fallback)."""

    def __init__(
        self,
        vad: AsyncVAD,
        llm_client: AsyncLLMClient,
        use_pulse: bool = True,
        sample_rate: int = TARGET_SR,
        channels: int = 1,
        chunk_ms: int = 200,
        listen_event: asyncio.Event | None = None,  # NEW: gate
        discard_ms_on_resume: int = 300,  # NEW: discard residual audio
    ):
        self.vad = vad
        self.llm_client = llm_client
        self.use_pulse = use_pulse
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self._proc: asyncio.subprocess.Process | None = None
        self.listen_event = listen_event or asyncio.Event()
        self.discard_ms_on_resume = discard_ms_on_resume
        self._discarding = False
        self._discard_remaining_samples = 0

    async def _default_pulse_source(self) -> str | None:
        proc = await asyncio.create_subprocess_exec(
            'pactl',
            'list',
            'sources',
            'short',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        lines = out.decode().splitlines()
        # Prefer a non-monitor source
        mic_candidates = [ll.split()[1] for ll in lines if 'monitor' not in ll.lower()]
        return mic_candidates[0] if mic_candidates else None

    async def _open_pulse_stream(self) -> asyncio.subprocess.Process:
        source = await self._default_pulse_source()
        if source is None:
            raise RuntimeError('No non-monitor PulseAudio source found')
        return await asyncio.create_subprocess_exec(
            'parec',
            '--device',
            source,
            '--rate',
            str(self.sample_rate),
            '--channels',
            str(self.channels),
            '--format',
            's16le',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    def resume_listening(self) -> None:
        # Prepare discard window to flush residual playback bleed
        self._discarding = True
        samples_to_discard = int(self.sample_rate * (self.discard_ms_on_resume / 1000.0))
        self._discard_remaining_samples = samples_to_discard
        self.listen_event.set()

    def pause_listening(self) -> None:
        self.listen_event.clear()

    async def stream(self) -> AsyncIterator[UserTurn]:
        if self.use_pulse:
            if self._proc is None:
                self._proc = await self._open_pulse_stream()
            bytes_per_sample = 2
            chunk_samples = int(self.sample_rate * (self.chunk_ms / 1000.0))
            chunk_bytes = chunk_samples * self.channels * bytes_per_sample
            pending = np.empty((0,), dtype=np.float32)

            while True:
                # Wait until listening enabled
                await self.listen_event.wait()

                # safe check for if stdout is none
                if self._proc.stdout is None:
                    break

                buf = await self._proc.stdout.read(chunk_bytes)
                if not buf:
                    break
                arr = np.frombuffer(buf, dtype=np.int16)
                float_arr = arr.astype(np.float32) / 32767.0

                # Discard initial samples after resume
                if self._discarding:
                    need = self._discard_remaining_samples
                    if need <= float_arr.size:
                        float_arr = float_arr[need:]
                        self._discarding = False
                        self._discard_remaining_samples = 0
                    else:
                        self._discard_remaining_samples -= float_arr.size
                        await asyncio.sleep(0)
                        continue  # continue discarding

                pending = np.concatenate((pending, float_arr))

                # Feed frames to VAD
                while len(pending) >= self.vad.frame_size:
                    frame = pending[: self.vad.frame_size]
                    pending = pending[self.vad.frame_size :]
                    segment = await self.vad.process_live_frame(frame)
                    if segment is not None and segment.size:
                        segment = normalize_volume(
                            segment, audio_norm_target_dBFS, increase_only=True
                        )
                        audio_bytes = _segment_to_wav_bytes(segment)
                        text = await self.llm_client.transcribe_bytes(audio_bytes)
                        yield UserTurn(text=text, audio_bytes=audio_bytes, audio_array=segment)
                    await asyncio.sleep(0)
        else:
            # Fallback: PyAudio (apply same gating/discard)
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * self.chunk_ms / 1000),
            )
            try:
                pending = np.empty((0,), dtype=np.float32)
                while True:
                    await self.listen_event.wait()
                    buf = stream.read(
                        int(self.sample_rate * self.chunk_ms / 1000), exception_on_overflow=False
                    )
                    arr = np.frombuffer(buf, dtype=np.int16)
                    float_arr = arr.astype(np.float32) / 32767.0

                    if self._discarding:
                        need = self._discard_remaining_samples
                        if need <= float_arr.size:
                            float_arr = float_arr[need:]
                            self._discarding = False
                            self._discard_remaining_samples = 0
                        else:
                            self._discard_remaining_samples -= float_arr.size
                            await asyncio.sleep(0)
                            continue

                    pending = np.concatenate((pending, float_arr))
                    while len(pending) >= self.vad.frame_size:
                        frame = pending[: self.vad.frame_size]
                        pending = pending[self.vad.frame_size :]
                        segment = await self.vad.process_live_frame(frame)
                        if segment is not None and segment.size:
                            segment = normalize_volume(
                                segment, audio_norm_target_dBFS, increase_only=True
                            )
                            audio_bytes = _segment_to_wav_bytes(segment)
                            text = await self.llm_client.transcribe_bytes(audio_bytes)
                            yield UserTurn(text=text, audio_bytes=audio_bytes, audio_array=segment)
                        await asyncio.sleep(0)
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

    def __aiter__(self) -> AsyncIterator[UserTurn]:
        return self.stream()

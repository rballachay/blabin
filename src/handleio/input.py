from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
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


class AudioInputProcessor:
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

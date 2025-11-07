import asyncio
import io
import tempfile
import wave
from typing import Protocol

import pyaudio

from src.llm.client import AsyncLLMClient


class OutputProcessor(Protocol):
    async def output(
        self, text: str, llm_client: AsyncLLMClient, speak_allowed: bool = True
    ) -> None: ...
    async def aclose(self) -> None: ...


class TextOutputProcessor:
    """Writes assistant messages to stdout."""

    async def output(
        self, text: str, llm_client: AsyncLLMClient, speak_allowed: bool = True
    ) -> None:
        if not text:
            return
        print(f'Agent: {text}\n')

    async def aclose(self) -> None:
        return


class AudioOutputProcessor:
    """Generates TTS via llm_client and plays WAV bytes with PyAudio.
    Falls back to stdout when speak_allowed=False.
    """

    def __init__(self, use_pulse: bool = True) -> None:
        if not use_pulse:
            self._pa = pyaudio.PyAudio()
        else:
            self._pa = None
        self._stream = None

        self.use_pulse = use_pulse

    async def _play_wav_bytes(self, audio_bytes: bytes) -> None:
        wav_io = io.BytesIO(audio_bytes)
        with wave.open(wav_io, 'rb') as wave_file:
            if self._pa:
                self._stream = self._pa.open(
                    format=self._pa.get_format_from_width(wave_file.getsampwidth()),
                    channels=wave_file.getnchannels(),
                    rate=wave_file.getframerate(),
                    output=True,
                )
                chunk_size = 1024
                data = wave_file.readframes(chunk_size)
                while data:
                    if self._stream is not None:
                        await asyncio.to_thread(self._stream.write, data)
                    data = wave_file.readframes(chunk_size)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    async def _play_wav_bytes_pulse(self, audio_bytes: bytes) -> None:
        if not audio_bytes or audio_bytes[:4] != b'RIFF':
            return
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            proc = await asyncio.create_subprocess_exec(
                'paplay',
                tmp.name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

    async def output(
        self, text: str, llm_client: AsyncLLMClient, speak_allowed: bool = True
    ) -> None:
        if not text:
            return
        if not speak_allowed:
            print(f'Agent: {text}\n')
            return
        audio_bytes = await llm_client.text_to_speech(text)

        if self.use_pulse:
            await self._play_wav_bytes_pulse(audio_bytes)
        else:
            await self._play_wav_bytes(audio_bytes)

    async def aclose(self) -> None:
        # Ensure stream/device closed
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()

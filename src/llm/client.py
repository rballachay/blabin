import asyncio
import io
import wave
from typing import Any

from google import genai
from google.genai import types


class AsyncLLMClient:
    def __init__(self, api_key: str):
        """
        Initializes the Gemini client with the provided API key.
        """
        self.client = genai.Client(api_key=api_key)

    async def send_request(self, prompt: list[dict[str, Any]]) -> str:
        """
        Stream the Gemini LLM response as it arrives.
        Returns text chunks from the server.
        """
        response = await self.client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,  # type: ignore[arg-type]
        )
        return response.text or ''

    def _wrap_pcm_as_wav(
        self, pcm: bytes, *, rate: int = 24000, channels: int = 1, sample_width: int = 2
    ) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Gemini TTS API.
        Always returns WAV bytes (RIFF/WAVE).
        """
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model='gemini-2.5-flash-preview-tts',
            contents=[text],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                response_modalities=['AUDIO'],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            ),
        )

        # Validate response
        if not response or not response.candidates:
            return b''
        content = response.candidates[0].content
        if not content or not content.parts:
            return b''
        part = content.parts[0]
        inline = getattr(part, 'inline_data', None)
        if not inline or not inline.data:
            return b''

        data = bytes(inline.data)
        mime = (getattr(inline, 'mime_type', '') or '').lower()

        # If already a WAV container, return as-is
        if data[:4] == b'RIFF' and b'WAVE' in data[8:16]:
            return data
        if 'wav' in mime:
            return data

        # Otherwise assume raw PCM 16-bit mono 24 kHz and wrap into WAV
        try:
            return self._wrap_pcm_as_wav(data, rate=24000, channels=1, sample_width=2)
        except Exception:
            return b''

    async def transcribe_bytes(self, audio_bytes: bytes) -> str:
        # Send to LLM with audio transcription prompt
        contents: list[Any] = [
            'Transcribe this audio clip, it will be in french. Do not any additonal comments.',
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/wav',
            ),
        ]

        # Collect response
        transcription = await self.send_request(contents)  # type: ignore[arg-type]
        return transcription.strip()

    async def close(self) -> None:
        """
        Close the client if needed (placeholder for compatibility).
        """
        pass

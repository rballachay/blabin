import asyncio
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

    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Gemini TTS API.
        Returns WAV audio bytes.
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
        # Add safety checks for None values
        if (
            response.candidates
            and len(response.candidates) > 0
            and response.candidates[0].content
            and response.candidates[0].content.parts
            and len(response.candidates[0].content.parts) > 0
            and response.candidates[0].content.parts[0].inline_data
            and response.candidates[0].content.parts[0].inline_data.data
        ):
            return bytes(response.candidates[0].content.parts[0].inline_data.data)
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

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
            model='gemini-2.0-flash', contents=prompt
        )
        return response.text

    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Gemini TTS API.
        Returns WAV audio bytes.
        """
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model='gemini-2.5-flash-preview-tts',
            contents=[text],
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
        return bytes(response.candidates[0].content.parts[0].inline_data.data)

    async def close(self) -> None:
        """
        Close the client if needed (placeholder for compatibility).
        """
        pass

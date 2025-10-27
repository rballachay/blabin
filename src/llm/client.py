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

    async def get_speaker_name(self, audio_bytes: bytes) -> str:
        """
        Use Gemini to identify if this is a known speaker.
        Returns (is_known, speaker_name or None).
        """
        prompt = [
            {
                'role': 'user',
                'parts': [
                    {'inline_data': {'mime_type': 'audio/wav', 'data': audio_bytes}},
                    {
                        'text': (
                            'Please analyze this voice, and determine if they say their name. If so,'
                            "tell me their name. If not, just say 'unknown'. "
                            "Format: either 'unknown' or the name only."
                        )
                    },
                ],
            }
        ]

        response = await self.client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        text = response.text.strip().lower()

        return text

    async def is_confirmation(self, text: str) -> bool:
        """
        Check if the given text is a confirmation (yes) response.
        """
        prompt = [
            {
                'role': 'user',
                'parts': [
                    {'text': text},
                    {
                        'text': (
                            'You are a conversation assistant. '
                            'Analyze the text and determine if the speaker is confirming yes or no.'
                            "Respond with exactly 'NONE' if it's not a confirmation, "
                            'or else respond with exactly "YES" or "NO".'
                        ),
                    },
                ],
            },
        ]
        response = await self.client.aio.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        )

        text = response.text.strip().upper()

        if text == 'YES':
            return True
        return False

    async def transcribe_bytes(self, audio_bytes: bytes) -> str:
        # Send to LLM with audio transcription prompt
        prompt = [
            {
                'role': 'user',
                'parts': [
                    {
                        'inline_data': {
                            'mime_type': 'audio/wav',
                            'data': audio_bytes,
                        }
                    },
                    {
                        'text': 'What is being said in this audio? Only return the transcription, no other text.'
                    },
                ],
            }
        ]

        # Collect response
        transcription = await self.send_request(prompt)
        return transcription.strip()

    async def close(self) -> None:
        """
        Close the client if needed (placeholder for compatibility).
        """
        pass

from collections.abc import AsyncGenerator

from google import genai


class AsyncLLMClient:
    def __init__(self, api_key: str):
        """
        Initializes the Gemini client with the provided API key.
        """
        self.client = genai.Client(api_key=api_key)

    async def send_request(self, prompt: str) -> AsyncGenerator[str]:
        """
        Stream the Gemini LLM response as it arrives.
        Yields text chunks from the server.
        """
        async for chunk in await self.client.aio.models.generate_content_stream(
            model='gemini-2.0-flash', contents=prompt
        ):
            yield chunk.text

    async def close(self) -> None:
        """
        Close the client if needed (placeholder for compatibility).
        """
        pass  # genai.Client() does not need explicit closing

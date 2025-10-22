from collections.abc import AsyncGenerator

import httpx


class AsyncLLMClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={
                'x-goog-api-key': self.api_key,
                'Content-Type': 'application/json',
            }
        )

    async def send_request(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream the Gemini LLM response as it arrives (SSE-style).
        Yields text chunks from the server.
        """
        payload = {'contents': [{'parts': [{'text': prompt}]}]}

        async with self.client.stream('POST', self.api_url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield line

    async def close(self) -> None:
        await self.client.aclose()

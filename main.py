# entrypoint.py
import asyncio
import os

from dotenv import load_dotenv

from src.llm_client.client import AsyncLLMClient

load_dotenv()

API_KEY = str(os.getenv('LLM_API_KEY'))
API_URL = str(os.getenv('LLM_API_URL'))


async def main() -> None:
    client = AsyncLLMClient(api_url=API_URL, api_key=API_KEY)
    try:
        async for chunk in client.send_request('Hello!'):
            print(chunk, end='', flush=True)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())

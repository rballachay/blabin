# entrypoint.py
import asyncio
import os

from dotenv import load_dotenv

from src.llm_client.client import AsyncLLMClient

load_dotenv()

API_KEY = str(os.getenv('API_KEY'))


async def main() -> None:
    client = AsyncLLMClient(api_key=API_KEY)
    try:
        async for chunk in client.send_request('Bonjour juste pour tester'):
            print(chunk, end='', flush=True)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())

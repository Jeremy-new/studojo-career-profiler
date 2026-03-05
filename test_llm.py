import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import time

load_dotenv()

async def test_llm():
    print("Testing Azure OpenAI connection...")
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    )
    
    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            messages=[{"role": "user", "content": "Say hello world"}],
            timeout=10.0
        )
        print(f"Success! Took {time.time() - start:.2f}s")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"FAILED after {time.time() - start:.2f}s: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm())

"""
Test script for HuggingFace TGI endpoint (OpenAI-compatible).
Validates streaming translation works before integrating into agent.py.
"""
import asyncio
from openai import AsyncOpenAI

# Hardcoded for testing
HF_ENDPOINT = "https://zhr3wzqjdk9vxlby.us-east-1.aws.endpoints.huggingface.cloud/v1"
HF_TOKEN = "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"

# Test Arabic text (common phrase)
TEST_ARABIC = "مرحبا كيف حالك اليوم"


async def test_streaming():
    """Test streaming translation from Arabic to English."""
    print(f"Endpoint: {HF_ENDPOINT}")
    print(f"Testing with: {TEST_ARABIC}\n")
    
    client = AsyncOpenAI(
        base_url=HF_ENDPOINT,
        api_key=HF_TOKEN,
    )
    
    print("--- Streaming Response ---")
    result = ""
    
    stream = await client.chat.completions.create(
        model="tgi",  # Standard TGI model name
        messages=[
            {"role": "system", "content": "Translate to English. Return only the translation."},
            {"role": "user", "content": TEST_ARABIC}
        ],
        stream=True,
        max_tokens=256,
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            result += token
    
    print("\n--- End ---")
    print(f"\nFull result: {result}")
    return result


async def test_non_streaming():
    """Test non-streaming for comparison."""
    print("\n--- Non-Streaming Test ---")
    
    client = AsyncOpenAI(
        base_url=HF_ENDPOINT,
        api_key=HF_TOKEN,
    )
    
    response = await client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "Translate to English. Return only the translation."},
            {"role": "user", "content": TEST_ARABIC}
        ],
        stream=False,
        max_tokens=256,
    )
    
    result = response.choices[0].message.content
    print(f"Result: {result}")
    return result


async def main():
    try:
        await test_streaming()
        await test_non_streaming()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())


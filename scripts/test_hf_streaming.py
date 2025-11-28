"""
Interactive test script for HuggingFace TGI SSE streaming.
Uses httpx (not OpenAI SDK) to match the reference implementation.

Run:
  uv run python scripts/test_hf_streaming.py

Then type Arabic text and watch the translation stream token-by-token.
"""
import asyncio
import json
import time
import sys

import httpx

# Hardcoded HF endpoint (same as reference)
HF_ENDPOINT = "https://zhr3wzqjdk9vxlby.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"
HF_API_KEY = "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"
HF_MODEL = "tgi"  # or "tencent/Hunyuan-MT-7B" depending on endpoint config


async def stream_translation(arabic_text: str, target_lang: str = "English"):
    """
    Stream translation from HF endpoint using httpx SSE.
    Prints tokens as they arrive (typing effect).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_API_KEY}",
    }
    
    prompt = f"Translate to {target_lang}:\n\n{arabic_text}"
    
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 512,
    }
    
    t0 = time.time()
    first_token_time = None
    acc = ""
    token_count = 0
    
    print(f"\n🔄 Streaming from: {HF_ENDPOINT}")
    print(f"📝 Input: {arabic_text}")
    print(f"🎯 Target: {target_lang}")
    print("-" * 50)
    print("📤 Response: ", end="", flush=True)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", HF_ENDPOINT, headers=headers, json=payload) as resp:
                status = resp.status_code
                if status != 200:
                    body = await resp.aread()
                    print(f"\n❌ HTTP {status}: {body.decode()}")
                    return
                
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    
                    s = line.strip()
                    if not s:
                        continue
                    
                    # Parse SSE format: "data: {...}"
                    if s.startswith("data:"):
                        s = s[len("data:"):].strip()
                    
                    if s == "[DONE]":
                        break
                    
                    # Skip non-JSON lines
                    if not (s.startswith("{") and s.endswith("}")):
                        continue
                    
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    
                    # Extract delta content
                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    
                    delta = choices[0].get("delta", {}).get("content")
                    if not delta:
                        continue
                    
                    # Track timing
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = int((first_token_time - t0) * 1000)
                        # Don't print TTFT inline, save for summary
                    
                    # Accumulate and print token
                    acc += delta
                    token_count += 1
                    print(delta, end="", flush=True)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Summary
    total_ms = int((time.time() - t0) * 1000)
    ttft_ms = int((first_token_time - t0) * 1000) if first_token_time else 0
    
    print("\n" + "-" * 50)
    print(f"✅ Complete!")
    print(f"   Tokens: {token_count}")
    print(f"   TTFT: {ttft_ms}ms")
    print(f"   Total: {total_ms}ms")
    print(f"   Full text: {acc}")
    
    return acc


async def interactive_loop():
    """Interactive REPL for testing translations."""
    print("=" * 60)
    print("🌐 HuggingFace MT Streaming Test (httpx SSE)")
    print("=" * 60)
    print("Type Arabic text to translate, or 'q' to quit.")
    print("Default target: English (type 'lang:Spanish' to change)")
    print()
    
    target_lang = "English"
    
    while True:
        try:
            user_input = input(f"[{target_lang}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "q":
            print("👋 Bye!")
            break
        
        # Language switch command
        if user_input.lower().startswith("lang:"):
            target_lang = user_input[5:].strip() or "English"
            print(f"🎯 Target language set to: {target_lang}")
            continue
        
        # Stream translation
        await stream_translation(user_input, target_lang)
        print()


async def quick_test():
    """Quick automated test with sample Arabic."""
    samples = [
        "مرحبا كيف حالك اليوم",
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
    ]
    
    print("=" * 60)
    print("🧪 Quick Test Mode")
    print("=" * 60)
    
    for sample in samples:
        await stream_translation(sample)
        print("\n")
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    # Check for --quick flag
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(interactive_loop())


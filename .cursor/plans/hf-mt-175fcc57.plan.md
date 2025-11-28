<!-- 175fcc57-c886-4778-86f8-62fe61c47825 69036e51-0ec2-424c-b540-6ebfdec470ea -->
# TTS Prosody Continuity + Voice Change

## Overview

Refactor TTS architecture to maintain prosody continuity across utterances using Cartesia's `context_id` mechanism. Switch to a male multilingual voice for Sonic-3.

## Part 1: Voice Selection

**Selected Voice:** Blake

**ID:** `a167e0f3-df7e-4d52-a9c3-f949145efdab`

**Description:** Energetic American adult male

Sonic-3 supports 40+ languages including `ar` (Arabic), `hi` (Hindi), `en`, `ur` (Urdu - via `hi`). Voice works cross-lingually.

**Note:** Urdu (`ur`) is NOT in Sonic-3's official language list. Consider using `hi` (Hindi) for Urdu content since they share the same script variant.

## Part 2: Architecture Change

### Current (No Prosody Continuity)

```python
# synthesize_and_play() - called per utterance
tts = create_tts(language)     # NEW TTS each time
tts_stream = tts.stream()      # NEW context_id each time
tts_stream.push_text(text)
tts_stream.end_input()         # Context dies
```

### New (Prosody Preserved)

```python
# tts_consumer() - ONE stream per language for session lifetime
tts = create_tts(language)
tts_stream = tts.stream()      # ONE context_id for entire session

while True:
    job = await queue.get()
    if job is None:            # Shutdown sentinel
        break
    text, pair_id = job
    tts_stream.push_text(text)
    tts_stream.flush()         # Mark segment boundary, context STAYS ALIVE
    async for audio in tts_stream:
        await audio_source.capture_frame(audio.frame)

tts_stream.end_input()         # Context dies at session end
await tts_stream.aclose()
```

## Part 3: Code Changes in [src/agent.py](src/agent.py)

### 1. Update `create_tts()` to include voice

```python
MALE_VOICE_ID = "a167e0f3-df7e-4d52-a9c3-f949145efdab"  # Blake

def create_tts(language: str) -> cartesia.TTS:
    return cartesia.TTS(
        model="sonic-3",
        language=language,
        voice=MALE_VOICE_ID,
    )
```

### 2. Rewrite `tts_consumer()` for persistent stream

```python
async def tts_consumer(
    queue: asyncio.Queue,
    audio_source: rtc.AudioSource,
    language: str,
):
    logger.info(f"TTS consumer started for {language}")
    tts = create_tts(language)
    tts_stream = tts.stream()
    
    try:
        while True:
            job = await queue.get()
            if job is None:
                logger.info(f"TTS consumer shutting down for {language}")
                break
            
            text, pair_id = job
            try:
                logger.info(f"[TTS START] pair={pair_id} lang={language}")
                tts_stream.push_text(text)
                tts_stream.flush()
                async for audio in tts_stream:
                    await audio_source.capture_frame(audio.frame)
                logger.info(f"[TTS DONE] pair={pair_id} lang={language}")
            except Exception as e:
                logger.error(f"TTS failed for pair={pair_id} lang={language}: {e}")
            finally:
                queue.task_done()
    finally:
        tts_stream.end_input()
        await tts_stream.aclose()
        await tts.aclose()
```

### 3. Remove `synthesize_and_play()` function (no longer needed)

## Part 4: Deploy

```bash
lk agent deploy
```

## Risk Notes

- If TTS stream errors mid-session, entire language lane dies. May need reconnection logic.
- Cartesia contexts expire 1 second after last audio. Long pauses may reset prosody anyway.

### To-dos

- [ ] Create scripts/test_hf_endpoint.py to validate HF streaming
- [ ] Add MT_ENDPOINT and MT_API_KEY constants to agent.py
- [ ] Uncomment and adapt translate_mt, mt_ticker, run_mt_stream
- [ ] Add MALE_VOICE_ID constant and update create_tts() to use it
- [ ] Rewrite tts_consumer() for persistent stream with flush() pattern
- [ ] Remove synthesize_and_play() function
- [ ] Deploy with lk agent deploy
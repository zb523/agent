<!-- 62ac927a-bd90-48c3-b51a-169628a0fce7 c32a4a81-bb4c-49eb-8612-709085e1d472 -->
# TTS Queue Serialization

## Problem

Multiple translations committing in quick succession cause concurrent `capture_frame()` calls on the same AudioSource, leading to interleaved/garbled audio.

## Solution

Add an `asyncio.Queue` per output language with a dedicated consumer task that processes TTS jobs sequentially. This follows the official LiveKit pattern from `elevenlabs_tts.py`.

---

## Implementation

### 1. Add TTS Queue Infrastructure

```python
# In transcription_agent(), after creating audio_sources:
tts_queues: dict[str, asyncio.Queue] = {}
tts_tasks: dict[str, asyncio.Task] = {}

for lang in OUTPUT_LANGUAGES:
    queue = asyncio.Queue()
    tts_queues[lang] = queue
    tts_tasks[lang] = asyncio.create_task(
        tts_consumer(queue, audio_sources[lang], lang)
    )
```

### 2. Add TTS Consumer Function

```python
async def tts_consumer(
    queue: asyncio.Queue,
    audio_source: rtc.AudioSource,
    language: str,
):
    """Process TTS jobs sequentially for a single language."""
    while True:
        job = await queue.get()
        if job is None:  # Shutdown signal
            break
        
        text, pair_id = job
        try:
            await synthesize_and_play(text, language, audio_source)
            logger.info(f"[TTS DONE] pair={pair_id} lang={language}")
        except Exception as e:
            logger.error(f"TTS failed for pair={pair_id} lang={language}: {e}")
        finally:
            queue.task_done()
```

### 3. Modify translate_and_publish()

Change from direct TTS call to queue submission:

```python
async def translate_and_publish(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
    target_lang: str,
    tts_queues: dict[str, asyncio.Queue],  # Changed from audio_sources
):
    try:
        translation = await translate_text(text, target_lang)
        
        # Publish translation text (immediate)
        await ctx.room.local_participant.send_text(...)
        logger.info(f"[TRANSLATED] pair={pair_id} lang={target_lang}: {translation}")
        
        # Queue TTS job (serialized playback)
        await tts_queues[target_lang].put((translation, pair_id))
        
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed: {e}")
```

### 4. Update process_stt_events() Call Site

```python
# Pass tts_queues instead of audio_sources
for lang in OUTPUT_LANGUAGES:
    asyncio.create_task(
        translate_and_publish(ctx, participant, committed, buffer.pair_id, lang, tts_queues)
    )
```

### 5. Cleanup on Agent Shutdown

```python
# In transcription_agent() finally block or on agent cancel:
for lang, queue in tts_queues.items():
    await queue.put(None)  # Signal shutdown
for task in tts_tasks.values():
    await task
```

---

## Files Changed

| File | Changes |

|------|---------|

| `src/agent.py` | Add queue infrastructure, consumer task, modify translate_and_publish |

| `scripts/local_test.py` | Mirror changes for local testing |

---

## Behavior After Fix

```
Timeline:
0s  - Commit 1 → Translation → Queue: [Job1] → Consumer plays Job1
6s  - Commit 2 → Translation → Queue: [Job1, Job2] → Consumer waits
10s - Job1 done → Consumer plays Job2 (no overlap)
```

### To-dos

- [ ] Add livekit-agents[speechmatics] to pyproject.toml
- [ ] Refactor agent.py to use standalone Speechmatics STT
- [ ] Implement text stream publishing for incremental transcripts
- [ ] Test end-to-end with room connection
- [ ] Add livekit-agents[speechmatics] to pyproject.toml
- [ ] Refactor agent.py to use standalone Speechmatics STT
- [ ] Implement text stream publishing for incremental transcripts
- [ ] Test end-to-end with room connection
- [ ] Implement TranscriptBuffer dataclass with find_commit_point() and commit() methods
- [ ] Update Speechmatics STT config to disable EOU (EndOfUtteranceMode.NONE)
- [ ] Rewrite process_stt_events() with buffer accumulation, change detection, and commit logic
- [ ] Add publish_buffer() and translate_and_publish() with pair_id attributes
- [ ] Handle stream end - publish incomplete buffer without translation
- [ ] Rewrite TranscriptBuffer with committed_idx and uncommitted property
- [ ] Update INTERIM_TRANSCRIPT handler to use uncommitted text only
- [ ] Update publish functions for new buffer structure
- [ ] Test with local_test.py to verify no duplicate commits
- [ ] Add livekit-agents[speechmatics] to pyproject.toml
- [ ] Refactor agent.py to use standalone Speechmatics STT
- [ ] Implement text stream publishing for incremental transcripts
- [ ] Test end-to-end with room connection
- [ ] Add livekit-agents[speechmatics] to pyproject.toml
- [ ] Refactor agent.py to use standalone Speechmatics STT
- [ ] Implement text stream publishing for incremental transcripts
- [ ] Test end-to-end with room connection
- [ ] Implement TranscriptBuffer dataclass with find_commit_point() and commit() methods
- [ ] Update Speechmatics STT config to disable EOU (EndOfUtteranceMode.NONE)
- [ ] Rewrite process_stt_events() with buffer accumulation, change detection, and commit logic
- [ ] Add publish_buffer() and translate_and_publish() with pair_id attributes
- [ ] Handle stream end - publish incomplete buffer without translation
- [ ] Update translation to use GPT-4.1 and make language-agnostic
- [ ] Add tts_queues and tts_tasks dictionaries in transcription_agent()
- [ ] Add tts_consumer() function for sequential TTS playback
- [ ] Change translate_and_publish() to queue TTS jobs instead of direct playback
- [ ] Add graceful shutdown for TTS consumer tasks
- [ ] Mirror queue changes in local_test.py
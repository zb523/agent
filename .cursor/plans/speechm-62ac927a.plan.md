<!-- 62ac927a-bd90-48c3-b51a-169628a0fce7 46b278ed-710a-4589-be54-78bcc6b89da8 -->
# Partial Machine Translation (MT)

## Summary

Add streaming MT translations for incomplete transcripts. Every 200ms, if the buffer has grown, send MT requests to a HuggingFace endpoint for all output languages in parallel. MT produces `status="incomplete"` translations; GPT produces `status="complete"` translations.

---

## Payload Schema Update

| Status | Type | Source | TTS? |

|--------|------|--------|------|

| `incomplete` | `transcript` | Speechmatics | No |

| `complete` | `transcript` | Speechmatics | No |

| `incomplete` | `translation` | HuggingFace MT | **No** |

| `complete` | `translation` | GPT-4.1 | **Yes** |

---

## MT Logic

```python
# Global 200ms ticker
EVERY 200ms:
    IF buffer.uncommitted has grown since last MT:
        FOR EACH lang in OUTPUT_LANGUAGES:
            IF lang's MT stream is NOT currently active:
                START streaming MT request for lang
```

### Rules

1. **Global 200ms tick** triggers check for all languages
2. **Per-language serialization** - only start MT if that language is free
3. **Streaming response** - MT streams tokens, publish when stream completes
4. **No TTS** for incomplete translations
5. **GPT wins** - when GPT response arrives, pair_id is done (no more MT)

---

## Implementation

### 1. Add MT Configuration

```python
# Placeholder - user will provide actual endpoint
MT_ENDPOINT = "https://your-hf-endpoint.huggingface.cloud"
MT_ENABLED = False  # Toggle to enable MT (commented out initially)
```

### 2. Add MT State Tracking

```python
@dataclass
class MTState:
    last_translated_text: str = ""  # Track what we last sent to MT
    active_streams: dict[str, bool] = field(default_factory=dict)  # per-language
```

### 3. Add MT Request Function

```python
async def translate_mt(text: str, target_lang: str) -> AsyncIterator[str]:
    """Stream MT translation from HuggingFace endpoint (OpenAI-compatible)."""
    # Placeholder - will use httpx or openai client with stream=True
    pass
```

### 4. Add Global 200ms Ticker

```python
async def mt_ticker(ctx, participant, buffer, mt_state):
    """Run MT every 200ms if buffer grew."""
    while True:
        await asyncio.sleep(0.2)  # 200ms
        
        if buffer.uncommitted and buffer.uncommitted != mt_state.last_translated_text:
            mt_state.last_translated_text = buffer.uncommitted
            
            for lang in OUTPUT_LANGUAGES:
                if not mt_state.active_streams.get(lang, False):
                    asyncio.create_task(
                        run_mt_stream(ctx, participant, buffer.uncommitted, buffer.pair_id, lang, mt_state)
                    )
```

### 5. Add MT Stream Handler

```python
async def run_mt_stream(ctx, participant, text, pair_id, lang, mt_state):
    """Run MT stream for a single language."""
    mt_state.active_streams[lang] = True
    try:
        result = ""
        async for token in translate_mt(text, lang):
            result += token
        
        # Publish incomplete translation
        await publish_translation(ctx, participant, result, pair_id, lang, status="incomplete")
    finally:
        mt_state.active_streams[lang] = False
```

### 6. Integrate into process_audio_track

Start the MT ticker task alongside the STT processing.

---

## Files Changed

| File | Changes |

|------|---------|

| `src/agent.py` | Add MT config, state, ticker, stream handler |

| `reference.md` | Update payload schema with incomplete translations |

---

## Environment Variables

```
HF_ENDPOINT=https://your-endpoint.huggingface.cloud  # Placeholder
HF_API_KEY=...  # If needed
```

---

## Notes

- MT code will be **commented out** initially until endpoint is configured
- Keep all code in `agent.py` for now (split later)
- No TTS for incomplete translations

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
- [ ] Add language config and update payloads with language attribute
- [ ] Implement parallel translation to es/fr/ar on commit
- [ ] Add Cartesia TTS and create audio tracks per language
- [ ] Synthesize and publish TTS audio after translation
- [ ] Update local_test.py to mirror agent.py changes
- [ ] Commit and push changes to GitHub
- [ ] Add MT_ENDPOINT placeholder and MT_ENABLED toggle
- [ ] Add MTState dataclass for tracking last_translated_text and active_streams
- [ ] Add translate_mt() function (OpenAI-compatible streaming)
- [ ] Add mt_ticker() coroutine with 200ms interval
- [ ] Add run_mt_stream() for per-language MT handling
- [ ] Integrate MT ticker into process_audio_track()
- [ ] Update publish functions to handle status=incomplete translations
- [ ] Update reference.md with new payload schema
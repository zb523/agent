<!-- 62ac927a-bd90-48c3-b51a-169628a0fce7 f0fb94d6-0a5d-4989-9ed9-7b377b190a5e -->
# Multi-Language Translation + TTS Pipeline

## Phase 1: GPT-4.1 Translation Model

Update `translate_to_spanish()` to use GPT-4.1 and make it language-agnostic.

```python
# agent.py changes
async def translate_text(text: str, target_lang: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"Translate to {LANG_NAMES[target_lang]}. Return only the translation."},
            {"role": "user", "content": text}
        ],
    )
    return response.choices[0].message.content.strip()
```

## Phase 2: Multi-Language Configuration

Add language configuration and update payloads to include `language` attribute.

```python
# Constants
INPUT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ["es", "fr", "ar"]
LANG_NAMES = {"es": "Spanish", "fr": "French", "ar": "Arabic", "en": "English"}
```

Update all `send_text()` calls to include `"language"` attribute per reference.md.

## Phase 3: Parallel Translation

On commit, fire N parallel translation tasks (one per output language).

```python
# In process_stt_events(), on commit:
if committed:
    await publish_complete(ctx, participant, committed, buffer.pair_id)
    
    # Parallel translation to all output languages
    for lang in OUTPUT_LANGUAGES:
        asyncio.create_task(
            translate_and_publish(ctx, participant, committed, buffer.pair_id, lang)
        )
```

Update `translate_and_publish()` to accept target language:

```python
async def translate_and_publish(ctx, participant, text: str, pair_id: int, target_lang: str):
    try:
        translation = await translate_text(text, target_lang)
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "translation",
            "language": target_lang,
            "participant_identity": participant.identity,
            "original_text": text,
        }
        await ctx.room.local_participant.send_text(translation, topic="lk.transcription", attributes=attributes)
        
        # Phase 4: TTS will be added here
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed for pair={pair_id}: {e}")
```

## Phase 4: Cartesia TTS Setup

Add Cartesia plugin and create audio tracks per language.

```python
from livekit.plugins import cartesia

# Create TTS instances per language (reusable)
def create_tts(language: str) -> cartesia.TTS:
    return cartesia.TTS(
        model="sonic-3",
        language=language,
    )
```

Create and publish audio tracks on agent start:

```python
# In transcription_agent(), after connect:
audio_sources: dict[str, rtc.AudioSource] = {}
audio_tracks: dict[str, rtc.LocalAudioTrack] = {}

for lang in OUTPUT_LANGUAGES:
    source = rtc.AudioSource(24000, 1)  # Cartesia outputs 24kHz mono
    track = rtc.LocalAudioTrack.create_audio_track(f"tts-{lang}", source)
    await ctx.room.local_participant.publish_track(track)
    audio_sources[lang] = source
    audio_tracks[lang] = track
```

## Phase 5: TTS on Translation Complete

After translation, synthesize and publish audio.

```python
async def translate_and_publish(ctx, participant, text: str, pair_id: int, target_lang: str, audio_sources: dict):
    try:
        translation = await translate_text(text, target_lang)
        
        # Publish text
        await ctx.room.local_participant.send_text(translation, topic="lk.transcription", attributes={...})
        
        # TTS: synthesize and publish audio
        await synthesize_and_play(translation, target_lang, audio_sources[target_lang])
        
    except Exception as e:
        logger.error(f"Translation/TTS to {target_lang} failed: {e}")

async def synthesize_and_play(text: str, language: str, audio_source: rtc.AudioSource):
    tts = create_tts(language)
    tts_stream = tts.stream()
    tts_stream.push_text(text)
    tts_stream.end_input()
    
    async for audio in tts_stream:
        await audio_source.capture_frame(audio.frame)
```

## Phase 6: Update local_test.py

Sync local_test.py with agent.py changes for local testing (prints instead of publishes).

## Phase 7: Testing with LiveKit Playground

1. Run agent: `uv run src/agent.py dev`
2. Open https://agents-playground.livekit.io/
3. Connect with microphone
4. Verify: transcripts, translations (es/fr/ar), and TTS audio tracks

## File Changes Summary

| File | Changes |

|------|---------|

| `agent.py` | GPT-4.1, multi-lang, parallel translation, TTS tracks |

| `local_test.py` | Mirror agent.py logic for local testing |

| `reference.md` | Already done - payload schemas |

| `pyproject.toml` | Add `livekit-agents[cartesia]` if not present |

| `.env.local` | Ensure CARTESIA_API_KEY is set |

## Environment Variables Required

```
OPENAI_API_KEY=...
SPEECHMATICS_API_KEY=...
CARTESIA_API_KEY=...
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
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
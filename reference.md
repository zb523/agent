# Payload Reference

All messages are published to the LiveKit room via `room.local_participant.send_text()` on topic `lk.transcription`.

## Attributes Schema

| Attribute | Type | Description |
|-----------|------|-------------|
| `pair_id` | string | Sequential ID linking a transcript chunk to its translations (e.g., `"1"`, `"2"`) |
| `status` | string | `"incomplete"` (still accumulating) or `"complete"` (committed) |
| `type` | string | `"transcript"` or `"translation"` |
| `language` | string | ISO 639-1 language code (e.g., `"en"`, `"es"`, `"fr"`, `"ar"`) |
| `participant_identity` | string | Identity of the speaking participant |

---

## Payload Types

### 1. Transcript (Incomplete)

Published as text accumulates, before commit.

```python
# Attributes
{
    "pair_id": "1",
    "status": "incomplete",
    "type": "transcript",
    "language": "en",  # Source language
    "participant_identity": "user-123",
}
# Text body
"Hello how are you"
```

### 2. Transcript (Complete)

Published when commit conditions are met (5+ words + punctuation).

```python
# Attributes
{
    "pair_id": "1",
    "status": "complete",
    "type": "transcript",
    "language": "en",
    "participant_identity": "user-123",
}
# Text body
"Hello how are you today,"
```

### 3. Translation

Published after LLM translation completes. One per target language.

```python
# Attributes
{
    "pair_id": "1",
    "status": "complete",
    "type": "translation",
    "language": "es",  # Target language
    "participant_identity": "user-123",
    "original_text": "Hello how are you today,",
}
# Text body
"Hola, ¿cómo estás hoy?"
```

---

## Example Flow

Speaker says: "Hello how are you today, I wanted to ask you something."

```
1. [INCOMPLETE] pair=1, type=transcript, lang=en, text="Hello how are"
2. [INCOMPLETE] pair=1, type=transcript, lang=en, text="Hello how are you today,"
3. [COMPLETE]   pair=1, type=transcript, lang=en, text="Hello how are you today,"
4. [COMPLETE]   pair=1, type=translation, lang=es, text="Hola, ¿cómo estás hoy?"
5. [COMPLETE]   pair=1, type=translation, lang=fr, text="Bonjour, comment ça va aujourd'hui?"
6. [COMPLETE]   pair=1, type=translation, lang=ar, text="مرحبًا، كيف حالك اليوم؟"
7. [INCOMPLETE] pair=2, type=transcript, lang=en, text="I wanted to ask"
8. [INCOMPLETE] pair=2, type=transcript, lang=en, text="I wanted to ask you something."
9. [COMPLETE]   pair=2, type=transcript, lang=en, text="I wanted to ask you something."
10. ... translations for pair=2 arrive
```

---

## Configuration

```python
# Hardcoded for now
INPUT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ["es", "fr", "ar"]  # Spanish, French, Arabic
```

---

## TTS Audio Tracks

Audio is published via LiveKit audio tracks (not data messages). One track per output language:

| Track Name | Language | Sample Rate |
|------------|----------|-------------|
| `tts-es` | Spanish | 24kHz mono |
| `tts-fr` | French | 24kHz mono |
| `tts-ar` | Arabic | 24kHz mono |

Each translation triggers TTS synthesis via Cartesia Sonic 3, which streams audio frames to the corresponding track.

```python
# Track creation in agent.py
for lang in OUTPUT_LANGUAGES:
    source = rtc.AudioSource(24000, 1)  # 24kHz mono
    track = rtc.LocalAudioTrack.create_audio_track(f"tts-{lang}", source)
    await ctx.room.local_participant.publish_track(track)
```

---

## Environment Variables

```
SPEECHMATICS_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```


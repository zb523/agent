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

## Future: TTS Payloads

When TTS is added (Cartesia Sonic 3), audio will be published separately:

```python
# Attributes (planned)
{
    "pair_id": "1",
    "type": "tts_audio",
    "language": "es",
    "participant_identity": "user-123",
}
# Binary audio data published via audio track or byte stream
```


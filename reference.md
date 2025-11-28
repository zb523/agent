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

## Status/Type Matrix

| Status | Type | Source | TTS? |
|--------|------|--------|------|
| `incomplete` | `transcript` | Speechmatics | No |
| `complete` | `transcript` | Speechmatics | No |
| `incomplete` | `translation` | HuggingFace MT | **No** |
| `complete` | `translation` | GPT-4.1 | **Yes** |

- **incomplete translations** are produced by MT every 200ms (when enabled)
- **complete translations** are produced by GPT-4.1 on commit and trigger TTS
- When GPT completes a pair, no further MT translations are sent for that pair

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

### 3. Translation (Incomplete) - MT

Published every 200ms from HuggingFace MT endpoint (when enabled). No TTS.

```python
# Attributes
{
    "pair_id": "1",
    "status": "incomplete",
    "type": "translation",
    "language": "es",  # Target language
    "participant_identity": "user-123",
    "original_text": "Hello how are you",  # Source text at time of MT
}
# Text body
"Hola, ¿cómo estás?"
```

### 4. Translation (Complete) - GPT

Published after GPT-4.1 translation completes. Triggers TTS.

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

### Without MT (default)
```
1. [INCOMPLETE] pair=1, type=transcript, lang=en, text="Hello how are"
2. [INCOMPLETE] pair=1, type=transcript, lang=en, text="Hello how are you today,"
3. [COMPLETE]   pair=1, type=transcript, lang=en, text="Hello how are you today,"
4. [COMPLETE]   pair=1, type=translation, lang=es, text="Hola, ¿cómo estás hoy?"
5. [COMPLETE]   pair=1, type=translation, lang=fr, text="Bonjour, comment ça va aujourd'hui?"
6. [COMPLETE]   pair=1, type=translation, lang=ar, text="مرحبًا، كيف حالك اليوم؟"
7. [INCOMPLETE] pair=2, type=transcript, lang=en, text="I wanted to ask"
...
```

### With MT enabled (MT_ENABLED=True)
```
1.  [INCOMPLETE] pair=1, type=transcript,   lang=en, text="Hello how are"
2.  [INCOMPLETE] pair=1, type=translation,  lang=es, text="Hola cómo" (MT @200ms)
3.  [INCOMPLETE] pair=1, type=transcript,   lang=en, text="Hello how are you today,"
4.  [INCOMPLETE] pair=1, type=translation,  lang=es, text="Hola, ¿cómo estás hoy?" (MT @400ms)
5.  [COMPLETE]   pair=1, type=transcript,   lang=en, text="Hello how are you today,"
6.  [COMPLETE]   pair=1, type=translation,  lang=es, text="Hola, ¿cómo estás hoy?" (GPT wins, TTS)
7.  [INCOMPLETE] pair=2, type=transcript,   lang=en, text="I wanted"
...
```

Note: Once GPT publishes `complete` translation, MT stops for that pair_id.

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

## Quran Tagging & TTS Handling

When `input_lang="ar"`, detected Quranic verses are wrapped in `[QURAN]` tags in the translation output. These tags are **stripped before TTS** so the audio sounds natural.

### Tag Format

```
[QURAN ref="Surah:Verse"]canonical translation text[/QURAN]
[QURAN ref="Surah:Start-End"]multi-verse span[/QURAN]
```

### Examples

| GPT Output (Published) | TTS Input (Spoken) |
|------------------------|-------------------|
| `Allah says: [QURAN ref="1:1"]In the name of Allah, the Most Gracious, the Most Merciful.[/QURAN]` | `Allah says: In the name of Allah, the Most Gracious, the Most Merciful.` |
| `[QURAN ref="112:1-2"]Say, "He is Allah, [who is] One, Allah, the Eternal Refuge."[/QURAN]` | `Say, "He is Allah, who is One, Allah, the Eternal Refuge."` |
| `The imam recited [QURAN ref="2:255"]Allah - there is no deity except Him[/QURAN] and continued.` | `The imam recited Allah - there is no deity except Him and continued.` |

### Strip Function

```python
def strip_tts_tags(text: str) -> str:
    """Strip [QURAN ref="..."] tags and [bracket] words for clean TTS."""
    text = re.sub(r'\[QURAN ref="[^"]*"\]', '', text)  # Remove opening tags
    text = re.sub(r'\[/QURAN\]', '', text)             # Remove closing tags
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)        # [word] -> word
    return re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
```

### Flow

```
Arabic Speech → STT → Quran Detection → GPT Translation (with tags) → Published
                                                                    ↓
                                                              strip_tts_tags()
                                                                    ↓
                                                              TTS (clean audio)
```

**Key points:**
- Tags are in the **published text** (for UI highlighting)
- Tags are **stripped before TTS** (natural speech)
- Bracketed words like `[who is]` also get unbracketed for TTS

---

## Environment Variables

```
SPEECHMATICS_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# Optional: HuggingFace MT endpoint (for incomplete translations)
HF_ENDPOINT=https://your-endpoint.huggingface.cloud  # Placeholder
HF_API_KEY=...
```


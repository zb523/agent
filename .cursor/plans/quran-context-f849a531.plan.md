<!-- f849a531-d2d2-4569-a74b-642d33daa100 b03f9c01-a4fb-409d-9dbe-35918f045e7b -->
# Buffer Logic Fix + Local Testing

## Changes to `src/agent.py`

### 1. Remove MAX_WORDS constant

Delete line 118:

```python
MAX_WORDS = 25  # Soft-commit threshold when buffer gets too long
```

### 2. Fix TokenBuffer.find_hard_punct_idx()

Replace the method to check for ANY punctuation at index >= 4 (5th word):

```python
def find_punct_idx(self) -> Optional[int]:
    """Find first token ending with any punctuation after MIN_WORDS tokens."""
    PUNCT_CHARS = '.?!,;:،؟'
    for i, tok in enumerate(self.tokens):
        if i >= MIN_WORDS - 1 and tok and tok[-1] in PUNCT_CHARS:
            return i + 1
    return None
```

Rename method from `find_hard_punct_idx` to `find_punct_idx` (no hard/soft distinction).

### 3. Remove find_soft_punct_idx()

Delete the entire method (lines 164-169) - no longer needed.

### 4. Simplify process_stt_events commit logic

Replace the commit check section:

```python
# Check for commit point (any punct after MIN_WORDS)
punct_idx = buffer.find_punct_idx()
if punct_idx:
    committed, _ = buffer.commit_at(punct_idx)
    await handle_commit(committed)
```

Remove the MAX_WORDS soft-commit block entirely.

---

## Changes to `scripts/local_test.py`

### 1. Add pyaudio as dev dependency

```bash
uv add pyaudio --dev
```

### 2. Replace TranscriptBuffer with TokenBuffer

Update the buffer class to match agent.py:

```python
@dataclass
class TokenBuffer:
    pair_id: int = 1
    tokens: List[str] = field(default_factory=list)
    last_published_text: str = ""
    
    @property
    def text(self) -> str:
        return " ".join(self.tokens)
    
    def append(self, token: str):
        self.tokens.append(token)
    
    def find_punct_idx(self) -> Optional[int]:
        PUNCT_CHARS = '.?!,;:،؟'
        for i, tok in enumerate(self.tokens):
            if i >= MIN_WORDS - 1 and tok and tok[-1] in PUNCT_CHARS:
                return i + 1
        return None
    
    def commit_at(self, idx: int) -> tuple[str, List[str]]:
        committed = " ".join(self.tokens[:idx])
        self.tokens = self.tokens[idx:]
        return committed, self.tokens
```

### 3. Change STT config to match agent.py

```python
stt = speechmatics.STT(
    language=INPUT_LANGUAGE,
    enable_partials=False,  # Token-based like agent.py
    end_of_utterance_mode=EndOfUtteranceMode.NONE,
)
```

### 4. Update event handler for FINAL tokens

Process FINAL_TRANSCRIPT instead of INTERIM_TRANSCRIPT:

```python
elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
    if event.alternatives:
        token = event.alternatives[0].text.strip()
        if not token:
            continue
        
        buffer.append(token)
        current_text = buffer.text
        
        if current_text != buffer.last_published_text:
            publish_to_room(buffer.pair_id, current_text, "incomplete", "transcript", INPUT_LANGUAGE)
            buffer.last_published_text = current_text
        
        punct_idx = buffer.find_punct_idx()
        if punct_idx:
            committed, _ = buffer.commit_at(punct_idx)
            publish_to_room(buffer.pair_id, committed, "complete", "transcript", INPUT_LANGUAGE)
            # ... translation tasks ...
```

---

## Testing

Run locally with:

```bash
uv run python scripts/local_test.py
```

Verify:

1. Words appear one by one as "incomplete"
2. Commit happens at ANY punctuation after 5 words
3. No MAX_WORDS soft-commit behavior
4. Buffer accumulates indefinitely until punctuation

### To-dos

- [ ] Create src/retriever.py with QuranIndex, DirectIndex, detection functions
- [ ] Modify agent.py: add config, load indexes at startup, run detection after commit
- [ ] Replace translate_text with full legacy system prompt + verified candidates
- [ ] Add rapidfuzz dependency via uv add
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Create src/retriever.py with QuranIndex, DirectIndex, detection functions
- [ ] Modify agent.py: add config, load indexes at startup, run detection after commit
- [ ] Replace translate_text with full legacy system prompt + verified candidates
- [ ] Add rapidfuzz dependency via uv add
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Uncomment httpx import at line 11
- [ ] Uncomment and update save_to_worker() with auth header and error handling
- [ ] Pass session_id from config to translate_and_publish()
- [ ] Call save_to_worker() after publish_translation in translate_and_publish()
- [ ] Call save_to_worker() in shutdown flush block
- [ ] Verify httpx is in dependencies, add if missing
- [ ] Add retry loop around STT stream creation with 3 attempts
- [ ] Run lk agent deploy
- [ ] Create src/retriever.py with QuranIndex, DirectIndex, detection functions
- [ ] Modify agent.py: add config, load indexes at startup, run detection after commit
- [ ] Replace translate_text with full legacy system prompt + verified candidates
- [ ] Add rapidfuzz dependency via uv add
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Create src/retriever.py with QuranIndex, DirectIndex, detection functions
- [ ] Modify agent.py: add config, load indexes at startup, run detection after commit
- [ ] Replace translate_text with full legacy system prompt + verified candidates
- [ ] Add rapidfuzz dependency via uv add
- [ ] Add strip_tts_tags() function with regex for QURAN tags and brackets
- [ ] Apply strip_tts_tags() before TTS queue in translate_and_publish()
- [ ] Add CLI test cases for tag stripping edge cases
- [ ] Add agent_name="khutbah-interpreter" to @server.rtc_session decorator
- [ ] Parse ctx.job.metadata with fallback to hardcoded defaults
- [ ] Move config and Quran loading inside transcription_agent() function
- [ ] Add shutdown hook with buffer flush + force translate + TTS
- [ ] Add commented save_to_worker() placeholder for future Worker integration
- [ ] Uncomment httpx import at line 11
- [ ] Uncomment and update save_to_worker() with auth header and error handling
- [ ] Pass session_id from config to translate_and_publish()
- [ ] Call save_to_worker() after publish_translation in translate_and_publish()
- [ ] Call save_to_worker() in shutdown flush block
- [ ] Verify httpx is in dependencies, add if missing
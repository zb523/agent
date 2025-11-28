Here is a concrete plan.

I will first outline the steps in plain English.

Then I will show where in `transcription_agent` you change things, with focused +/- style edits.

I will treat:

- **Truth A** = `local_test_rt.py` (Speechmatics RT + `transcription.py` helpers)
- **Truth B** = this `khutbah-interpreter` agent file you just pasted

We keep Truth B for everything except transcription.

We replace the transcription parts with Truth A behavior.

---

## High level plan

1. **Import and configure the RT client and transcription helpers**

   - Bring in Speechmatics RT types.
   - Bring in `current_buffer_text`, `find_strict_eos_idx`, `find_relaxed_tail_idx`, `decide_commit_split` from `transcription.py`.
   - Set a default RT URL if not present.
   - Introduce a commit constant `MIN_WORDS_BEFORE_COMMIT = 10` to match the RT test.

2. **Make `TokenBuffer` a dumb container only**

   - Change `TokenBuffer.text` to use `current_buffer_text(self.tokens)` instead of `" ".join`.
   - Stop using `TokenBuffer.find_punct_idx` and `commit_at` for commit decisions.
   - You will still use `TokenBuffer.pair_id` and `tokens` as convenient storage.

3. **Replace plugin based STT with Speechmatics RT in `process_audio_track`**

   - Remove `speechmatics.STT(...)`, `stt_stream`, and all `SpeechEventType` handling.
   - Use `speechmatics.rt.AsyncClient` with `SMTranscriptionConfig` and `SMAudioFormat` like in local_test.
   - Make an `event_q` queue.
   - One coroutine reads audio frames from `rtc.AudioStream(track)` and calls `send_audio` on the RT client.
   - RT callbacks push messages into `event_q`.
   - Another coroutine consumes `event_q` and updates the buffer tokens exactly like `process_events()` in `local_test_rt.py`.

4. **Port the exact buffer and commit logic from `local_test_rt.py`**

For each `"final"` RT message:

   - Extract `new_tokens` from `msg["results"][..]["alternatives"][0]["content"]`.
   - If buffer is empty and there is no active pair, create a new pair id and set `buffer.pair_id` to that.
   - `buffer.tokens.extend(new_tokens)`.
   - `buf_text = current_buffer_text(buffer.tokens)`.
   - If `buf_text` is new, publish a partial transcript using your existing `publish_buffer` function.

Commit logic:

   - Compute `wc = number of tokens that have any alphanumeric characters`.

   - Only try to commit when `wc >= MIN_WORDS_BEFORE_COMMIT`.

   - Prefer hard punctuation:

     - `strict_idx = find_strict_eos_idx(buffer.tokens)`

If not `None` and `wc >= MIN_WORDS_BEFORE_COMMIT`, call `decide_commit_split` with `reason="HARD_PUNCT"`.

   - If there is no strict index, try relaxed comma tail:

     - `soft_idx = find_relaxed_tail_idx(buffer.tokens)`

If not `None` and `soft_idx >= MIN_WORDS_BEFORE_COMMIT`, call `decide_commit_split` with `reason="SOFT_COMMA_MAX"`.

   - `decide_commit_split` returns `commit_tokens`, `carry_tokens`, `allowed`.

   - If not `allowed` or `commit_tokens` is empty, do nothing.

   - Otherwise:

     - `commit_text = current_buffer_text(commit_tokens)`.
     - `carry_text = current_buffer_text(carry_tokens)` if not empty.

Then:

   - Publish TR final for `commit_text` with `publish_complete`.
   - Run Quran detection on `commit_text`.
   - For each output language, call `translate_and_publish` so you get TL and TTS.
   - Append `commit_text` to any history you want in this agent.

Pair rollover:

   - If `carry_text` exists:

     - Increment `buffer.pair_id` by 1.
     - Set `buffer.tokens = carry_tokens`.
     - Immediately publish a TR partial for `carry_text` with the new pair id.
     - Update `buffer.last_published_text`.

   - If there is no `carry_text`:

     - Clear `buffer.tokens`.
     - Increment `buffer.pair_id` by 1, or set it to the next id you prefer.
     - Clear `buffer.last_published_text`.

5. **Keep your translation, TTS, Worker API, and shutdown logic as is**

   - `translate_text` stays as it is, Quran aware and fancy.
   - `translate_and_publish` stays as it is, including `save_to_worker` and TTS queueing.
   - `tts_consumer` and Cartesia TTS stay the same.
   - The shutdown flush at the bottom of `process_audio_track` can still take `buffer.text` and do one last forced translation. You just need to make sure `buffer.text` returns `current_buffer_text(tokens)`.

6. **Wrap RT in the same retry loop**

   - Keep `MAX_STT_RETRIES` and `STT_RETRY_DELAY`.
   - Each attempt:

     - Open an RT session.
     - Create tasks for `send_audio` and `process_events`.
     - Wait for them to finish.
     - On timeout or specific errors, close RT, sleep, and retry.

---

## Concrete edits, step by step

I will point at the main blocks and show what to add or remove without dumping the whole file.

### Step 1. Imports and globals

At the top near your other imports, add the RT and transcription helpers.

Right after:

```python
from livekit.plugins.speechmatics import EndOfUtteranceMode
```

add:

```python
# Speechmatics RT client
from speechmatics.rt import (
    AsyncClient as SMAsyncClient,
    ServerMessageType as SMServerMessageType,
    TranscriptionConfig as SMTranscriptionConfig,
    AudioFormat as SMAudioFormat,
    AudioEncoding as SMAudioEncoding,
)

# DO NOT make this src.transcription
from transcription import (
    current_buffer_text,
    find_strict_eos_idx,
    find_relaxed_tail_idx,
    decide_commit_split,
)
```

Near the top, after `load_dotenv(".env.local")`, add:

```python
# Default RT URL if not set
os.environ.setdefault("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")
SPEECHMATICS_RT_URL = os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "").strip()
if not SPEECHMATICS_API_KEY:
    logger.error("SPEECHMATICS_API_KEY missing")
```

Replace your old buffer constants:

```python
# Buffer configuration
PUNCTUATION = re.compile(r'[.!?,;:]')
MIN_WORDS = 5
```

with:

```python
# Buffer configuration
MIN_WORDS_BEFORE_COMMIT = 10  # match local_test_rt behavior
```

You do not need `PUNCTUATION` any more.

---

### Step 2. Adjust TokenBuffer

Change `TokenBuffer.text` to use `current_buffer_text`.

Replace:

```python
    @property
    def text(self) -> str:
        """Join all tokens into full text."""
        return " ".join(self.tokens)
```

with:

```python
    @property
    def text(self) -> str:
        """Join all tokens into full text using the same rules as transcription.py."""
        return current_buffer_text(self.tokens)
```

You can leave `find_punct_idx` and `commit_at` for now, but we will stop using them. If you want to be clean, you can later delete those methods and any references.

---

### Step 3. Replace plugin STT in `process_audio_track`

Inside `process_audio_track`, right now you have:

```python
    # Speechmatics STT (token-based: each word = separate FINAL event)
    # Match legacy config for low-latency word-by-word streaming
    stt = speechmatics.STT(
        language=input_lang,
        operating_point="enhanced",
        max_delay=1.0,  # Force finals every 1s max (matches legacy)
        enable_partials=False,
        end_of_utterance_mode=EndOfUtteranceMode.NONE,
    )

    stt_stream = stt.stream()
    audio_stream = rtc.AudioStream(track)
    buffer = TokenBuffer()
```

Replace this whole block with:

```python
    # Speechmatics RT configuration (same behavior as local_test_rt)
    audio_stream = rtc.AudioStream(track)
    buffer = TokenBuffer()
    event_q: asyncio.Queue = asyncio.Queue()
```

You no longer create `stt` or `stt_stream`. You will use RT instead.

---

### Step 4. Replace `push_audio` and `process_stt_events` with RT version

Right now you have:

```python
    async def push_audio():
        async for audio_event in audio_stream:
            stt_stream.push_frame(audio_event.frame)
        stt_stream.end_input()

    async def process_stt_events():
        nonlocal buffer
        ...
        async for event in stt_stream:
            if event.type == SpeechEventType.START_OF_SPEECH:
                ...
            elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                ...
            elif event.type == SpeechEventType.END_OF_SPEECH:
                ...
```

Delete both of these functions.

Replace them with two new ones: `send_audio` and `process_events`, wired to RT.

Add:

```python
    async def send_audio(sm_client: SMAsyncClient):
        """Read audio frames from LiveKit and forward raw PCM to Speechmatics RT."""
        async for audio_event in audio_stream:
            frame = audio_event.frame
            data = getattr(frame, "data", None)
            if data is None:
                continue
            pcm_bytes = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
            await sm_client.send_audio(pcm_bytes)

    async def process_events():
        """Process RT ADD_TRANSCRIPT messages exactly like local_test_rt."""
        nonlocal buffer

        # Pair state
        pair_revs: Dict[int, int] = {}
        last_partial_text: str = ""
        last_partial_emit_ts: float = 0.0

        while True:
            etype, msg, _ts = await event_q.get()
            if etype != "final":
                continue

            # Extract tokens from RT message
            results = msg.get("results", []) if isinstance(msg, dict) else []
            new_tokens: List[str] = []
            for r in results:
                alts = r.get("alternatives", [])
                if not alts:
                    continue
                content = alts[0].get("content", "")
                if content:
                    new_tokens.append(content)

            if not new_tokens:
                continue

            # Start a new pair when buffer is empty
            if not buffer.tokens:
                buffer.pair_id = max(buffer.pair_id, 1)
                pair_revs.setdefault(buffer.pair_id, 0)
                logger.info(f"[PAIR] started pair_id={buffer.pair_id}")

            buffer.tokens.extend(new_tokens)
            buf_text = buffer.text
            logger.info(
                f"[RT] FINAL tokens+={len(new_tokens)} buffer_len={len(buffer.tokens)}: {buf_text}"
            )

            # Partial snapshot (TR incomplete)
            if buf_text and buf_text != last_partial_text:
                pid = buffer.pair_id
                rev = pair_revs.get(pid, 0) + 1
                pair_revs[pid] = rev
                await publish_buffer(
                    ctx, participant, buffer, status="incomplete", input_lang=input_lang
                )
                last_partial_text = buf_text
                last_partial_emit_ts = time.perf_counter()

            # Commit gating
            wc = len([t for t in buffer.tokens if t])
            if wc < MIN_WORDS_BEFORE_COMMIT:
                continue

            strict_idx: Optional[int]
            try:
                strict_idx = find_strict_eos_idx(buffer.tokens)
            except Exception:
                strict_idx = None

            commit_kind: Optional[str] = None
            idx: Optional[int] = None

            if strict_idx is not None and wc >= MIN_WORDS_BEFORE_COMMIT:
                commit_kind = "HARD_PUNCT"
                idx = int(strict_idx)
            else:
                soft_idx: Optional[int]
                try:
                    soft_idx = find_relaxed_tail_idx(buffer.tokens)
                except Exception:
                    soft_idx = None
                if soft_idx is not None and soft_idx >= MIN_WORDS_BEFORE_COMMIT:
                    commit_kind = "SOFT_COMMA_MAX"
                    idx = int(soft_idx)

            if idx is None or commit_kind is None:
                continue

            try:
                commit_tokens, carry_tokens, allowed = decide_commit_split(
                    buffer.tokens,
                    idx,
                    reason=commit_kind,
                    min_words=MIN_WORDS_BEFORE_COMMIT,
                )
            except Exception:
                commit_tokens, carry_tokens, allowed = (
                    buffer.tokens[:idx],
                    buffer.tokens[idx:],
                    True,
                )

            if not (allowed and commit_tokens):
                continue

            pid = buffer.pair_id
            commit_text = current_buffer_text(commit_tokens)
            carry_text = current_buffer_text(carry_tokens) if carry_tokens else ""

            # Update buffer to carry only
            buffer.tokens = list(carry_tokens)

            rev = pair_revs.get(pid, 0) + 1
            pair_revs[pid] = rev

            logger.info(
                f"[TR COMMIT] pair={pid} rev={rev} reason={commit_kind}: {commit_text}"
            )

            # Handle this commit with your existing pipeline
            await handle_commit_chunk(
                commit_text,
                pid,
                input_lang,
                output_langs,
                quran_enabled,
                quran_index,
                direct_index,
                ar_map,
                translations,
                tts_queues,
                ctx,
                participant,
                config,
            )

            # Rollover to next pair if there is carry text
            if carry_text:
                buffer.pair_id = pid + 1
                pair_revs[buffer.pair_id] = 1
                buffer.tokens = carry_tokens
                buffer.last_published_text = ""
                last_partial_text = carry_text
                await publish_buffer(
                    ctx, participant, buffer, status="incomplete", input_lang=input_lang
                )
            else:
                buffer.tokens = []
                buffer.pair_id = pid + 1
                buffer.last_published_text = ""
                last_partial_text = ""
```

You see there is a call to `handle_commit_chunk`. That is just a small helper we define below that wraps your current `handle_commit` logic.

---

### Step 5. Extract your commit handler into a helper

Inside `process_audio_track` you currently have an inner `handle_commit` function:

```python
        async def handle_commit(committed: str):
            """Handle a committed chunk: publish, detect Quran, translate."""
            # Publish complete transcript
            await publish_complete(ctx, participant, committed, buffer.pair_id, input_lang)
            ...
            # Run Quran detection
            ...
            # Parallel translation for all output_langs
            ...
            old_pair = buffer.pair_id
            buffer.pair_id += 1
            buffer.last_published_text = ""
            logger.info(f"🔄 ROLLOVER: {old_pair} -> {buffer.pair_id}")
```

We no longer want this function tied to the plugin STT loop.

Instead, move this logic into a small top level helper inside `process_audio_track`, just before `process_events`:

```python
    async def handle_commit_chunk(
        committed: str,
        pair_id: int,
        input_lang: str,
        output_langs: List[str],
        quran_enabled: bool,
        quran_index: Optional[QuranIndex],
        direct_index: Optional[DirectIndex],
        ar_map: Dict[str, Dict],
        translations: Dict[str, Dict[str, str]],
        tts_queues: Dict[str, asyncio.Queue],
        ctx: JobContext,
        participant: rtc.RemoteParticipant,
        config: dict,
    ):
        """Publish a committed chunk, run Quran detection, translate, and queue TTS."""
        await publish_complete(ctx, participant, committed, pair_id, input_lang)

        detections: List[Detection] = []
        if quran_enabled and quran_index and direct_index:
            fuzzy = detect_refs(quran_index, committed)
            direct = detect_refs_direct(direct_index, committed)
            detections = combine_detections(fuzzy, direct)
            if detections:
                logger.info(f"🕌 QURAN DETECTED pair={pair_id}: {[d.ref for d in detections]}")

        for lang in output_langs:
            asyncio.create_task(
                translate_and_publish(
                    ctx,
                    participant,
                    committed,
                    pair_id,
                    lang,
                    tts_queues,
                    detections=detections,
                    quran_enabled=quran_enabled,
                    ar_map=ar_map,
                    translations=translations,
                    session_id=config.get("session_id", ""),
                )
            )
```

This is almost exactly your old inner `handle_commit`, but it now takes `pair_id` as an argument and does not touch `buffer.pair_id`. The RT commit logic decides how `buffer.pair_id` rolls over.

You can then delete the old `handle_commit` and its call from the plugin STT loop, because we are no longer using that loop.

---

### Step 6. Wrap RT in your retry loop

Your existing retry loop looks like this:

```python
    try:
        for attempt in range(MAX_STT_RETRIES):
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(push_audio())
                    tg.create_task(process_stt_events())
                break  # Success
            except ...
    finally:
        ...
        await stt_stream.aclose()
```

Replace the body of the `try` section with RT session management:

```python
    try:
        for attempt in range(MAX_STT_RETRIES):
            try:
                async with SMAsyncClient(
                    api_key=SPEECHMATICS_API_KEY,
                    url=SPEECHMATICS_RT_URL,
                ) as sm_client:

                    @sm_client.on(SMServerMessageType.RECOGNITION_STARTED)
                    def _on_started(msg: Any) -> None:
                        logger.info("Speechmatics RT started: %s", msg)

                    @sm_client.on(SMServerMessageType.ADD_TRANSCRIPT)
                    def _on_add_transcript(msg: Any) -> None:
                        asyncio.create_task(event_q.put(("final", msg, time.time())))

                    fmt = SMAudioFormat(
                        encoding=SMAudioEncoding.PCM_S16LE,
                        sample_rate=48000,
                    )
                    cfg = SMTranscriptionConfig(
                        language=input_lang,
                        operating_point="enhanced",
                        max_delay=1.0,
                        max_delay_mode="flexible",
                        enable_partials=False,
                        enable_entities=False,
                        diarization="none",
                    )

                    await sm_client.start_session(
                        transcription_config=cfg,
                        audio_format=fmt,
                    )

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(send_audio(sm_client))
                        tg.create_task(process_events())

                break  # Success
            except asyncio.CancelledError:
                logger.info(f"Transcription cancelled for {participant.identity}")
                break
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str and attempt < MAX_STT_RETRIES - 1:
                    logger.warning(
                        f"STT connection timeout (attempt {attempt + 1}/{MAX_STT_RETRIES}), "
                        f"retrying in {STT_RETRY_DELAY}s..."
                    )
                    await asyncio.sleep(STT_RETRY_DELAY)
                else:
                    logger.exception(
                        f"Error in transcription for {participant.identity} "
                        f"after {attempt + 1} attempts: {e}"
                    )
                    break
    finally:
        remaining_text = buffer.text
        ...
        logger.info(f"Transcription ended for {participant.identity}")
```

In the `finally` block, keep your current shutdown flush logic.

You just need to remove `await stt_stream.aclose()` because you do not have `stt_stream` any more.

---

### Step 7. Check shutdown flush uses `buffer.text`

Your `finally` already does:

```python
        remaining_text = buffer.text
        if remaining_text and remaining_text != buffer.last_published_text:
            ...
            # translate and TTS
```

Since we changed `TokenBuffer.text` to use `current_buffer_text`, this will now flush any remaining tokens exactly like local_test_rt.

---

Once you do these changes, the flow will be:

- LiveKit audio → Speechmatics RT client.
- RT sends `ADD_TRANSCRIPT` → `event_q`.
- `process_events` updates `buffer.tokens`, publishes partials, commits with the same rules as local_test_rt.
- Each commit goes through `handle_commit_chunk` which uses your existing Quran aware GPT translation and TTS pipeline.
- Everything after that (TTS queues, Worker API, lifecycle) is unchanged.

When you have made the edits, paste the updated `process_audio_track` plus the new helper imports and I can sanity check it.
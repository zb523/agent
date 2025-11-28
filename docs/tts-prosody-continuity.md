# TTS Prosody Continuity: Cartesia Context Behavior

## Summary

We use direct Cartesia WebSocket API (bypassing LiveKit plugin) to maintain prosody continuity across TTS utterances. This document explains why, the limitations, and our decision to accept the current implementation.

---

## Why Direct API Instead of LiveKit Plugin

The LiveKit Cartesia plugin (`livekit-plugins-cartesia`) is designed for **one utterance per stream**. It generates a fresh `context_id` for each `SynthesizeStream` instance, meaning prosody resets between calls.

### LiveKit Plugin Limitations

1. **Single-segment design**: The base `SynthesizeStream` class has a deprecation guard that drops text after the first `flush()`:

```python
# From livekit-agents/tts/tts.py
def push_text(self, token: str) -> None:
    if not self._mtc_text:
        if self._num_segments >= 1:
            logger.warning("...handling multiple segments is deprecated...")
            return  # TEXT IS SILENTLY DROPPED
```

2. **Fresh context per stream**: Each `tts.stream()` call generates a new UUID:

```python
# From livekit-plugins-cartesia/tts.py
cartesia_context_id = utils.shortuuid()  # New UUID per stream
```

3. **No way to inject persistent context_id**: The plugin doesn't expose a way to reuse a context across multiple streams.

### References

- LiveKit Cartesia Plugin: https://docs.livekit.io/agents/models/tts/plugins/cartesia/
- Plugin source: https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-cartesia

---

## Cartesia Context Mechanics

Cartesia's WebSocket API uses `context_id` and `continue` to maintain prosody across inputs.

### How It Works

```json
{"transcript": "Hello!", "context_id": "my-session", "continue": true, ...}
{"transcript": "How are you?", "context_id": "my-session", "continue": true, ...}
{"transcript": "Goodbye!", "context_id": "my-session", "continue": false, ...}
```

- Same `context_id` = same voice state
- `continue: true` = more inputs coming, preserve prosody
- `continue: false` = context ends

### Critical Limitation: 1-Second Expiry

> **"Contexts automatically expire 1 second after the last audio output is streamed. Attempting to send another input on the same context_id after expiry is not supported."**
> 
> — [Cartesia Docs: Contexts](https://docs.cartesia.ai/api-reference/tts/working-with-web-sockets/contexts)

This means:
- If >1 second passes between audio chunks, context silently expires
- Subsequent sends on that `context_id` have undefined behavior
- No explicit error is returned for expired contexts

### Other Context Rules

1. **Transcripts concatenated verbatim**: You're responsible for spacing. Stripped text can cause join issues like `"word.Another"` instead of `"word. Another"`.

2. **Field stability required**: All fields except `transcript`, `continue`, and `duration` must stay the same within a context.

3. **WebSocket idle timeout**: 5 minutes of inactivity closes the connection (separate from context expiry).

### References

- Cartesia Contexts: https://docs.cartesia.ai/api-reference/tts/working-with-web-sockets/contexts
- Stream Inputs: https://docs.cartesia.ai/build-with-cartesia/capability-guides/stream-inputs-using-continuations
- Concurrency/Timeouts: https://docs.cartesia.ai/use-the-api/concurrency-limits-and-timeouts

---

## Our Implementation

### Current Approach

```python
# src/agent.py - tts_consumer()

# One WebSocket per language
# One context_id per language for session lifetime
context_id = f"tts-{language}-{uuid.uuid4().hex[:8]}"

# Jobs serialized through queue
while True:
    job = await queue.get()
    # Send with same context_id, continue=True
    await ws.send_json({
        "context_id": context_id,
        "transcript": text,
        "continue": True,
        ...
    })
    # Receive audio until done
    # Context stays alive between jobs
```

### Why This Works (Most of the Time)

When TTS is the bottleneck (queue backs up):
- Jobs process back-to-back
- Gap between "last audio of job N" and "first audio of job N+1" is ~100-300ms
- Context never expires

```
T=0.0   Job 1 finishes, last_audio
T=0.001 Job 2 pulled from queue, sent
T=0.1   Job 2 first audio arrives
        → Only 100ms gap, context ALIVE
```

### When It Breaks

When queue is empty and waiting for TL pipeline:
- No jobs to process
- Audio stops flowing
- After 1 second, context expires silently

```
T=0.0   Job 1 finishes
T=0.001 Queue empty, waiting...
T=1.0   CONTEXT EXPIRES
T=1.5   Job 2 arrives
        → Context dead, prosody won't continue
```

---

## Decision: Accept Current Behavior

### Rationale

1. **TTS is typically the bottleneck**: In our pipeline, TTS generation is slower than the TL pipeline. Jobs queue up, meaning back-to-back processing with no expiry risk.

2. **Natural pauses are acceptable**: For khutbah translation, natural pauses between ayat/points should feel like boundaries anyway. A prosody reset after a pause isn't necessarily bad.

3. **No official workaround**: Neither LiveKit nor Cartesia provide an official mechanism for maintaining context across gaps >1 second. Any solution would be a workaround.

4. **Complexity vs benefit**: Adding timestamp tracking and context regeneration adds complexity for marginal benefit in our use case.

### What We Accept

- Prosody continuity is **best-effort**, not guaranteed
- If imam pauses >1 second, prosody will reset
- If TL pipeline is slow, prosody will reset
- This is fine for our use case

### Potential Future Improvements

If prosody continuity becomes critical:

1. **Track last audio timestamp**:
```python
if time.time() - last_audio_ts > 0.8:
    context_id = f"tts-{language}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Context expired, regenerating")
```

2. **Add spacing to transcripts**:
```python
if tts_text and not tts_text.endswith((" ", "\n", ".", "!", "?")):
    tts_text += " "
```

3. **Reconnect on error**:
```python
if data.get("type") == "error":
    await ws.close()
    context_id = new_uuid()
    ws = await session.ws_connect(ws_url)
```

---

## Package Versions

As of implementation:
- `livekit-agents`: 1.3.x
- `livekit-plugins-cartesia`: 1.3.x
- Cartesia API version: `2024-06-10`
- Model: `sonic-3`

---

## Related Files

- `src/agent.py`: Main agent with `tts_consumer()` implementation
- `pyproject.toml`: Dependencies including `aiohttp` for direct WebSocket

---

*Last updated: November 2024*


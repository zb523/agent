#!/usr/bin/env python3
"""
Local mic -> Speechmatics RT -> buffering/commit test.

This script copies the STT buffering and commit logic from the RT
path in your StreamingAgent, but runs standalone and prints to console.

Run from the repo root so .env.local is picked up:
  uv run python local_test_rt.py
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import pyaudio
from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

from speechmatics.rt import (
    AsyncClient as SMAsyncClient,
    ServerMessageType as SMServerMessageType,
    TranscriptionConfig as SMTranscriptionConfig,
    AudioFormat as SMAudioFormat,
    AudioEncoding as SMAudioEncoding,
)

# DO NOT MAKE IT src.transcription
# transcription.py must be importable as plain "transcription"
from transcription import (
    current_buffer_text,
    find_strict_eos_idx,
    find_relaxed_tail_idx,
    decide_commit_split,
)

# -----------------------------------------------------------------------------
# Env and basic config
# -----------------------------------------------------------------------------

load_dotenv(".env.local")

os.environ.setdefault("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "").strip()
if not SPEECHMATICS_API_KEY:
    print("ERROR: SPEECHMATICS_API_KEY not set in environment or .env.local")
    raise SystemExit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set, translations will echo source text")
    openai_client: Optional[AsyncOpenAI] = None  # type: ignore[assignment]
else:
    if AsyncOpenAI is None:
        print("WARNING: openai package not installed, translations will echo source text")
        openai_client = None  # type: ignore[assignment]
    else:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SPEECHMATICS_RT_URL = os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")

# One input and one output language, like the reference app.py
SPEECHMATICS_LANG_CODES: Dict[str, str] = {
    "Arabic": "ar",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Russian": "ru",
    "Persian": "fa",
    "Farsi": "fa",
    "Turkish": "tr",
    "Urdu": "ur",
}


def _map_display_to_code(display_name: str) -> str:
    name = (display_name or "").strip()
    if not name:
        return "ar"
    code = SPEECHMATICS_LANG_CODES.get(name)
    if code:
        return code
    for k, v in SPEECHMATICS_LANG_CODES.items():
        if k.lower() == name.lower():
            return v
    return "ar"


INPUT_LANG_DISPLAY = os.getenv("INPUT_LANG", "Arabic")
OUTPUT_LANG_DISPLAY = os.getenv("OUTPUT_LANG", "English")

INPUT_LANG_CODE = _map_display_to_code(INPUT_LANG_DISPLAY)

# Commit gating, copied from reference
MIN_WORDS_BEFORE_COMMIT = 10

# Audio config
SAMPLE_RATE = 48000
CHANNELS = 1
FRAMES_PER_BUFFER = 960  # 20 ms at 48 kHz


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

async def translate_text(text: str) -> str:
    """Translate to OUTPUT_LANG_DISPLAY using OpenAI if available, else echo."""
    content = (text or "").strip()
    if not content:
        return ""

    if openai_client is None:
        return content

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the user's message into natural, fluent {OUTPUT_LANG_DISPLAY}. "
                        "Only return the translation."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        if not resp.choices:
            return content
        out = resp.choices[0].message.content or ""
        return out.strip()
    except Exception as e:
        print(f"[TL] ERROR calling OpenAI: {e}")
        return content


def print_tr_partial(pair_id: int, rev: int, text: str) -> None:
    print(f"[TR][PARTIAL] pair={pair_id} rev={rev}: {text}")


def print_tr_commit(pair_id: int, rev: int, reason: str, text: str) -> None:
    print(f"[TR][COMMIT] pair={pair_id} rev={rev} reason={reason}: {text}")


def print_tl_commit(pair_id: int, text: str, original: str) -> None:
    print(f"[TL][FINAL] pair={pair_id} lang={OUTPUT_LANG_DISPLAY}: {text}")
    print(f"          (original: {original})")


async def tts_consumer(queue: asyncio.Queue) -> None:
    """Simulate TTS with a sleep, one job at a time."""
    print("[TTS] consumer started")
    while True:
        job = await queue.get()
        if job is None:
            print("[TTS] consumer stopping")
            queue.task_done()
            break
        text, pair_id = job
        try:
            words = len((text or "").split())
            print(f"[TTS] START pair={pair_id} words={words}")
            await asyncio.sleep(words * 0.1)
            print(f"[TTS] DONE  pair={pair_id}")
        except Exception as e:
            print(f"[TTS] ERROR pair={pair_id}: {e}")
        finally:
            queue.task_done()


async def handle_commit(
    commit_text: str,
    pair_id: int,
    tts_queue: asyncio.Queue,
) -> None:
    """On each committed Arabic chunk, translate and queue fake TTS."""
    tl = await translate_text(commit_text)
    print_tl_commit(pair_id, tl, commit_text)
    await tts_queue.put((tl, pair_id))


# -----------------------------------------------------------------------------
# Main run logic
# -----------------------------------------------------------------------------

async def run() -> None:
    print("")
    print("=" * 80)
    print("LOCAL MIC TEST: Speechmatics RT -> buffer/commit -> translation")
    print("=" * 80)
    print(f"Input language display: {INPUT_LANG_DISPLAY} (code={INPUT_LANG_CODE})")
    print(f"Output language display: {OUTPUT_LANG_DISPLAY}")
    print(f"MIN_WORDS_BEFORE_COMMIT = {MIN_WORDS_BEFORE_COMMIT}")
    print("Mic: 48 kHz mono, 20 ms frames")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print("")

    # Shared state that mirrors StreamingAgent RT path
    buffer_tokens: List[str] = []
    pair_id_counter: int = 0
    current_pid: Optional[int] = None
    pair_revs: Dict[int, int] = {}
    last_partial_text: str = ""
    last_partial_emit_ts: float = 0.0
    arabic_history: List[str] = []

    event_q: asyncio.Queue = asyncio.Queue()

    # TTS queue and worker
    tts_queue: asyncio.Queue = asyncio.Queue()
    tts_task = asyncio.create_task(tts_consumer(tts_queue))

    # Mic setup
    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    print("[MIC] opened, streaming audio to Speechmatics RT")
    print("")

    async def process_events() -> None:
        nonlocal buffer_tokens, pair_id_counter, current_pid
        nonlocal pair_revs, last_partial_text, last_partial_emit_ts, arabic_history

        while True:
            etype, msg, _ts = await event_q.get()
            if etype != "final":
                continue

            # Parse final tokens from Speechmatics RT message
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

            # Start pair lazily when first tokens arrive
            if not buffer_tokens and current_pid is None:
                pair_id_counter += 1
                current_pid = pair_id_counter
                pair_revs[current_pid] = 0
                print(f"[PAIR] started pair_id={current_pid}")

            # Grow buffer incrementally, like RT path
            try:
                buffer_tokens.extend(new_tokens)
            except Exception:
                buffer_tokens = list(new_tokens)

            buf_text = current_buffer_text(buffer_tokens)
            print(
                f"[RT] FINAL tokens+={len(new_tokens)} buffer_len={len(buffer_tokens)}: {buf_text}"
            )

            # Snapshot before commit, like the RT path
            if current_pid is not None:
                if buf_text and buf_text != (last_partial_text or ""):
                    pid = current_pid
                    rev = pair_revs.get(pid, 0) + 1
                    pair_revs[pid] = rev
                    print_tr_partial(pid, rev, buf_text)
                    last_partial_text = buf_text
                    last_partial_emit_ts = time.perf_counter()

            # Commit check: punctuation driven, with MIN_WORDS_BEFORE_COMMIT
            wc = len([t for t in buffer_tokens if t])
            strict_idx: Optional[int]
            try:
                strict_idx = find_strict_eos_idx(buffer_tokens)
            except Exception:
                strict_idx = None

            commit_kind: Optional[str] = None
            idx: Optional[int] = None

            if strict_idx is not None and wc >= MIN_WORDS_BEFORE_COMMIT:
                commit_kind = "HARD_PUNCT"
                idx = int(strict_idx)
            elif wc >= MIN_WORDS_BEFORE_COMMIT:
                try:
                    soft_idx = find_relaxed_tail_idx(buffer_tokens)
                except Exception:
                    soft_idx = None
                if soft_idx is not None and soft_idx >= MIN_WORDS_BEFORE_COMMIT:
                    commit_kind = "SOFT_COMMA_MAX"
                    idx = int(soft_idx)

            if (
                idx is not None
                and commit_kind is not None
                and current_pid is not None
            ):
                # Decide split exactly like decide_commit_split in reference
                try:
                    commit_tokens, carry_tokens, allowed = decide_commit_split(
                        buffer_tokens,
                        idx,
                        reason=commit_kind,
                        min_words=MIN_WORDS_BEFORE_COMMIT,
                    )
                except Exception:
                    commit_tokens, carry_tokens, allowed = (
                        buffer_tokens[:idx],
                        buffer_tokens[idx:],
                        True,
                    )

                if not (allowed and commit_tokens):
                    continue

                pid = current_pid
                commit_text = current_buffer_text(commit_tokens)
                buffer_tokens = carry_tokens

                rev = pair_revs.get(pid, 0) + 1
                pair_revs[pid] = rev

                print_tr_commit(pid, rev, commit_kind, commit_text)

                # Translation and fake TTS
                asyncio.create_task(handle_commit(commit_text, pid, tts_queue))

                arabic_history.append(commit_text)
                if len(arabic_history) > 10:
                    arabic_history = arabic_history[-10:]

                carry_text = (
                    current_buffer_text(carry_tokens) if carry_tokens else ""
                )
                if carry_text:
                    pair_id_counter += 1
                    current_pid = pair_id_counter
                    pair_revs[current_pid] = 1
                    print(f"[PAIR] carry -> new pair_id={current_pid}")
                    print_tr_partial(current_pid, 1, carry_text)
                    last_partial_text = carry_text
                    last_partial_emit_ts = time.perf_counter()
                else:
                    print(f"[PAIR] closed pair_id={pid}")
                    current_pid = None
                    last_partial_text = ""

    async def send_audio() -> None:
        fmt = SMAudioFormat(
            encoding=SMAudioEncoding.PCM_S16LE,
            sample_rate=SAMPLE_RATE,
        )
        cfg = SMTranscriptionConfig(
            language=INPUT_LANG_CODE,
            operating_point="enhanced",
            max_delay=1.0,
            max_delay_mode="flexible",
            enable_partials=False,
            enable_entities=False,
            diarization="none",
        )

        async with SMAsyncClient(
            api_key=SPEECHMATICS_API_KEY,
            url=SPEECHMATICS_RT_URL,
        ) as sm_client:

            @sm_client.on(SMServerMessageType.RECOGNITION_STARTED)
            def _on_started(msg: Any) -> None:
                print(f"[SM] recognition started: {msg}")

            @sm_client.on(SMServerMessageType.ADD_TRANSCRIPT)
            def _on_add_transcript(msg: Any) -> None:
                # Treat every ADD_TRANSCRIPT as final chunk, like reference RT path
                asyncio.create_task(event_q.put(("final", msg, time.time())))

            await sm_client.start_session(
                transcription_config=cfg,
                audio_format=fmt,
            )

            try:
                while True:
                    data = await asyncio.to_thread(
                        mic_stream.read,
                        FRAMES_PER_BUFFER,
                        exception_on_overflow=False,
                    )
                    await sm_client.send_audio(data)
            finally:
                try:
                    await sm_client.stop_session()
                except Exception:
                    pass

    audio_task = asyncio.create_task(send_audio())
    events_task = asyncio.create_task(process_events())

    try:
        await asyncio.gather(audio_task, events_task)
    except asyncio.CancelledError:
        pass
    finally:
        audio_task.cancel()
        events_task.cancel()
        try:
            await asyncio.gather(audio_task, events_task, return_exceptions=True)
        except Exception:
            pass

        # Drain pending translations
        pending = []
        while not tts_queue.empty():
            try:
                pending.append(tts_queue.get_nowait())
            except Exception:
                break

        if pending:
            print(f"[TTS] draining {len(pending)} queued items")

        await tts_queue.put(None)
        try:
            await asyncio.wait_for(tts_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("[TTS] did not stop cleanly")

        mic_stream.stop_stream()
        mic_stream.close()
        p.terminate()

        print("")
        print("=" * 80)
        print("Stopped")
        print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")

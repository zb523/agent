#!/usr/bin/env python3
"""
Local test script for the transcription + translation pipeline.
Uses LiveKit's Speechmatics STT plugin directly (same as agent.py).

Run: uv run python scripts/local_test.py
"""
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyaudio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from livekit.agents.stt import SpeechEventType
from livekit.plugins import speechmatics
from livekit.plugins.speechmatics import EndOfUtteranceMode
from livekit import rtc

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")

# Validate API keys
if not os.getenv("SPEECHMATICS_API_KEY"):
    print("ERROR: SPEECHMATICS_API_KEY not found in .env.local")
    sys.exit(1)
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in .env.local")
    sys.exit(1)

# OpenAI client
openai_client = AsyncOpenAI()

# =============================================================================
# Language Configuration (mirrors agent.py)
# =============================================================================
INPUT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ["ar"]  # Match agent.py
LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
}

# Buffer configuration
PUNCTUATION = re.compile(r'[.!?,;:]')
MIN_WORDS = 5
SAMPLE_RATE = 48000
CHANNELS = 1


@dataclass
class TranscriptBuffer:
    """
    Buffer for tracking cumulative Speechmatics transcripts.
    
    Speechmatics INTERIM events contain the FULL transcript so far (cumulative),
    not incremental updates. We track committed_idx to know what we've already
    processed, and only look at uncommitted text for new commits.
    """
    pair_id: int = 1
    full_text: str = ""
    committed_idx: int = 0
    last_published_text: str = ""
    
    @property
    def uncommitted(self) -> str:
        """Text after the committed position, stripped."""
        return self.full_text[self.committed_idx:].lstrip()
    
    def find_commit_point(self) -> int | None:
        """Find commit point in UNCOMMITTED text only (first punct after MIN_WORDS)."""
        text = self.uncommitted
        if not text:
            return None
        words_seen = 0
        for i, char in enumerate(text):
            if char == ' ':
                words_seen += 1
            if words_seen >= MIN_WORDS - 1 and PUNCTUATION.match(char):
                return i + 1
        return None
    
    def commit(self) -> str:
        """
        Commit text up to the commit point. Returns the committed text.
        Advances committed_idx so we don't re-commit the same text.
        """
        idx = self.find_commit_point()
        if idx is None:
            return ""
        
        uncommitted = self.uncommitted
        committed = uncommitted[:idx].strip()
        
        raw_uncommitted = self.full_text[self.committed_idx:]
        lstrip_offset = len(raw_uncommitted) - len(raw_uncommitted.lstrip())
        self.committed_idx += lstrip_offset + idx
        
        return committed


# =============================================================================
# Translation (GPT-4.1)
# =============================================================================
async def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using GPT-4.1."""
    response = await openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"Translate to {LANG_NAMES[target_lang]}. Return only the translation, nothing else."},
            {"role": "user", "content": text}
        ],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Mock Publishing (prints to console)
# =============================================================================
def publish_to_room(pair_id: int, text: str, status: str, msg_type: str, language: str, original: str = None):
    """Simulate publishing to room (prints to console)."""
    if msg_type == "transcript":
        icon = "📝" if status == "incomplete" else "✅"
        print(f"  {icon} [{status.upper()}] pair={pair_id} lang={language}: {text}")
    else:
        lang_emoji = {"es": "🇪🇸", "fr": "🇫🇷", "ar": "🇸🇦"}.get(language, "🌐")
        print(f"  {lang_emoji} [TRANSLATED] pair={pair_id} lang={language}: {text}")
        if original:
            print(f"       (original: {original})")


# =============================================================================
# TTS Consumer (Queue-based serialization - mirrors agent.py)
# =============================================================================
async def tts_consumer(queue: asyncio.Queue, language: str):
    """
    Process TTS jobs sequentially for a single language.
    In local test, we simulate TTS with a sleep.
    """
    print(f"  [TTS CONSUMER] Started for {language}")
    while True:
        job = await queue.get()
        if job is None:  # Shutdown signal
            print(f"  [TTS CONSUMER] Shutting down for {language}")
            break
        
        text, pair_id = job
        try:
            print(f"  🔊 [TTS START] pair={pair_id} lang={language}: '{text[:30]}...'")
            # Simulate TTS playback time (roughly 100ms per word)
            word_count = len(text.split())
            await asyncio.sleep(word_count * 0.1)
            print(f"  🔊 [TTS DONE] pair={pair_id} lang={language}")
        except Exception as e:
            print(f"  ❌ TTS failed for pair={pair_id} lang={language}: {e}")
        finally:
            queue.task_done()


async def handle_translation(text: str, pair_id: int, target_lang: str, tts_queue: asyncio.Queue):
    """Translate to a single language and queue TTS."""
    try:
        translation = await translate_text(text, target_lang)
        publish_to_room(pair_id, translation, "complete", "translation", target_lang, original=text)
        
        # Queue TTS job for serialized playback (mirrors agent.py)
        await tts_queue.put((translation, pair_id))
        print(f"       [TTS QUEUED] pair={pair_id} lang={target_lang}")
        
    except Exception as e:
        print(f"  ❌ Translation to {target_lang} failed: {e}")


async def run():
    buffer = TranscriptBuffer()
    pending_translations = []
    
    print("\n" + "=" * 80)
    print("LOCAL TRANSCRIPTION + MULTI-LANGUAGE TRANSLATION TEST")
    print("=" * 80)
    print(f"Using LiveKit Speechmatics STT plugin (same as agent.py)")
    print(f"Buffer: MIN_WORDS={MIN_WORDS}, punctuation: . ! ? , ; :")
    print(f"Input: {INPUT_LANGUAGE} ({LANG_NAMES[INPUT_LANGUAGE]})")
    print(f"Output: {', '.join(f'{lang} ({LANG_NAMES[lang]})' for lang in OUTPUT_LANGUAGES)}")
    print("Translation model: GPT-4.1")
    print("TTS: Simulated with queue serialization (mirrors agent.py)")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")

    # ==========================================================================
    # Create TTS queues and consumer tasks (serialized playback)
    # ==========================================================================
    tts_queues: dict[str, asyncio.Queue] = {}
    tts_tasks: dict[str, asyncio.Task] = {}
    
    for lang in OUTPUT_LANGUAGES:
        queue: asyncio.Queue[Optional[tuple[str, int]]] = asyncio.Queue()
        tts_queues[lang] = queue
        tts_tasks[lang] = asyncio.create_task(tts_consumer(queue, lang))

    # Create STT
    stt = speechmatics.STT(
        language=INPUT_LANGUAGE,
        enable_partials=True,
        end_of_utterance_mode=EndOfUtteranceMode.NONE,
    )
    
    stt_stream = stt.stream()
    
    # Open microphone with PyAudio
    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=960,  # 20ms at 48kHz
    )
    print("🎤 Microphone opened. Listening...\n")

    async def push_audio():
        """Push mic audio to STT."""
        try:
            while True:
                data = await asyncio.to_thread(
                    mic_stream.read, 960, exception_on_overflow=False
                )
                frame = rtc.AudioFrame(
                    data=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=CHANNELS,
                    samples_per_channel=960,
                )
                stt_stream.push_frame(frame)
        except asyncio.CancelledError:
            stt_stream.end_input()

    async def process_events():
        """Process STT events with fixed buffer tracking."""
        nonlocal buffer, pending_translations
        
        async for event in stt_stream:
            if event.type == SpeechEventType.START_OF_SPEECH:
                print("  🎤 [START OF SPEECH]")
                
            elif event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                if event.alternatives:
                    text = event.alternatives[0].text
                    if not text:
                        continue
                    
                    buffer.full_text = text
                    uncommitted = buffer.uncommitted
                    
                    # Publish if uncommitted text changed
                    if uncommitted and uncommitted != buffer.last_published_text:
                        publish_to_room(buffer.pair_id, uncommitted, "incomplete", "transcript", INPUT_LANGUAGE)
                        buffer.last_published_text = uncommitted
                    
                    # Check for commit
                    if buffer.find_commit_point():
                        committed = buffer.commit()
                        if committed:
                            # Publish complete transcript
                            publish_to_room(buffer.pair_id, committed, "complete", "transcript", INPUT_LANGUAGE)
                            
                            # Parallel translation + queue TTS for all output languages
                            for lang in OUTPUT_LANGUAGES:
                                task = asyncio.create_task(
                                    handle_translation(committed, buffer.pair_id, lang, tts_queues[lang])
                                )
                                pending_translations.append(task)
                            
                            old = buffer.pair_id
                            buffer.pair_id += 1
                            buffer.last_published_text = ""
                            print(f"  🔄 ROLLOVER: {old} -> {buffer.pair_id}")
                            
            elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                if event.alternatives:
                    text = event.alternatives[0].text
                    if text:
                        print(f"  📋 [FINAL]: {text}")
                        
            elif event.type == SpeechEventType.END_OF_SPEECH:
                print("  🔇 [END OF SPEECH]")

    audio_task = None
    try:
        audio_task = asyncio.create_task(push_audio())
        await process_events()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + "=" * 80)
        print("STOPPING...")
        
        if audio_task:
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
        
        if pending_translations:
            print(f"Waiting for {len(pending_translations)} translations...")
            await asyncio.gather(*pending_translations, return_exceptions=True)
        
        # Graceful shutdown of TTS consumers
        print("Shutting down TTS consumers...")
        for lang, queue in tts_queues.items():
            await queue.put(None)  # Signal shutdown
        for lang, task in tts_tasks.items():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                print(f"TTS consumer for {lang} did not shut down in time")
                task.cancel()
        
        # Flush any uncommitted text
        uncommitted = buffer.uncommitted
        if uncommitted and uncommitted != buffer.last_published_text:
            print("Flushing incomplete buffer:")
            publish_to_room(buffer.pair_id, uncommitted, "incomplete", "transcript", INPUT_LANGUAGE)
        
        await stt_stream.aclose()
        mic_stream.stop_stream()
        mic_stream.close()
        p.terminate()
        
        print(f"Final pair_id: {buffer.pair_id}")
        print(f"Total committed: {buffer.committed_idx} chars")
        print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBye!")

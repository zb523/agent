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
from dataclasses import dataclass, field
from pathlib import Path

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

# Constants
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
    full_text: str = ""           # Full cumulative transcript from Speechmatics
    committed_idx: int = 0        # Position we've committed up to
    last_published_text: str = "" # Last uncommitted text published (for dedup)
    
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
        
        # Calculate how much to advance committed_idx
        # Account for any whitespace we stripped with lstrip()
        raw_uncommitted = self.full_text[self.committed_idx:]
        lstrip_offset = len(raw_uncommitted) - len(raw_uncommitted.lstrip())
        self.committed_idx += lstrip_offset + idx
        
        return committed


async def translate_to_spanish(text: str) -> str:
    """Translate text to Spanish using OpenAI."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Translate to Spanish. Return only the translation."},
            {"role": "user", "content": text}
        ],
    )
    return response.choices[0].message.content.strip()


def publish_to_room(pair_id: int, text: str, status: str, msg_type: str, original: str = None):
    """Simulate publishing to room."""
    if msg_type == "transcript":
        icon = "📝" if status == "incomplete" else "✅"
        print(f"  {icon} [{status.upper()}] pair={pair_id}: {text}")
    else:
        print(f"  🌐 [TRANSLATED] pair={pair_id}: {text}")
        if original:
            print(f"     (original: {original})")


async def handle_translation(text: str, pair_id: int):
    """Translate and publish."""
    try:
        translation = await translate_to_spanish(text)
        publish_to_room(pair_id, translation, "complete", "translation", original=text)
    except Exception as e:
        print(f"  ❌ Translation failed: {e}")


async def run():
    buffer = TranscriptBuffer()
    pending_translations = []
    
    print("\n" + "=" * 80)
    print("LOCAL TRANSCRIPTION + TRANSLATION TEST")
    print("=" * 80)
    print(f"Using LiveKit Speechmatics STT plugin (same as agent.py)")
    print(f"Buffer: MIN_WORDS={MIN_WORDS}, punctuation: . ! ? , ; :")
    print("Speak in English -> Translated to Spanish")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")

    # Create STT
    stt = speechmatics.STT(
        language="en",
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
                    
                    # Update full cumulative text
                    buffer.full_text = text
                    
                    # Get uncommitted portion only
                    uncommitted = buffer.uncommitted
                    
                    # Publish if uncommitted text changed (dedup)
                    if uncommitted and uncommitted != buffer.last_published_text:
                        publish_to_room(buffer.pair_id, uncommitted, "incomplete", "transcript")
                        buffer.last_published_text = uncommitted
                    
                    # Check for commit point in uncommitted text
                    if buffer.find_commit_point():
                        committed = buffer.commit()
                        if committed:
                            # Publish complete
                            publish_to_room(buffer.pair_id, committed, "complete", "transcript")
                            
                            # Translate async
                            task = asyncio.create_task(handle_translation(committed, buffer.pair_id))
                            pending_translations.append(task)
                            
                            # Advance pair_id for next chunk
                            old = buffer.pair_id
                            buffer.pair_id += 1
                            buffer.last_published_text = ""
                            
                            # Show new uncommitted after commit
                            new_uncommitted = buffer.uncommitted
                            print(f"  🔄 ROLLOVER: {old} -> {buffer.pair_id}, new uncommitted='{new_uncommitted}'")
                            
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
        
        # Flush any uncommitted text
        uncommitted = buffer.uncommitted
        if uncommitted and uncommitted != buffer.last_published_text:
            print("Flushing incomplete buffer:")
            publish_to_room(buffer.pair_id, uncommitted, "incomplete", "transcript")
        
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

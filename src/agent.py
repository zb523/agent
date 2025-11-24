import asyncio
import logging
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.stt import SpeechEventType
from livekit.plugins import speechmatics
from livekit.plugins.speechmatics import EndOfUtteranceMode

logger = logging.getLogger("transcription-agent")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv(".env.local")

# OpenAI client for translation
openai_client = AsyncOpenAI()

# Constants
PUNCTUATION = re.compile(r'[.!?,;:]')
MIN_WORDS = 5

server = AgentServer()


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
            {"role": "system", "content": "Translate to Spanish. Return only the translation, nothing else."},
            {"role": "user", "content": text}
        ],
    )
    return response.choices[0].message.content.strip()


@server.rtc_session()
async def transcription_agent(ctx: JobContext):
    """
    Real-time transcription and translation agent using Speechmatics.
    Subscribes to participant audio, transcribes, buffers until punctuation,
    then translates to Spanish and publishes to room.
    """
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Transcription agent starting in room: {ctx.room.name}")

    # Connect to room first
    await ctx.connect()
    logger.info("Connected to room, waiting for participants...")

    # Track active transcription tasks
    active_tasks: dict[str, asyncio.Task] = {}

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
            task = asyncio.create_task(
                process_audio_track(ctx, track, participant)
            )
            active_tasks[participant.identity] = task

    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track unsubscribed from {participant.identity}")
            if participant.identity in active_tasks:
                active_tasks[participant.identity].cancel()
                del active_tasks[participant.identity]

    # Process any already-subscribed tracks
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Found existing audio track from {participant.identity}")
                task = asyncio.create_task(
                    process_audio_track(ctx, publication.track, participant)
                )
                active_tasks[participant.identity] = task

    # Keep agent alive
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Agent cancelled, cleaning up...")
        for task in active_tasks.values():
            task.cancel()


async def process_audio_track(
    ctx: JobContext,
    track: rtc.RemoteTrack,
    participant: rtc.RemoteParticipant,
):
    """
    Process audio from a participant, transcribe with Speechmatics,
    buffer and translate to Spanish when commit conditions are met.
    """
    logger.info(f"Starting transcription for {participant.identity}")

    # Create Speechmatics STT with partials enabled, EOU disabled (one-way pipeline)
    stt = speechmatics.STT(
        language="en",
        enable_partials=True,  # Critical: enables INTERIM_TRANSCRIPT events
        end_of_utterance_mode=EndOfUtteranceMode.NONE,  # Disable EOU - one-way pipeline
    )

    stt_stream = stt.stream()
    audio_stream = rtc.AudioStream(track)
    
    # Buffer for this participant
    buffer = TranscriptBuffer()

    async def push_audio():
        """Push audio frames to STT stream."""
        async for audio_event in audio_stream:
            stt_stream.push_frame(audio_event.frame)
        stt_stream.end_input()

    async def process_stt_events():
        """Process STT events with fixed buffer tracking."""
        nonlocal buffer
        
        async for event in stt_stream:
            if event.type == SpeechEventType.START_OF_SPEECH:
                logger.info(f"🎤 START OF SPEECH [{participant.identity}]")

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
                        await publish_buffer(ctx, participant, buffer, status="incomplete")
                        buffer.last_published_text = uncommitted
                    
                    # Check for commit point in uncommitted text
                    if buffer.find_commit_point():
                        committed = buffer.commit()
                        if committed:
                            # Publish complete transcript
                            await publish_complete(ctx, participant, committed, buffer.pair_id)
                            
                            # Translate asynchronously (don't block)
                            asyncio.create_task(
                                translate_and_publish(ctx, participant, committed, buffer.pair_id)
                            )
                            
                            # Advance pair_id for next chunk
                            old_pair = buffer.pair_id
                            buffer.pair_id += 1
                            buffer.last_published_text = ""
                            
                            logger.info(f"🔄 ROLLOVER: {old_pair} -> {buffer.pair_id}, uncommitted='{buffer.uncommitted}'")

            elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                if event.alternatives:
                    text = event.alternatives[0].text
                    if text:
                        logger.debug(f"📋 FINAL (info only): {text}")

            elif event.type == SpeechEventType.END_OF_SPEECH:
                logger.info(f"🔇 END OF SPEECH [{participant.identity}]")

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(push_audio())
            tg.create_task(process_stt_events())
    except asyncio.CancelledError:
        logger.info(f"Transcription cancelled for {participant.identity}")
    except Exception as e:
        logger.exception(f"Error in transcription for {participant.identity}: {e}")
    finally:
        # Handle stream end - publish incomplete buffer without translation
        uncommitted = buffer.uncommitted
        if uncommitted and uncommitted != buffer.last_published_text:
            logger.info(f"📤 FLUSH incomplete buffer on stream end: '{uncommitted}'")
            await publish_buffer(ctx, participant, buffer, status="incomplete")
        
        await stt_stream.aclose()
        logger.info(f"Transcription ended for {participant.identity}")


async def publish_buffer(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    buffer: TranscriptBuffer,
    status: str,
):
    """
    Publish current uncommitted transcript to room.
    """
    text = buffer.uncommitted
    if not text.strip():
        return

    try:
        attributes = {
            "pair_id": str(buffer.pair_id),
            "status": status,
            "type": "transcript",
            "participant_identity": participant.identity,
        }

        await ctx.room.local_participant.send_text(
            text,
            topic="lk.transcription",
            attributes=attributes,
        )
        logger.info(f"[{status.upper()}] pair={buffer.pair_id}: {text}")
    except Exception as e:
        logger.error(f"❌ Failed to publish buffer: {e}")


async def publish_complete(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
):
    """
    Publish a completed transcript chunk to room.
    """
    if not text.strip():
        return

    try:
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "transcript",
            "participant_identity": participant.identity,
        }

        await ctx.room.local_participant.send_text(
            text,
            topic="lk.transcription",
            attributes=attributes,
        )
        logger.info(f"[COMPLETE] pair={pair_id}: {text}")
    except Exception as e:
        logger.error(f"❌ Failed to publish complete: {e}")


async def translate_and_publish(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
):
    """
    Translate text to Spanish and publish to room.
    """
    try:
        translation = await translate_to_spanish(text)
        
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "translation",
            "participant_identity": participant.identity,
            "original_text": text,
        }
        
        await ctx.room.local_participant.send_text(
            translation,
            topic="lk.transcription",
            attributes=attributes,
        )
        logger.info(f"[TRANSLATED] pair={pair_id}: {translation}")
    except Exception as e:
        logger.error(f"❌ Translation failed for pair={pair_id}: {e}")


if __name__ == "__main__":
    cli.run_app(server)

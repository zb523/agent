import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.stt import SpeechEventType
from livekit.plugins import speechmatics, cartesia
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

# =============================================================================
# Language Configuration
# =============================================================================
INPUT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ["ar"]
LANG_NAMES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French", 
    "ar": "Arabic",
}

# Buffer configuration
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
# TTS (Cartesia Sonic 3)
# =============================================================================
def create_tts(language: str) -> cartesia.TTS:
    """Create a Cartesia TTS instance for the given language."""
    return cartesia.TTS(
        model="sonic-3",
        language=language,
    )


async def synthesize_and_play(text: str, language: str, audio_source: rtc.AudioSource):
    """Synthesize text to speech and play on the audio source."""
    try:
        tts = create_tts(language)
        tts_stream = tts.stream()
        tts_stream.push_text(text)
        tts_stream.end_input()
        
        async for audio in tts_stream:
            await audio_source.capture_frame(audio.frame)
    except Exception as e:
        logger.error(f"TTS synthesis failed for {language}: {e}")


# =============================================================================
# TTS Consumer (Queue-based serialization)
# =============================================================================
async def tts_consumer(
    queue: asyncio.Queue,
    audio_source: rtc.AudioSource,
    language: str,
):
    """
    Process TTS jobs sequentially for a single language.
    Ensures no concurrent capture_frame() calls on the same audio source.
    """
    logger.info(f"TTS consumer started for {language}")
    while True:
        job = await queue.get()
        if job is None:  # Shutdown signal
            logger.info(f"TTS consumer shutting down for {language}")
            break
        
        text, pair_id = job
        try:
            logger.info(f"[TTS START] pair={pair_id} lang={language}")
            await synthesize_and_play(text, language, audio_source)
            logger.info(f"[TTS DONE] pair={pair_id} lang={language}")
        except Exception as e:
            logger.error(f"TTS failed for pair={pair_id} lang={language}: {e}")
        finally:
            queue.task_done()


# =============================================================================
# Main Agent
# =============================================================================
@server.rtc_session()
async def transcription_agent(ctx: JobContext):
    """
    Real-time transcription and translation agent.
    - Transcribes with Speechmatics
    - Translates to multiple languages with GPT-4.1
    - Synthesizes audio with Cartesia Sonic 3 (serialized via queues)
    """
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Transcription agent starting in room: {ctx.room.name}")

    # Connect to room
    await ctx.connect()
    logger.info("Connected to room, waiting for participants...")

    # ==========================================================================
    # Create TTS audio tracks per output language
    # ==========================================================================
    audio_sources: dict[str, rtc.AudioSource] = {}
    
    for lang in OUTPUT_LANGUAGES:
        source = rtc.AudioSource(24000, 1)  # Cartesia outputs 24kHz mono
        track = rtc.LocalAudioTrack.create_audio_track(f"tts-{lang}", source)
        await ctx.room.local_participant.publish_track(track)
        audio_sources[lang] = source
        logger.info(f"Published TTS audio track: tts-{lang}")

    # ==========================================================================
    # Create TTS queues and consumer tasks (serialized playback)
    # ==========================================================================
    tts_queues: dict[str, asyncio.Queue] = {}
    tts_tasks: dict[str, asyncio.Task] = {}
    
    for lang in OUTPUT_LANGUAGES:
        queue: asyncio.Queue[Optional[tuple[str, int]]] = asyncio.Queue()
        tts_queues[lang] = queue
        tts_tasks[lang] = asyncio.create_task(
            tts_consumer(queue, audio_sources[lang], lang)
        )

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
                process_audio_track(ctx, track, participant, tts_queues)
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
                    process_audio_track(ctx, publication.track, participant, tts_queues)
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
    finally:
        # Graceful shutdown of TTS consumers
        logger.info("Shutting down TTS consumers...")
        for lang, queue in tts_queues.items():
            await queue.put(None)  # Signal shutdown
        for lang, task in tts_tasks.items():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"TTS consumer for {lang} did not shut down in time")
                task.cancel()


async def process_audio_track(
    ctx: JobContext,
    track: rtc.RemoteTrack,
    participant: rtc.RemoteParticipant,
    tts_queues: dict[str, asyncio.Queue],
):
    """
    Process audio from a participant:
    1. Transcribe with Speechmatics
    2. Buffer until commit (5+ words + punctuation)
    3. Translate to all output languages in parallel
    4. Queue TTS for serialized playback
    """
    logger.info(f"Starting transcription for {participant.identity}")

    # Speechmatics STT
    stt = speechmatics.STT(
        language=INPUT_LANGUAGE,
        enable_partials=True,
        end_of_utterance_mode=EndOfUtteranceMode.NONE,
    )

    stt_stream = stt.stream()
    audio_stream = rtc.AudioStream(track)
    buffer = TranscriptBuffer()

    async def push_audio():
        async for audio_event in audio_stream:
            stt_stream.push_frame(audio_event.frame)
        stt_stream.end_input()

    async def process_stt_events():
        nonlocal buffer
        
        async for event in stt_stream:
            if event.type == SpeechEventType.START_OF_SPEECH:
                logger.info(f"🎤 START OF SPEECH [{participant.identity}]")

            elif event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                if event.alternatives:
                    text = event.alternatives[0].text
                    if not text:
                        continue
                    
                    buffer.full_text = text
                    uncommitted = buffer.uncommitted
                    
                    # Publish incomplete transcript if changed
                    if uncommitted and uncommitted != buffer.last_published_text:
                        await publish_buffer(ctx, participant, buffer, status="incomplete")
                        buffer.last_published_text = uncommitted
                    
                    # Check for commit
                    if buffer.find_commit_point():
                        committed = buffer.commit()
                        if committed:
                            # Publish complete transcript
                            await publish_complete(ctx, participant, committed, buffer.pair_id)
                            
                            # Parallel translation + queue TTS for all output languages
                            for lang in OUTPUT_LANGUAGES:
                                asyncio.create_task(
                                    translate_and_publish(
                                        ctx, participant, committed, buffer.pair_id, 
                                        lang, tts_queues
                                    )
                                )
                            
                            old_pair = buffer.pair_id
                            buffer.pair_id += 1
                            buffer.last_published_text = ""
                            logger.info(f"🔄 ROLLOVER: {old_pair} -> {buffer.pair_id}")

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
        uncommitted = buffer.uncommitted
        if uncommitted and uncommitted != buffer.last_published_text:
            logger.info(f"📤 FLUSH incomplete buffer: '{uncommitted}'")
            await publish_buffer(ctx, participant, buffer, status="incomplete")
        
        await stt_stream.aclose()
        logger.info(f"Transcription ended for {participant.identity}")


# =============================================================================
# Publishing Functions
# =============================================================================
async def publish_buffer(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    buffer: TranscriptBuffer,
    status: str,
):
    """Publish current uncommitted transcript to room."""
    text = buffer.uncommitted
    if not text.strip():
        return

    try:
        attributes = {
            "pair_id": str(buffer.pair_id),
            "status": status,
            "type": "transcript",
            "language": INPUT_LANGUAGE,
            "participant_identity": participant.identity,
        }
        await ctx.room.local_participant.send_text(
            text, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[{status.upper()}] pair={buffer.pair_id} lang={INPUT_LANGUAGE}: {text}")
    except Exception as e:
        logger.error(f"Failed to publish buffer: {e}")


async def publish_complete(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
):
    """Publish a completed transcript chunk to room."""
    if not text.strip():
        return

    try:
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "transcript",
            "language": INPUT_LANGUAGE,
            "participant_identity": participant.identity,
        }
        await ctx.room.local_participant.send_text(
            text, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[COMPLETE] pair={pair_id} lang={INPUT_LANGUAGE}: {text}")
    except Exception as e:
        logger.error(f"Failed to publish complete: {e}")


async def translate_and_publish(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
    target_lang: str,
    tts_queues: dict[str, asyncio.Queue],
):
    """
    Translate text to target language, publish translation, and queue TTS.
    TTS is queued (not direct) to ensure serialized playback.
    """
    try:
        # Translate
        translation = await translate_text(text, target_lang)
        
        # Publish translation text (immediate)
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "translation",
            "language": target_lang,
            "participant_identity": participant.identity,
            "original_text": text,
        }
        await ctx.room.local_participant.send_text(
            translation, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[TRANSLATED] pair={pair_id} lang={target_lang}: {translation}")
        
        # Queue TTS job for serialized playback
        await tts_queues[target_lang].put((translation, pair_id))
        logger.info(f"[TTS QUEUED] pair={pair_id} lang={target_lang}")
        
    except Exception as e:
        logger.error(f"Translation/TTS to {target_lang} failed for pair={pair_id}: {e}")


if __name__ == "__main__":
    cli.run_app(server)

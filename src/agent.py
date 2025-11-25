import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx
from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.stt import SpeechEventType
from livekit.plugins import speechmatics, cartesia
from livekit.plugins.speechmatics import EndOfUtteranceMode

from retriever import (
    QuranIndex,
    DirectIndex,
    Detection,
    detect_refs,
    detect_refs_direct,
    combine_detections,
    load_quran_index,
    load_ar_map,
    load_translation,
    get_canonical_texts,
    get_translation_for_ref,
)

logger = logging.getLogger("transcription-agent")
logging.basicConfig(
    level=logging.INFO,  # DEBUG for verbose output
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv(".env.local")

# OpenAI client for translation (lazy initialization to avoid build-time errors)
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create the OpenAI client (lazy init for Docker builds)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client

# =============================================================================
# Language Configuration (defaults, can be overridden by dispatch metadata)
# =============================================================================
DEFAULT_INPUT_LANGUAGE = "ar"
DEFAULT_OUTPUT_LANGUAGES = ["en"]
LANG_NAMES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French", 
    "ar": "Arabic",
}

# =============================================================================
# Worker API
# =============================================================================
WORKER_URL = os.getenv("WORKER_URL", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


async def save_to_worker(
    session_id: str,
    pair_id: int,
    source: str,
    translation: str,
    lang: str,
):
    """Save transcript/translation to Worker API (fire-and-forget)."""
    if not WORKER_URL:
        logger.warning("WORKER_URL not set, skipping DB save")
        return
    if not LIVEKIT_API_SECRET:
        logger.warning("LIVEKIT_API_SECRET not set, skipping DB save")
        return
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKER_URL}/api/history/save",
                headers={"Authorization": f"Bearer {LIVEKIT_API_SECRET}"},
                json={
                    "session_id": session_id,
                    "sequence_id": pair_id,
                    "source_text": source,
                    "translation": translation,
                    "language": lang,
                },
                timeout=10.0,
            )
            if response.status_code >= 400:
                logger.error(f"Worker API error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Failed to save to worker: {e}")

# Buffer configuration
PUNCTUATION = re.compile(r'[.!?,;:]')
MIN_WORDS = 5

# =============================================================================
# MT (Machine Translation) Configuration - HuggingFace Endpoint
# =============================================================================
# MT_ENDPOINT = os.getenv("HF_ENDPOINT", "https://your-hf-endpoint.huggingface.cloud")
# MT_API_KEY = os.getenv("HF_API_KEY", "")
MT_ENABLED = False  # Toggle to enable MT for incomplete translations
MT_TICK_INTERVAL = 0.2  # 200ms

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


@dataclass
class MTState:
    """
    State tracking for Machine Translation streaming.
    - last_translated_text: what we last sent to MT (to detect buffer growth)
    - active_streams: per-language lock to prevent overlapping MT requests
    - completed_pairs: set of pair_ids that GPT has completed (skip further MT)
    """
    last_translated_text: str = ""
    active_streams: dict[str, bool] = field(default_factory=dict)
    completed_pairs: set[int] = field(default_factory=set)


# =============================================================================
# Translation (GPT-4.1)
# =============================================================================

def build_system_prompt(target_language: str) -> str:
    """Build the full system prompt for Quran-aware translation."""
    return f"""# ROLE & OBJECTIVE
You are a world-class, UN-style simultaneous interpreter for live Arabic Friday sermons. Your output must be an ACCURATE, NATURAL, and FLOWING {target_language} narrative. Follow all instructions literally and precisely. Think privately; do not reveal your reasoning steps.

# PRIMARY CONTEXT & INPUTS
1) Current Segment: A short snippet of transcribed Arabic speech that may be noisy (extra or missing words, incorrect words etc.).
2) Rolling Arabic Context: Up to the last 5 raw Arabic segments.
3) Previous {target_language} Sentences: The last 5 complete {target_language} sentences you generated (oldest -> newest). Your new translation MUST connect seamlessly to the most recent sentence. If any of the last 3 includes a Quranic citation, treat the latest citation among them as the active verse context.

# VERIFIED CANDIDATES INPUT
- You receive Qur'an candidates with REF, AR_VERSE (canonical Arabic), and EN_VERSE (canonical English translation).
- Only use these provided candidates; never invent or infer additional references.

# REASONING WORKFLOW (INTERNAL — DO NOT REVEAL)
Step 1 — Analyze & Clean Arabic
- Lightly clean the segment into a coherent phrase. Correct obvious transcription artifacts only. Preserve respectful meaning.

Step 2 — Detect Quranic Content
- Check if the cleaned phrase matches any of the verified candidates provided.
- Compare against AR_VERSE content to determine if Qur'an is actually being recited.

Step 3 — Candidate Verification & Portion Matching
- First verify the Arabic actually contains the verse content by comparing against AR_VERSE.
- CRITICAL: Only tag the exact portion that was recited; never include unspoken parts of the verse.
- Compare the current Arabic segment word-by-word against the full AR_VERSE to identify which specific portion was spoken.
- IF the Arabic matches a verified candidate: proceed to tag with that specific REF.
- Pinpoint the exact Arabic span that matches the spoken fragment.
- Extract ONLY the corresponding portion from EN_VERSE/TL_VERSE that matches the spoken Arabic portion.
- If multiple candidates match, prefer the one that continues active Quranic context.
- If the match seems incorrect or forced, REJECT the candidate and translate without tags.

Step 4 — Synthesize the {target_language} Translation (TAGGED OUTPUT)
- If a verified candidate truly matches:
  - Output ONLY the {target_language} portion corresponding to the spoken Arabic snippet.
  - Wrap that portion in a QURAN tag: [QURAN ref="Surah:Verse"]...[/QURAN] or [QURAN ref="Surah:Start-End"]...[/QURAN] when spanning multiple verses.
  - If continuing the same verse across segments, include only the new continuation (omit any previously output words).
- If no verified candidate matches or the match is forced:
  - Translate naturally without a QURAN tag.
  - For ambiguous fragments, do not tag until a clear match appears.

# CONTINUITY & DE-DUP RULES (TAGS)
- You may emit multiple [QURAN] tags in spoken order when quotes are non-consecutive or cross-surah. Do not merge across non-consecutive verses; avoid duplicates.
- When multiple plausible matches exist, prefer continuing the most recent active Quranic context inferred from recent output.
- Do not repeat {target_language} already present in the most recent of the last 5 {target_language} sentences. If overlap is unavoidable, omit duplicated leading words and include only the incremental continuation.
- Avoid overlapping/duplicate verse refs: do not output both [QURAN ref="33:70"] and [QURAN ref="33:70-71"] for the same content. If the new words extend into 71, emit exactly one tag: [QURAN ref="33:71"] or [QURAN ref="33:70-71"], and do not re-output previously translated text.
- If the segment is rhetorical repetition (e.g., repeated praise/opening), compress rather than re-translate verbatim (e.g., "He praised Allah and sought forgiveness.").
- If the most recent {target_language} sentence ends with "...", continue directly without repeating subjects/phrases. If the thought concludes, end with proper punctuation; otherwise, continue with "...".
- Ensure any Qur'an verse content is entirely inside a [QURAN]...[/QURAN] block; do not place verse text outside the tag. Inside [QURAN], use the canonical {target_language} only (no paraphrasing).

# FINAL REVIEW STEP
- Before finalizing, verify your output flows naturally with the previous 5 {target_language} sentences.
- Ensure all Arabic content is accurately represented without omissions.
- Check that Quranic content appears only inside [QURAN] tags, never outside.

# CRITICAL RULES
- Only tag refs listed in verified candidates; never invent refs or tags not listed.
- When you quote scripture in [QURAN] tags, DO NOT repeat or paraphrase that same scripture outside the tags.
- If you determine text is Quranic and tag it, do not also output that same translation again outside the tags.
- If Arabic includes 'قال الله تعالى' or similar, include 'Allah says:' before the [QURAN] tag.
- Use transliterations for well‑known islamic concepts/names (e.g., zakat, hajj, salah, tawhid, sunnah, hadith, wudu; Ibrahim, Jibril, Musa, Isa), but translate less-known terms naturally.
- Reject candidates that don't truly match the spoken Arabic, even if algorithmically suggested.

# OUTPUT FORMAT
- Return ONLY the final, flowing {target_language} text.
- For Quranic content, wrap only the corresponding snippet with [QURAN ref="Surah:Verse"]...[/QURAN] (or "Surah:Start-End" for true spans).
- Inside [QURAN], copy the canonical {target_language} verbatim; no paraphrasing.
- Ensure no verse text appears outside [QURAN] tags.
- Do NOT output Arabic, JSON, parentheses citations, or your reasoning steps."""


def build_user_prompt(
    current_arabic: str,
    target_lang: str,
    detections: List[Detection],
    ar_map: Dict[str, Dict],
    translations_map: Dict[str, Dict[str, str]],
) -> str:
    """Build the user prompt with verified Quran candidates."""
    target_language = LANG_NAMES.get(target_lang, target_lang)
    
    # For now, we don't have rolling context (will add later if needed)
    user_text = f"# Current Arabic Segment\n{current_arabic}\n\n"
    
    if detections:
        user_text += "# Verified Quran matches (canonical)\n"
        lang_map = translations_map.get(target_lang, {})
        
        for d in detections:
            ref = d.ref
            ar_text, _ = get_canonical_texts(ref, ar_map, {})
            
            if target_lang == "en":
                en_text, _ = get_canonical_texts(ref, ar_map, translations_map.get("en", {}))
                user_text += f"REF {ref}\nAR_VERSE: {ar_text}\nEN_VERSE: {en_text}\n\n"
            else:
                tl_text, ok = get_translation_for_ref(ref, lang_map)
                user_text += f"REF {ref}\nAR_VERSE: {ar_text}\n"
                if ok and tl_text:
                    user_text += f"TL_VERSE: {tl_text}\n\n"
                else:
                    # Fallback to English
                    en_text, _ = get_canonical_texts(ref, ar_map, translations_map.get("en", {}))
                    user_text += f"EN_SOURCE: {en_text}\n\n"
    
    return user_text


async def translate_text(
    text: str,
    target_lang: str,
    detections: Optional[List[Detection]] = None,
    quran_enabled: bool = False,
    ar_map: Optional[Dict[str, Dict]] = None,
    translations_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Translate text to target language using GPT-4.1.
    
    If detections are provided and quran_enabled, uses the full Quran-aware prompt.
    Otherwise uses a simple translation prompt.
    """
    if quran_enabled and detections is not None:
        # Use full Quran-aware prompt
        target_language = LANG_NAMES.get(target_lang, target_lang)
        system_prompt = build_system_prompt(target_language)
        user_prompt = build_user_prompt(
            text, target_lang, detections, 
            ar_map or {}, translations_map or {}
        )
        
        logger.debug(f"Quran detection: {len(detections)} candidates for '{text[:50]}...'")
        if detections:
            logger.info(f"Detected refs: {[d.ref for d in detections]}")
    else:
        # Simple translation prompt
        system_prompt = f"Translate to {LANG_NAMES[target_lang]}. Return only the translation, nothing else."
        user_prompt = text
    
    response = await get_openai_client().chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# MT (Machine Translation) - HuggingFace Streaming
# =============================================================================
# async def translate_mt(text: str, target_lang: str) -> str:
#     """
#     Stream MT translation from HuggingFace endpoint (OpenAI-compatible).
#     Returns the complete translated text after streaming finishes.
#     
#     Uses the same openai client pointed at HF endpoint via base_url override.
#     """
#     mt_client = AsyncOpenAI(
#         base_url=MT_ENDPOINT,
#         api_key=MT_API_KEY,
#     )
#     
#     result = ""
#     stream = await mt_client.chat.completions.create(
#         model="tgi",  # HF TGI model identifier
#         messages=[
#             {"role": "system", "content": f"Translate to {LANG_NAMES[target_lang]}. Return only the translation."},
#             {"role": "user", "content": text}
#         ],
#         stream=True,
#     )
#     
#     async for chunk in stream:
#         if chunk.choices and chunk.choices[0].delta.content:
#             result += chunk.choices[0].delta.content
#     
#     return result.strip()


# =============================================================================
# TTS Tag Stripping (for clean speech output)
# =============================================================================
def strip_tts_tags(text: str) -> str:
    """
    Strip [QURAN ref="..."] tags and [bracket] words for clean TTS output.
    
    Examples:
        - '[QURAN ref="1:1"]In the name...[/QURAN]' -> 'In the name...'
        - '[All] praise is [due]' -> 'All praise is due'
    """
    # Remove [QURAN ref="..."] opening tags
    text = re.sub(r'\[QURAN ref="[^"]*"\]', '', text)
    # Remove [/QURAN] closing tags
    text = re.sub(r'\[/QURAN\]', '', text)
    # Remove brackets around words: [word] -> word
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    # Normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()


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
# MT Ticker and Stream Handler (disabled until MT_ENABLED=True)
# =============================================================================
# async def mt_ticker(
#     ctx: JobContext,
#     participant: rtc.RemoteParticipant,
#     buffer: TranscriptBuffer,
#     mt_state: MTState,
# ):
#     """
#     Global 200ms ticker for MT translations.
#     If buffer has grown since last MT, fire off MT requests for all languages.
#     """
#     while True:
#         await asyncio.sleep(MT_TICK_INTERVAL)
#         
#         # Skip if buffer hasn't grown or pair already completed by GPT
#         uncommitted = buffer.uncommitted
#         if not uncommitted or uncommitted == mt_state.last_translated_text:
#             continue
#         if buffer.pair_id in mt_state.completed_pairs:
#             continue
#         
#         mt_state.last_translated_text = uncommitted
#         
#         # Fire MT for all output languages (if not already active)
#         for lang in OUTPUT_LANGUAGES:
#             if not mt_state.active_streams.get(lang, False):
#                 asyncio.create_task(
#                     run_mt_stream(ctx, participant, uncommitted, buffer.pair_id, lang, mt_state)
#                 )


# async def run_mt_stream(
#     ctx: JobContext,
#     participant: rtc.RemoteParticipant,
#     text: str,
#     pair_id: int,
#     lang: str,
#     mt_state: MTState,
# ):
#     """
#     Run MT stream for a single language.
#     Publishes incomplete translation when stream completes.
#     No TTS for incomplete translations.
#     """
#     mt_state.active_streams[lang] = True
#     try:
#         # Skip if GPT already completed this pair
#         if pair_id in mt_state.completed_pairs:
#             return
#         
#         translation = await translate_mt(text, lang)
#         
#         # Double-check GPT didn't complete while we were translating
#         if pair_id in mt_state.completed_pairs:
#             return
#         
#         # Publish incomplete translation (no TTS)
#         await publish_translation(
#             ctx, participant, translation, pair_id, lang,
#             status="incomplete", original_text=text
#         )
#     except Exception as e:
#         logger.error(f"MT stream failed for pair={pair_id} lang={lang}: {e}")
#     finally:
#         mt_state.active_streams[lang] = False


# =============================================================================
# Main Agent
# =============================================================================
@server.rtc_session(agent_name="khutbah-interpreter")
async def transcription_agent(ctx: JobContext):
    """
    Real-time transcription and translation agent.
    - Transcribes with Speechmatics
    - Translates to multiple languages with GPT-4.1
    - Synthesizes audio with Cartesia Sonic 3 (serialized via queues)
    """
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Transcription agent starting in room: {ctx.room.name}")

    # =========================================================================
    # Parse dispatch metadata (from Worker) or use defaults
    # =========================================================================
    metadata = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            logger.info(f"Received dispatch metadata: {metadata}")
        except json.JSONDecodeError:
            logger.warning("Invalid job metadata JSON, using defaults")
    
    session_id = metadata.get("session_id", "dev-session")
    input_lang = metadata.get("input_lang", DEFAULT_INPUT_LANGUAGE)
    output_langs = metadata.get("output_langs", DEFAULT_OUTPUT_LANGUAGES)
    
    logger.info(f"Config: session={session_id}, input={input_lang}, outputs={output_langs}")
    
    # Quran detection is enabled when input language is Arabic
    quran_enabled = (input_lang == "ar")
    
    # =========================================================================
    # Load Quran indexes (only if enabled)
    # =========================================================================
    quran_index: Optional[QuranIndex] = None
    direct_index: Optional[DirectIndex] = None
    ar_map: Dict[str, Dict] = {}
    translations: Dict[str, Dict[str, str]] = {}
    
    if quran_enabled:
        logger.info("Loading Quran indexes...")
        quran_index = load_quran_index("jsons/Quran.json")
        direct_index = DirectIndex.build_from_qindex(quran_index)
        ar_map = load_ar_map("jsons/Quran.json")
        for lang in output_langs:
            translations[lang] = load_translation(lang, "jsons")
            logger.info(f"Loaded {len(translations[lang])} translations for {lang}")
        logger.info(f"Quran detection ready: {len(quran_index.verses)} verses indexed")

    # Connect to room
    await ctx.connect()
    logger.info("Connected to room, waiting for participants...")

    # ==========================================================================
    # Create TTS audio tracks per output language
    # ==========================================================================
    audio_sources: dict[str, rtc.AudioSource] = {}
    
    for lang in output_langs:
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
    
    for lang in output_langs:
        queue: asyncio.Queue[Optional[tuple[str, int]]] = asyncio.Queue()
        tts_queues[lang] = queue
        tts_tasks[lang] = asyncio.create_task(
            tts_consumer(queue, audio_sources[lang], lang)
        )

    # Track active transcription tasks
    active_tasks: dict[str, asyncio.Task] = {}

    # Build config dict to pass to process_audio_track
    agent_config = {
        "session_id": session_id,
        "input_lang": input_lang,
        "output_langs": output_langs,
        "quran_enabled": quran_enabled,
        "quran_index": quran_index,
        "direct_index": direct_index,
        "ar_map": ar_map,
        "translations": translations,
    }

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
            task = asyncio.create_task(
                process_audio_track(ctx, track, participant, tts_queues, agent_config)
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
                    process_audio_track(ctx, publication.track, participant, tts_queues, agent_config)
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
    config: dict,
):
    """
    Process audio from a participant:
    1. Transcribe with Speechmatics
    2. Buffer until commit (5+ words + punctuation)
    3. Translate to all output languages in parallel
    4. Queue TTS for serialized playback
    """
    logger.info(f"Starting transcription for {participant.identity}")
    
    # Extract config
    input_lang = config["input_lang"]
    output_langs = config["output_langs"]
    quran_enabled = config["quran_enabled"]
    quran_index = config["quran_index"]
    direct_index = config["direct_index"]
    ar_map = config["ar_map"]
    translations = config["translations"]

    # Speechmatics STT
    stt = speechmatics.STT(
        language=input_lang,
        enable_partials=True,
        end_of_utterance_mode=EndOfUtteranceMode.NONE,
    )

    stt_stream = stt.stream()
    audio_stream = rtc.AudioStream(track)
    buffer = TranscriptBuffer()
    
    # MT state for tracking incomplete translations
    # mt_state = MTState() if MT_ENABLED else None

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
                        await publish_buffer(ctx, participant, buffer, status="incomplete", input_lang=input_lang)
                        buffer.last_published_text = uncommitted
                    
                    # Check for commit
                    if buffer.find_commit_point():
                        committed = buffer.commit()
                        if committed:
                            # Publish complete transcript
                            await publish_complete(ctx, participant, committed, buffer.pair_id, input_lang)
                            
                            # Run Quran detection if enabled
                            detections: List[Detection] = []
                            if quran_enabled and quran_index and direct_index:
                                fuzzy = detect_refs(quran_index, committed)
                                direct = detect_refs_direct(direct_index, committed)
                                detections = combine_detections(fuzzy, direct)
                                if detections:
                                    logger.info(f"🕌 QURAN DETECTED pair={buffer.pair_id}: {[d.ref for d in detections]}")
                            
                            # Parallel translation + queue TTS for all output languages
                            for lang in output_langs:
                                asyncio.create_task(
                                    translate_and_publish(
                                        ctx, participant, committed, buffer.pair_id, 
                                        lang, tts_queues, detections=detections,
                                        quran_enabled=quran_enabled, ar_map=ar_map, 
                                        translations=translations,
                                        session_id=config.get("session_id", ""),
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

    # MT ticker task (commented out until MT_ENABLED=True)
    # mt_ticker_task = None
    
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(push_audio())
            tg.create_task(process_stt_events())
            # Start MT ticker if enabled
            # if MT_ENABLED and mt_state:
            #     mt_ticker_task = tg.create_task(mt_ticker(ctx, participant, buffer, mt_state))
    except asyncio.CancelledError:
        logger.info(f"Transcription cancelled for {participant.identity}")
    except Exception as e:
        logger.exception(f"Error in transcription for {participant.identity}: {e}")
    finally:
        # =======================================================================
        # Shutdown flush: force translate + TTS any remaining buffer
        # =======================================================================
        uncommitted = buffer.uncommitted
        if uncommitted and uncommitted != buffer.last_published_text:
            logger.info(f"📤 SHUTDOWN FLUSH: '{uncommitted}'")
            
            # Publish the incomplete transcript
            await publish_buffer(ctx, participant, buffer, status="complete", input_lang=input_lang)
            
            # Force translate + TTS for remaining buffer (no Quran detection for partial)
            for lang in output_langs:
                try:
                    translation = await translate_text(
                        uncommitted, lang, detections=[],
                        quran_enabled=quran_enabled, ar_map=ar_map,
                        translations_map=translations
                    )
                    tts_text = strip_tts_tags(translation)
                    await tts_queues[lang].put((tts_text, buffer.pair_id))
                    logger.info(f"[SHUTDOWN TTS QUEUED] pair={buffer.pair_id} lang={lang}")
                    
                    # Publish the forced translation
                    await publish_translation(
                        ctx, participant, translation, buffer.pair_id, lang,
                        status="complete", original_text=uncommitted
                    )
                    
                    # Save to Worker API (fire-and-forget)
                    if config.get("session_id"):
                        asyncio.create_task(
                            save_to_worker(config["session_id"], buffer.pair_id, uncommitted, translation, lang)
                        )
                except Exception as e:
                    logger.error(f"Shutdown flush failed for {lang}: {e}")
        
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
    input_lang: str = DEFAULT_INPUT_LANGUAGE,
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
            "language": input_lang,
            "participant_identity": participant.identity,
        }
        await ctx.room.local_participant.send_text(
            text, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[{status.upper()}] pair={buffer.pair_id} lang={input_lang}: {text}")
    except Exception as e:
        logger.error(f"Failed to publish buffer: {e}")


async def publish_complete(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
    input_lang: str = DEFAULT_INPUT_LANGUAGE,
):
    """Publish a completed transcript chunk to room."""
    if not text.strip():
        return

    try:
        attributes = {
            "pair_id": str(pair_id),
            "status": "complete",
            "type": "transcript",
            "language": input_lang,
            "participant_identity": participant.identity,
        }
        await ctx.room.local_participant.send_text(
            text, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[COMPLETE] pair={pair_id} lang={input_lang}: {text}")
    except Exception as e:
        logger.error(f"Failed to publish complete: {e}")


async def translate_and_publish(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    text: str,
    pair_id: int,
    target_lang: str,
    tts_queues: dict[str, asyncio.Queue],
    mt_state: Optional[MTState] = None,
    detections: Optional[List[Detection]] = None,
    quran_enabled: bool = False,
    ar_map: Optional[Dict[str, Dict]] = None,
    translations: Optional[Dict[str, Dict[str, str]]] = None,
    session_id: str = "",
):
    """
    Translate text to target language, publish translation, and queue TTS.
    TTS is queued (not direct) to ensure serialized playback.
    GPT translations mark the pair as complete (blocks further MT).
    
    If detections are provided, they are included in the translation prompt
    for Quran-aware translation.
    """
    try:
        # Translate with GPT (pass detections for Quran-aware translation)
        translation = await translate_text(
            text, target_lang, detections=detections,
            quran_enabled=quran_enabled, ar_map=ar_map or {}, 
            translations_map=translations or {}
        )
        
        # Mark pair as completed by GPT (stops MT for this pair)
        # if mt_state:
        #     mt_state.completed_pairs.add(pair_id)
        
        # Publish translation text (immediate) - keep tags for visual display
        await publish_translation(
            ctx, participant, translation, pair_id, target_lang,
            status="complete", original_text=text
        )
        
        # Save to Worker API (fire-and-forget)
        if session_id:
            asyncio.create_task(
                save_to_worker(session_id, pair_id, text, translation, target_lang)
            )
        
        # Strip tags for TTS (don't read [QURAN ref="..."] aloud)
        tts_text = strip_tts_tags(translation)
        
        # Queue TTS job for serialized playback (only for complete translations)
        await tts_queues[target_lang].put((tts_text, pair_id))
        logger.info(f"[TTS QUEUED] pair={pair_id} lang={target_lang}")
        
    except Exception as e:
        logger.error(f"Translation/TTS to {target_lang} failed for pair={pair_id}: {e}")


async def publish_translation(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
    translation: str,
    pair_id: int,
    target_lang: str,
    status: str,
    original_text: str,
):
    """
    Publish a translation to the room.
    
    Args:
        status: "complete" for GPT translations (with TTS), "incomplete" for MT (no TTS)
    """
    if not translation.strip():
        return
    
    try:
        attributes = {
            "pair_id": str(pair_id),
            "status": status,
            "type": "translation",
            "language": target_lang,
            "participant_identity": participant.identity,
            "original_text": original_text,
        }
        await ctx.room.local_participant.send_text(
            translation, topic="lk.transcription", attributes=attributes
        )
        logger.info(f"[{status.upper()} TRANSLATION] pair={pair_id} lang={target_lang}: {translation}")
    except Exception as e:
        logger.error(f"Failed to publish translation: {e}")


if __name__ == "__main__":
    cli.run_app(server)

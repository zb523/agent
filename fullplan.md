High Level Flow
1. Initialization & Dispatch
Parse Metadata: Read input/output languages and domain flags (Quran, Medical, Legal) from the job context.
Setup Tracks: Initialize audio output tracks for the target language.
2. Presence Signaling
Notify Backend: Immediately send an HTTP POST to the worker endpoint confirming the Agent is connected and ready.
Enable Listeners: This signal triggers the frontend/worker to allow listeners to join the room.
3. STT & Buffering
Ingest: Connect Speechmatics to the audio stream.
Buffer Logic: Accumulate incoming tokens into a buffer.
Commit Triggers: Commit the buffer when min_words >= 5 AND punctuation is detected (or soft limit reached).
UI Feed: Publish the raw transcript one word at a time to the LiveKit data channel as they arrive.
4. Fast Translation Track (Visual Subtitles)
Trigger: As the buffer accumulates (before commit).
Process: Stream the current buffer to Hunyuan-MT-7B.
Output: Stream the resulting text to the LiveKit data channel immediately for real-time subtitles.
5. Smart Translation Track (Voice Output)
Trigger: When the buffer commit logic fires (complete thought).
Prompting: Send the committed text to GPT-4.1. Select the specific System Prompt based on domain flags (e.g., if has_quran=true, use the Quran-specific prompt).
Context: Include the partial output from the Fast Track (Hunyuan) as context for the Smart Track.
Output: Generate the final, accurate text translation.
6. TTS Generation & Streaming
Synthesis: Send the GPT-4.1 output text to Cartesia TTS (using the sonic-3 model via the Bytes API).
Playback: Stream the received audio bytes directly to the LiveKit room’s audio track.
7. Persistence
Save: Post the final Transcript (TR) and Translation (TL) payload to the worker endpoint to be saved in Supabase.
8. Session Cleanup
Flush: On session end/disconnect, take whatever is left in the buffer, force a translation, and send it to the DB to ensure no data loss.
Phase 1: Infrastructure & Connectivity
Objective: Initialize the Agent environment, parse the configuration passed by the Worker, and establish the presence handshake.
Step 1.1: Project Scaffolding
Action: Initialize the Python environment.
Dependencies: livekit-agents, livekit-plugins-speechmatics, livekit-plugins-cartesia, openai, elasticsearch, httpx, python-dotenv.
Config: Create .env.local containing LIVEKIT_URL, LIVEKIT_API_SECRET, OPENAI_API_KEY, CARTESIA_API_KEY, WORKER_URL, WORKER_SECRET.
Verification: Run scripts/check_env.py to ensure all keys load correctly.
Step 1.2: The Agent Class (ContextualInterpreter)
Action: Create agent.py and define class ContextualInterpreter(livekit.agents.Agent).
Logic:
Inherit from base Agent.
State Initialization:
self.buffer = []: Token accumulator.
self.turn_counter = 0: The Sequence ID (Crucial for the Frontend "Zipper").
self.config = {}: To store session metadata.
self.http_client: httpx.AsyncClient (shared for all requests).
Docs Reference: LiveKit Agent Base Class
Step 1.3: Entrypoint & Metadata (Explicit Dispatch)
Action: Create main.py with async def entrypoint(ctx: JobContext).
Logic:
Primary Source: Read ctx.job.metadata (JSON string).
Extract: session_id, input_lang, output_langs.
Fallback: If metadata is empty (local dev), default to {"input_lang": "ar", "output_langs": ["en"], "session_id": "dev-session"}.
Store: Save to self.config.
Why: The Cloudflare Worker already queried the DB when it dispatched this agent; we trust its data to save latency.
Verification: Run python main.py dev. Check console logs to see the config loaded from defaults.
Step 1.4: Presence Signaling (The Handshake)
Action: Fire a webhook to the Worker immediately upon joining.
Logic: await self.http_client.post(WORKER_URL + "/api/sessions/agent-present", json={"session_id": self.config['session_id']}).
Why: This flips the frontend UI from "Waiting..." to "Live".
Verification: Check Worker logs to confirm the POST was received.

Phase 2: The "Ear" (Ingestion & Semantic Buffering)
Objective: Capture audio, manage the text stream, and generate the critical Sequence ID.
Step 2.1: Audio Subscription
Action: In agent.py, listen for track_subscribed.
Logic: If kind == "audio", initialize speechmatics.STT.
Verification: Speak into mic; ensure console logs "Audio track subscribed".
Step 2.2: Speechmatics Integration
Action: Configure speechmatics.STT(operating_point="enhanced", enable_partials=True).
Logic: Start the async loop to consume events.
Step 2.3: Buffering & Sequence Tracking
Action: Implement handle_transcription(event).
Logic:
Partial Event: Do not buffer. Pass to Phase 7 (Fast Track).
Final Event: Append tokens to self.buffer.
Verification: Speak. Watch console for partial updates vs final commits.
Step 2.4: The Semantic Commit Trigger
Action: Implement should_commit().
Logic:
Trigger: len(buffer) > 6 AND ends with .?! OR len(buffer) > 25.
Action:
Increment: self.turn_counter += 1.
Return: The joined text (Query) + The new turn_counter (Sequence ID).
Clear: self.buffer.
Verification: Speak a full sentence. Assert turn_counter goes 0 -> 1.

Phase 3: The Brain (Context Retrieval)
Objective: Fetch Quranic context if the input language is Arabic.
Step 3.1: Elasticsearch Service
Action: Create services/retriever.py.
Logic: async def get_quranic_context(query).
Query: Simple match query against Quran index. Return top result.
Verification: Test script with query "Bismillah".
Step 3.2: Hot Path Integration
Action: Inside the Agent loop (after Commit Trigger).
Logic:
If self.config['input_lang'] == 'ar': Call Retriever.
Store result in current_context.

Phase 4: Synthesis (Translation)
Objective: Generate translations grounded in the retrieved context.
Step 4.1: Prompt Builder
Action: Create services/llm.py.
Logic: build_prompt(source, context, target_lang).
Constraint: "Use the provided context strictly if it contains a Quran verse."
Step 4.2: GPT Generation
Action: In agent.py, iterate through self.config['output_langs'].
Logic:
Call OpenAI (non-streaming).
Async: Use asyncio.gather to run English and French generations in parallel.
Verification: Console log the generated translations.

Phase 5: The Mouth (TTS)
Objective: Stream audio to the room.
Step 5.1: Cartesia Setup
Action: Initialize cartesia.TTS in entrypoint.
Logic: Publish LocalAudioTrack to the room.
Step 5.2: Streaming
Action: Feed GPT output to Cartesia.
Logic: Write audio frames to the LocalAudioTrack.

Phase 6: Persistence (The Database Write)
Objective: Save data via the Worker API using the "Two-Step" method to prevent race conditions.
Step 6.1: The Anchor (Transcript)
Action: Immediately after Phase 2.4 (Commit Trigger).
Logic:
Payload: { "session_id": ..., "sequence_id": self.turn_counter, "source_text": ..., "type": "transcript" }
Call: POST {WORKER_URL}/api/history/save.
Response: Expect { "transcript_id": "uuid-123" }.
Store: Save this transcript_id temporarily in a variable.
Step 6.2: The Payload (Translations)
Action: As soon as Phase 4 (GPT) finishes for each language.
Logic:
Payload: { "transcript_id": "uuid-123", "language": "en", "text": "...", "type": "translation" }
Call: POST {WORKER_URL}/api/history/save.
Why: This allows the English text to be saved (and appear on Frontend) even if French is still generating.
Verification: Mock the Worker response. Ensure the Agent sends the Transcript request first, waits for the ID, then sends Translation requests.

Phase 7: The Fast Track (Visuals)
Objective: Send low-latency subtitles with the Sequence ID for the Frontend "Zipper".
Step 7.1: Hunyuan Client
Action: Create services/fast_mt.py.
Logic: HTTP wrapper for Hunyuan Endpoint.
Step 7.2: Data Channel Publishing
Action: On partial STT events.
Logic:
Send to Hunyuan.
Publish to LiveKit Data Channel.
Topic: transcription.
Payload: { "type": "partial", "text": "...", "seq": self.turn_counter }.
Critical: The seq field allows the Frontend to map this partial text to the correct chat bubble.
Verification: Monitor Data Channel in LiveKit sample app. Ensure seq matches the current turn.

Phase 8: Deployment & Lifecycle
Objective: Graceful shutdown.
Step 8.1: Shutdown Hook
Action: ctx.add_shutdown_callback(shutdown_hook).
Logic:
Flush buffer (Force Commit).
Send POST {WORKER_URL}/api/sessions/stop (or agent-present false) to clean up.
Step 8.2: Docker
Action: Create Dockerfile.
Command: lk agent deploy.


# Pipelines — step by step, with models

Current as of Phase 4 (wake word, voice-only identity, local LLM, Hermes). Companion pages:
[architecture](architecture.md) (the big picture and run loop),
[configuration](configuration.md) (every knob), [testing](testing.md).

## Models at a glance

| Role | Model | Where it runs | Code | Configurable via |
|---|---|---|---|---|
| Chat / RAG answers, tool calls, memory distillation | `Qwen3.6-35B-A3B-oQ6e-mtp` (MoE, ~3B active, oQ6e) on oMLX at `http://127.0.0.1:8001/v1`; OpenAI `gpt-5-mini` when `LLM_BASE_URL` is unset | local server (or cloud) | `brain/chat.py`, `brain/llm.py` | `REACHY_VEC_LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_DISABLE_THINKING`, `LLM_TIMEOUT_S` |
| Delegated multi-step tasks (`ask_hermes`) | the Hermes CLI, `reachy` profile (its own model; also the local model by default) | local subprocess | `brain/hermes.py` | `REACHY_VEC_HERMES_*` |
| Text embeddings (docs, memories, queries) | `BAAI/bge-small-en-v1.5`, 384-dim normalized; queries get the BGE instruction prefix (`REACHY_VEC_EMBEDDING_QUERY_PREFIX`) | local (sentence-transformers) | `store/embeddings.py` | `REACHY_VEC_EMBEDDING_MODEL` |
| Speech-to-text (default) | faster-whisper `base.en`, int8 | local | `audio/listen.py:MicTranscriber` | `REACHY_VEC_STT_MODEL` |
| Speech-to-text (alternative) | `gpt-4o-transcribe` (OpenAI API) | cloud | `audio/listen.py:OpenAITranscriber` | `REACHY_VEC_STT_BACKEND=openai` |
| Voice activity detection | silero-vad (16 kHz, 512-sample frames) | local | `audio/listen.py:_AudioCapture` | not configurable |
| Speaker ID | speechbrain ECAPA-TDNN `spkrec-ecapa-voxceleb`, 192-dim | local | `perception/voice.py` | `REACHY_VEC_VOICE_THRESHOLD` |
| Text-to-speech | macOS `say` or Qwen3-TTS 0.6B voice clone (mlx-audio) | local | `audio/speak.py` | `REACHY_VEC_TTS_BACKEND`, `REACHY_VEC_TTS_MODEL`, `REACHY_VEC_VOICE_SAMPLE`; clone clarity/loudness (see below): `REACHY_VEC_TTS_TARGET_RMS`, `REACHY_VEC_TTS_PRESENCE_DB`, `REACHY_VEC_TTS_PRESENCE_HZ` |

All local models lazy-load on first use; `reachy-vec run` warms them (plus
the OpenAI TLS connection) at startup so the first conversation isn't slow. A one-token LLM warm-up call (`ChatBrain.warm_up`) checks the server is reachable.

> **Where the I/O runs:** selectable via `MEDIA_SOURCE` / `reachy-vec run
> --source`. `mac` (default on a machine with no robot media) reads the
> steps below from the Mac's own camera/mic/speaker; `robot` drives them from
> the Reachy Mini's camera/mic/speaker over the SDK (`RobotCamera`,
> `RobotAudioSource`, `RobotAudioSink` — robot audio out needs the `qwen-tts`
> backend). `auto` picks robot when the daemon offers media, else Mac. See
> [architecture.md → Known gaps](architecture.md#known-gaps--toward-a-real-robot-deploy).

> **Clone clarity/loudness** (`audio/speak.py:_process_loudness`): the Qwen
> clone's level and brightness track the (usually quiet, dull) reference
> sample, so each sentence is post-processed before playback — a presence
> high-shelf (`TTS_PRESENCE_DB` dB above `TTS_PRESENCE_HZ`) for intelligibility,
> then a boost toward `TTS_TARGET_RMS` followed by a tanh soft limiter that
> tames transients so the sustained speech reaches the target without clipping.
> Any of the three set to 0 disables that step (`say` backend is unaffected).
> `REACHY_VEC_TTS_TAP=<dir>` dumps each clip as played, to A/B the settings.

## 1. Knowledge ingestion

Two entry points writing into the same LanceDB `docs` table; queried
identically at runtime.

**`reachy-vec sync-mongo`** (`store/mongo_sync.py`) — MongoDB is the source
of truth:

1. Read every document in `aixlab.demos` (title, project, authors, tags,
   url, note). `aixlab.users` is never read.
2. Format each demo as plain text (`format_demo`). Mongo's own 1536-dim
   `embedding` field is **ignored** — everything lives in one local BGE space.

   Docs retrieval is hybrid: every `add_doc_chunks` write refreshes a
   native Lance FTS index on `text`, and per-turn search combines BM25
   with the query vector (RRF). Scores shown to the LLM stay cosine 0..1.
   A DB without the index (created before this feature) transparently
   falls back to vector-only until the next ingest/sync rebuilds it.

3. Chunk (`ingestion.chunk_text`: paragraphs packed to ≤1000 chars).
4. Embed all chunks with BGE **before** deleting anything, then replace all
   rows whose source starts with `demo: ` — idempotent, and a mid-run
   failure never leaves the store partially emptied.

**`reachy-vec ingest <path>`** (`store/ingestion.py`) — for extra local
docs: walk `.md`/`.txt` files, chunk the same way, embed, append with the
file path as `source`. Note: re-ingesting the same file appends duplicate
chunks (no replace logic, unlike sync-mongo).

## 2b. Voice identity pipeline (who is talking?)

`audio/listen.py` (utterance audio) → `perception/voice.py` →
`perception/fusion.py:from_voice`. Voice is the only identity signal.

1. Every utterance's raw audio rides along with its transcript
   (`Utterance(text, audio)`); audio shorter than `VOICE_MIN_UTTERANCE_S`
   (1 s) is "can't tell".
2. ECAPA embeds the audio (192-dim); k-NN majority vote (k=5, cosine)
   against the `voices` table, gated by `VOICE_THRESHOLD` (0.30) with the
   0.05 borderline margin under it.
3. `from_voice` decides the turn's identity:
   - match ≥ threshold → that person
   - below threshold → stranger (offered voice enrollment; side-effect
     tools refuse)
   - borderline or too-short audio → can't tell: answered anonymously, not
     greeted by name, never written to the store
4. Voice profiles come from two sources:
   - **enrolled**: a stranger says their name and a few spoken sentences
     (`ENROLL_VOICE_PHRASES`);
   - **passive**: matches ≥ threshold + 0.05 bank the utterance embedding
     (capped at `VOICE_PASSIVE_CAP` = 10 rows per person, oldest evicted) —
     the margin stands in for a second modality to confirm the match.
     Existing enrollees never re-enroll — profiles grow as they talk.
   Raw audio is never persisted, only embeddings.

## 3. Voice pipeline (what did they say?)

`audio/listen.py` — mic → VAD → STT:

1. Record 32 ms frames at 16 kHz from the default mic.
2. silero-vad scores each frame; speech = probability > 0.5.
3. `collect_utterance` accumulates from first speech until 0.8 s of quiet
   (or `SILENCE_TIMEOUT_S` with no speech at all → conversation ends).
4. Transcribe the utterance:
   - **local** (default): faster-whisper `base.en`, int8, English-only.
   - **openai**: WAV → `gpt-4o-transcribe`. More accurate, ~1 s latency.
   - Both are primed with the demo titles as an initial vocabulary prompt
     so demo names transcribe correctly.

## 4. Chat / RAG pipeline (one turn, `brain/chat.py:ChatBrain.respond`)

0. The per-turn voice identity (pipeline 2b) arrives as `identity=`;
   retrieval, `save_note`, and `send_message` all follow **the turn's
   speaker**, not whoever started the conversation. Anonymous turns get
   docs-only retrieval and side-effect tools politely refuse.
1. Embed the question with BGE (same space as the docs).
2. Retrieve locally, every turn (no retrieval tool-calls):
   - top-5 `docs` chunks with cosine scores, and
   - top-3 of **the turn speaker's** `memories` rows.
3. Build one user message: `[context + scores] + [memories] + "Name: question"`,
   appended to the visit's history (max 20 messages, reset per visit).
4. One **streaming** LLM call (local oMLX; thinking disabled via
   `enable_thinking=False`, and any `<think>` text is stripped by
   `ThinkFilter`) with the Reachy personality prompt and the tool list.
   Speaker names are used naturally, with a "(Speaker changed: now X.)"
   note when the voice changes mid-visit. Completed sentences are spoken as they arrive (~1 s to
   first sound). The model itself judges relevance: grounded answer naming
   the demo, or general knowledge with a casual "off the top of my head"
   signal (none for chit-chat).
5. If the model called tools: execute them, append results, make a
   **second** LLM call for the final spoken reply (tools cost one extra
   round-trip; sentence audio is suppressed while tool calls are pending).

**Tools** (all in `brain/chat.py`):

| Tool | Effect | Guardrails |
|---|---|---|
| `open_url` | opens a demo in the lab browser | http(s) only; runs only if the user's turn contains an `OPEN_URL_TRIGGER_WORDS` word (open, show, pull up, ...) as a whole word |
| `save_note` | stores a memory about the current person | requires a recognized visit; near-duplicates (cosine ≥ 0.97) skipped |
| `send_message` | queues a relay in the `messages` table | recipient must be enrolled (case-insensitive name match) |
| `get_weather` | live conditions via Open-Meteo (no API key) | location from `WEATHER_LAT/LON`; 5 s timeout |
| `get_time` | current local date and time | none (pure `datetime`) |
| `ask_hermes` | hands a multi-step task to the Hermes CLI (`reachy` profile, read-only + scheduling toolsets, run from `data/hermes_cwd`) | off unless `HERMES_ENABLED=true` and the `hermes` binary is on PATH; speaks "Let me look into that.", waits `HERMES_WAIT_S` (20 s); a slower task for a recognised speaker is delivered later as a `messages` row from "Hermes" (anonymous: dropped); one task at a time; `HERMES_TASK_TIMEOUT_S` (600 s) cap |
| `web_search` | live web answer via Tavily (`answer` field) | off unless `WEB_SEARCH=true` + `TAVILY_API_KEY`; 5 s timeout; prompt limits it to genuine current-info needs; Tavily 432/433 (out of credits) → spoken "used up my allowance", 429 → rate-limited message |

## 5. Memory pipeline (Phase 2)

- **Explicit**: the model calls `save_note` when asked to remember something.
- **Implicit**: when a conversation ends (`end_conversation`), one extra
  LLM call **per enrolled person who spoke** reviews the visit and
  distills up to 3 third-person notes each (or `NONE` for chit-chat).
- Both paths embed the note with BGE, skip near-duplicates (cosine ≥ 0.97
  vs the person's nearest existing memory), and write `memories` rows.
- Recall is automatic: step 2 of every chat turn searches this person's
  memories with the question vector.

## 6. Messenger pipeline (Phase 3)

1. Sender (recognized by voice) says "tell Bob …" → model calls `send_message` →
   row in `messages` with `delivered_at=""`.
2. At turn boundaries, when the recipient's voice is recognized, the loop
   speaks each pending message oldest-first ("By the way, Alice left you a
   message: …") and stamps `delivered_at`. Late Hermes results use the
   same path (sender "Hermes").

## 7. The Oracle loop tying it together

`brain/oracle.py:OracleLoop` is a synchronous state machine (all
dependencies injected; the whole thing runs in pytest on fakes):

asleep/idle (mic open, listen in `WAKE_LISTEN_S` windows for a `WAKE_WORDS`
entry) → wake (words after "Reachy" are the first question) → greet
(spoken at most every `GREET_COOLDOWN_S` per person) → deliver queued
messages → listen/think/speak loop → 30 s silence ends the visit (memories
distilled) → back to idle. Idle for `IDLE_SLEEP_S` (5 min) → sleep motion.
LLM errors produce a spoken apology and keep listening. Everything heard
and said is logged to `data/reachy.log` (delete to forget).

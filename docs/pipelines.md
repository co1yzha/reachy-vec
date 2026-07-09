# Pipelines — step by step, with models

Current as of Phase 2b (speaker ID + fusion). Companion pages:
[architecture](architecture.md) (the big picture and run loop),
[configuration](configuration.md) (every knob), [testing](testing.md).

## Models at a glance

| Role | Model | Where it runs | Code | Configurable via |
|---|---|---|---|---|
| Chat / RAG answers, tool calls, memory distillation | `gpt-5-mini` (OpenAI API) | cloud | `brain/chat.py` | `REACHY_VEC_LLM_MODEL` |
| Text embeddings (docs, memories, queries) | `BAAI/bge-small-en-v1.5`, 384-dim normalized; queries get the BGE instruction prefix (`REACHY_VEC_EMBEDDING_QUERY_PREFIX`) | local (sentence-transformers) | `store/embeddings.py` | `REACHY_VEC_EMBEDDING_MODEL` |
| Speech-to-text (default) | faster-whisper `base.en`, int8 | local | `audio/listen.py:MicTranscriber` | `REACHY_VEC_STT_MODEL` |
| Speech-to-text (alternative) | `gpt-4o-transcribe` (OpenAI API) | cloud | `audio/listen.py:OpenAITranscriber` | `REACHY_VEC_STT_BACKEND=openai` |
| Voice activity detection | silero-vad (16 kHz, 512-sample frames) | local | `audio/listen.py:_AudioCapture` | not configurable |
| Face detection + embedding | insightface `buffalo_s`, 512-dim | local (onnxruntime) | `perception/face.py` | not configurable (threshold is) |
| Speaker ID | speechbrain ECAPA-TDNN `spkrec-ecapa-voxceleb`, 192-dim | local | `perception/voice.py` | `REACHY_VEC_VOICE_THRESHOLD` |
| Text-to-speech | macOS `say` or Qwen3-TTS 0.6B voice clone (mlx-audio) | local | `audio/speak.py` | `REACHY_VEC_TTS_BACKEND`, `REACHY_VEC_VOICE_SAMPLE` |

All local models lazy-load on first use; `reachy-vec run` warms them (plus
the OpenAI TLS connection) at startup so the first conversation isn't slow.

> **Where the I/O runs:** today every step below reads from and writes to the
> **Mac's own** camera, mic, and speaker — the robot is motion-only
> (`media_backend="no_media"`). On-robot camera/mic/speaker is not wired yet;
> see [architecture.md → Known gaps](architecture.md#known-gaps--toward-a-real-robot-deploy).

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

## 2. Face pipeline (who is this?)

`perception/camera.py` → `perception/face.py` → `store/db.py:match_face`

1. Poll the webcam (`CAMERA_INDEX`).
2. insightface `buffalo_s` detects faces (det size 640×640); the largest
   bounding box wins; produces a normalized 512-dim embedding.
3. k-NN (k=5, cosine) against the `people` table — one row per enrolled
   capture — with a **majority vote** over person_ids; the winner's best
   similarity is the score.
4. Decision (the "never guess" rule):
   - score ≥ `FACE_THRESHOLD` (0.45) → known person
   - within 0.05 below threshold → treated as **no face at all** (probably
     a bad angle of a known person — don't greet, don't offer enrollment)
   - lower → unknown person (enrollment offered after 3 stable polls)

**Enrollment** (`enroll_person`): 5 guided captures ("look left…"), each
embedded and stored as its own `people` row; frames also saved to
`data/faces/{person_id}-{i}.jpg` for audit/re-embedding.

## 2b. Voice identity pipeline (who is talking?)

`audio/listen.py` (utterance audio) → `perception/voice.py` →
`perception/fusion.py`

1. Every utterance's raw audio rides along with its transcript
   (`Utterance(text, audio)`); audio shorter than `VOICE_MIN_UTTERANCE_S`
   (1 s) is "can't tell".
2. ECAPA embeds the audio (192-dim); k-NN majority vote (k=5, cosine)
   against the `voices` table, gated by `VOICE_THRESHOLD` (0.30) with the
   same 0.05 borderline margin as faces.
3. `fuse(face_obs, voice_obs)` decides the turn's identity — voice is the
   authority, face the tie-breaker:
   - voice knows a person → that person (they may be off-camera)
   - voice confidently unknown → anonymous (side-effect tools refuse)
   - voice can't tell → the recognized face, else anonymous
4. Voice profiles come from two sources:
   - **enrolled**: one spoken phrase captured right after face enrollment;
   - **passive**: after a turn where the face is a confident *solo* match
     and the voice doesn't contradict it, the utterance embedding is banked
     (capped at `VOICE_PASSIVE_CAP` = 10 rows per person, oldest evicted).
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

0. The fused per-turn identity (pipeline 2b) arrives as `identity=`;
   retrieval, `save_note`, and `send_message` all follow **the turn's
   speaker**, not whoever started the conversation. Anonymous turns get
   docs-only retrieval and side-effect tools politely refuse.
1. Embed the question with BGE (same space as the docs).
2. Retrieve locally, every turn (no retrieval tool-calls):
   - top-5 `docs` chunks with cosine scores, and
   - top-3 of **the turn speaker's** `memories` rows.
3. Build one user message: `[context + scores] + [memories] + "Name: question"`,
   appended to the visit's history (max 20 messages, reset per visit).
4. One **streaming** `gpt-5-mini` call with the Reachy personality prompt and
   the tool list. Completed sentences are spoken as they arrive (~1 s to
   first sound). The model itself judges relevance: grounded answer naming
   the demo, or general knowledge with a casual "off the top of my head"
   signal (none for chit-chat).
5. If the model called tools: execute them, append results, make a
   **second** LLM call for the final spoken reply (tools cost one extra
   round-trip; sentence audio is suppressed while tool calls are pending).

**Tools** (all in `brain/chat.py`):

| Tool | Effect | Guardrails |
|---|---|---|
| `open_url` | opens a demo in the lab browser | http(s) only |
| `save_note` | stores a memory about the current person | requires a recognized visit; near-duplicates (cosine ≥ 0.97) skipped |
| `send_message` | queues a relay in the `messages` table | recipient must be enrolled (case-insensitive name match) |
| `get_weather` | live conditions via Open-Meteo (no API key) | location from `WEATHER_LAT/LON`; 5 s timeout |

## 5. Memory pipeline (Phase 2)

- **Explicit**: the model calls `save_note` when asked to remember something.
- **Implicit**: when a conversation ends (`end_conversation`), one extra
  `gpt-5-mini` call **per enrolled person who spoke** reviews the visit and
  distills up to 3 third-person notes each (or `NONE` for chit-chat).
- Both paths embed the note with BGE, skip near-duplicates (cosine ≥ 0.97
  vs the person's nearest existing memory), and write `memories` rows.
- Recall is automatic: step 2 of every chat turn searches this person's
  memories with the question vector.

## 6. Messenger pipeline (Phase 3)

1. Sender (recognized) says "tell Bob …" → model calls `send_message` →
   row in `messages` with `delivered_at=""`.
2. When the recipient is next recognized, right after the greeting the loop
   speaks each pending message oldest-first ("By the way, Alice left you a
   message: …") and stamps `delivered_at`.

## 7. The Oracle loop tying it together

`brain/oracle.py:OracleLoop` is a synchronous state machine (all
dependencies injected; the whole thing runs in pytest on fakes):

idle (poll faces) → greet (spoken at most every `GREET_COOLDOWN_S` per
person, silent nod otherwise) → deliver queued messages → listen/think/speak
loop → 30 s silence ends the visit (memories distilled) → back to idle.
No faces for `IDLE_SLEEP_S` (5 min) → sleep motion; wakes on the next face.
OpenAI errors produce a spoken apology and keep listening. Everything heard
and said is logged to `data/reachy.log` (delete to forget).

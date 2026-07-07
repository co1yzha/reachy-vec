# Architecture — how the robot works

Current as of Phase 1.5. The design specs live in
[`docs/superpowers/specs/`](superpowers/specs/); this page is the operational
summary.

## The pieces

```
┌─────────────────────┐   WiFi    ┌──────────────────────────────────────┐
│  Reachy Mini        │◄─────────►│  Brain (Mac, Apple silicon)          │
│  (wireless / sim)   │  SDK      │                                      │
│  motions only for   │           │  eye:   Mac webcam → insightface     │
│  now; camera/mic    │           │  ears:  Mac mic → silero-VAD →       │
│  move on-robot when │           │         faster-whisper (base.en)     │
│  hardware arrives   │           │  mouth: macOS `say` (fish-speech     │
└─────────────────────┘           │         planned)                     │
                                  │  brain: OpenAI (gpt-4o) + RAG        │
       MongoDB Atlas              │  store: LanceDB (data/lancedb)       │
       aixlab.demos ──sync-mongo──►                                      │
       (source of truth)          └──────────────────────────────────────┘
```

- **Package layout:** `store/` (LanceDB + ingestion + mongo sync), `brain/`
  (RAG, oracle state machine), `perception/` (camera, face ID), `audio/`
  (listen/speak), `body/` (motions), `cli/` (one file per command).
- **One LanceDB database, three active tables:** `docs` (text chunks +
  384-dim BGE vectors), `people` (512-dim insightface vectors, one row per
  captured frame), `greetings` (per-person last-greeted timestamps).
  `memories` and `messages` arrive in Phases 2–3.
- **Everything heavy is behind a protocol with a test fake** (`Embedder`,
  `Body`, `Transcriber`, `Speaker`, `FaceMatcher`, `Camera`) — the whole
  state machine runs in pytest without devices, models, or network.

## The run loop (`reachy-vec run`)

```
STARTUP: load .env → KB non-empty? → webcam OK? → warm up STT/faces/embeddings
         (whisper primed with demo titles as vocabulary)
              ▼
  ┌───────── IDLE ◄──────────────────────────────────────────────┐
  │  poll webcam → face embedding → cosine match vs people table │
  │                                                              │
known face (≥ face_threshold)          unknown face (3 stable polls)
  ▼                                             ▼               │
GREET: cooldown expired → spoken       OFFER ENROLL: spoken yes │
"Hi <name>!" + greet motion;           → name → confirm → 5     │
else silent acknowledge                face captures → GREET;   │
  ▼                                    decline → polite exit ───┤
LISTENING: VAD-segmented utterance → whisper                    │
  │        30 s silence → goodbye nod ──────────────────────────┘
  ▼
THINKING (ChatBrain): embed question → scored LanceDB search → context +
  │   "<name>: <question>" appended to the conversation history (reset per
  │   visit) → one LLM call with the Reachy personality prompt. The model
  │   judges relevance: grounded answer naming the demo, or "Not from our
  │   team docs, but..." fallback. Follow-ups work via history.
  │   Tool calls (open_url → default browser on the Mac, http(s) only)
  │   add a second LLM round-trip — actions only, not answers.
  │   OpenAI error → spoken apology, keep listening
  ▼
SPEAKING: `say` + nod → back to LISTENING
```

Identity rule (from the parent spec): **never guess.** Below-threshold =
unknown; borderline (within 0.05 under threshold) = treated as no face at
all, so a bad angle of a known person is neither greeted nor re-enrolled.

## Data flow for knowledge

1. `reachy-vec sync-mongo` — reads `aixlab.demos` (title, project, authors,
   tags, url, note), formats each demo as text, chunks, embeds locally,
   **replaces** all `demo: `-sourced rows (idempotent). Mongo's own
   `embedding` field is ignored; `aixlab.users` is never read.
2. `reachy-vec ingest <path>` — adds `.md`/`.txt` files alongside.
3. Both are queried identically at runtime; sources are spoken/printed
   (`demo: Liverpool City Food Mapping`, `notes.md`).

## Configuration (env vars / `.env`, prefix `REACHY_VEC_`)

| Setting | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `gpt-4o` | OpenAI chat model for answers |
| `STT_MODEL` | `base.en` | faster-whisper size (local backend) |
| `STT_BACKEND` | `local` | `local` or `openai` (gpt-4o-transcribe) |
| `TTS_BACKEND` | `say` | `say` now; `fish-speech`/`openvoice` planned |
| `FACE_THRESHOLD` | `0.45` | cosine gate for recognizing a face |
| `GREET_COOLDOWN_S` | `7200` | spoken greeting at most every 2 h/person |
| `SILENCE_TIMEOUT_S` | `30` | quiet time that ends a conversation |
| `CAMERA_INDEX` | `0` | which webcam |
| `DATA_DIR` | `data` | LanceDB lives at `<data_dir>/lancedb` |

Secrets (no prefix): `OPENAI_API_KEY`, `MONGODB_URI` — both read from `.env`.

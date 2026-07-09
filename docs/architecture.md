# Architecture — how the robot works

Current as of Phase 3. The design specs live in
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
│  hardware arrives   │           │  voice: ECAPA speaker ID + fusion    │
└─────────────────────┘           │  mouth: macOS `say` / Qwen3-TTS      │
                                  │         voice clone (mlx-audio)      │
                                  │  brain: OpenAI (gpt-5-mini) + RAG    │
       MongoDB Atlas              │  store: LanceDB (data/lancedb)       │
       aixlab.demos ──sync-mongo──►                                      │
       (source of truth)          └──────────────────────────────────────┘
```

- **Package layout:** `store/` (LanceDB + ingestion + mongo sync), `brain/`
  (RAG, oracle state machine), `perception/` (camera, face ID, speaker ID,
  fusion), `audio/` (listen/speak), `body/` (motions), `cli/` (one file per
  command).
- **One LanceDB database, six active tables:** `docs` (text chunks +
  384-dim BGE vectors), `people` (512-dim insightface vectors, one row per
  captured frame), `voices` (192-dim ECAPA vectors, enrolled + passively
  banked), `greetings` (per-person last-greeted timestamps),
  `memories` (per-person notes: saved via the save_note tool or distilled
  automatically when a conversation ends, recalled by vector search each
  turn), `messages` (queued relays, delivered by voice on the recipient's
  next recognized sighting).
- **Everything heavy is behind a protocol with a test fake** (`Embedder`,
  `Body`, `Transcriber`, `Speaker`, `FaceMatcher`, `SpeakerIdentifier`,
  `Camera`) — the whole state machine runs in pytest without devices,
  models, or network.
- **Per-turn identity (Phase 2b):** every utterance is voice-identified
  (ECAPA vs the `voices` table) and fused with the face observation —
  voice is the authority, face the tie-breaker, never guess. Notes,
  memories, and messages follow whoever actually spoke; an unknown voice
  is answered but never written to the store.

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
THINKING (ChatBrain): embed question (BGE query prefix) → hybrid LanceDB
  search (vector + BM25, cosine scores) → context +
  │   "<name>: <question>" appended to the conversation history (reset per
  │   visit) → one LLM call with the Reachy personality prompt. The model
  │   judges relevance: grounded answer naming the demo, or a general-
  │   knowledge answer with a casual not-from-our-docs signal (none for
  │   chit-chat). Follow-ups work via history. Replies STREAM: each
  │   sentence is spoken as it generates (~1s to first sentence).
  │   Tools (second LLM round-trip, actions only): open_url (browser,
  │   http(s) only), save_note (remember about this person), send_message
  │   (relay to an enrolled teammate on next sighting), get_weather
  │   (Open-Meteo, no key, lab location from settings).
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

## Configuration

Env vars / `.env` with the `REACHY_VEC_` prefix; secrets (`OPENAI_API_KEY`,
`MONGODB_URI`) have no prefix. Full reference with defaults and tuning
notes: **[configuration.md](configuration.md)**. Per-pipeline detail (which
model runs at each step): **[pipelines.md](pipelines.md)**.

## Known gaps / toward a real-robot deploy

The diagram above is the *intended* shape. What actually ships today runs
the entire perception + speech loop on the **Mac's own devices**, and the
robot is used for motion only. Closing the gap to a robot people can walk up
to means, roughly in priority order (sequenced into shippable sub-phases in the
[Phase 4 bring-up spec](superpowers/specs/2026-07-09-phase4-hardware-bringup-design.md)):

1. **On-robot media — done in Phase 4a.** `reachy-vec run --source robot`
   (or `MEDIA_SOURCE=robot`) now drives the loop from the robot's own camera
   (`RobotCamera`), mic (`RobotAudioSource`), and speaker (`RobotAudioSink`,
   cloned-voice `qwen-tts` only) via the SDK's `mini.media`, all behind the
   existing `Camera` / `AudioSource` / `Speaker` protocols. `auto` falls back
   to the Mac's devices when the daemon offers no media. Still open: on-robot
   `say` output (renders Mac-side for now) and far-field mic gain tuning.
2. **`ROBOT_HOST` is declared but unused.** `make_body` always talks to the
   local daemon; there is no remote-address wiring or reconnection/health
   check during a session (any failure degrades silently to `NullBody`).
3. **Barge-in is specced but not built.** No `Speaker.stop()`, no
   `BargeInMonitor`, no `SpeechInterrupted` — see
   [phase-2c spec](superpowers/specs/2026-07-08-phase2c-voice-bargein-design.md).
   The robot cannot yet be interrupted mid-reply.
4. **Cloud dependency / no offline fallback.** Answers require the OpenAI
   API; a WiFi or API outage yields a spoken apology, not a degraded local
   answer. The Mac↔robot link is also assumed reliable.
5. **Attention is face-only.** No wake word and no sound localization —
   someone off-camera can't get the robot's attention or make it turn toward
   the speaker; multi-party turn-taking beyond voice attribution is untested.
6. **Operational polish.** No autostart/service, no metrics beyond
   `data/reachy.log`, and identity management (delete/rename/merge a person,
   consent + retention controls) is not exposed anywhere but the raw store.

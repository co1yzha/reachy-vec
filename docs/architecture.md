# Architecture — how the robot works

Current as of Phase 4 (wake word, voice-only identity, local LLM, Hermes). The design specs live in
[`docs/superpowers/specs/`](superpowers/specs/); this page is the operational
summary.

## The pieces

```
┌─────────────────────┐   WiFi    ┌──────────────────────────────────────┐
│  Reachy Mini        │◄─────────►│  Brain (Mac, Apple silicon)          │
│  (wireless / sim)   │  SDK      │                                      │
│  motions only for   │           │  eye:   (vision disabled for now)    │
│  now; camera/mic    │           │  ears:  Mac mic → silero-VAD →       │
│  move on-robot when │           │         faster-whisper (base.en)     │
│  hardware arrives   │           │  voice: ECAPA speaker ID (voice-only)│
└─────────────────────┘           │  mouth: macOS `say` / Qwen3-TTS      │
                                  │         voice clone (mlx-audio)      │
                                  │  brain: local oMLX LLM + RAG         │
       MongoDB Atlas              │  store: LanceDB (data/lancedb)       │
       aixlab.demos ──sync-mongo──►                                      │
       (source of truth)          └──────────────────────────────────────┘
```

- **Package layout:** `store/` (LanceDB + ingestion + mongo sync), `brain/`
  (RAG, oracle state machine, Hermes agent), `perception/` (speaker ID,
  identity), `audio/` (listen/speak), `body/` (motions), `cli/` (one file per
  command).
- **One LanceDB database, five tables:** `docs` (text chunks +
  384-dim BGE vectors), `voices` (192-dim ECAPA vectors, enrolled + passively
  banked), `greetings` (per-person last-greeted timestamps),
  `memories` (per-person notes: saved via the save_note tool or distilled
  automatically when a conversation ends, recalled by vector search each
  turn), `messages` (queued relays, delivered by voice on the recipient's
  next time that person speaks).
- **Everything heavy is behind a protocol with a test fake** (`Embedder`,
  `Body`, `Transcriber`, `Speaker`, `SpeakerIdentifier`,
  `Camera`, `Agent`) — the whole state machine runs in pytest without devices,
  models, or network.
- **Per-turn identity:** every utterance is voice-identified (ECAPA vs the
  `voices` table) — voice is the only identity signal; never guess. Notes,
  memories, and messages follow whoever actually spoke; an unknown voice
  is answered but never written to the store.

## The run loop (`reachy-vec run`)

```
STARTUP: load .env → KB non-empty? → warm up STT/voices/embeddings/LLM
         (demo titles snapped into transcripts after STT)
              ▼
  ┌───────── ASLEEP / IDLE ◄─────────────────────────────────────┐
  │  mic always open; listen in short windows for a WAKE_WORDS    │
  │  entry ("Reachy" + common mishearings)                        │
  ▼                                                               │
WAKE: words after "Reachy" are the first question. Voice ID →     │
known voice: greet (cooldown) + deliver queued messages;          │
stranger: offered voice enrollment (name + spoken sentences)      │
  ▼                                                               │
LISTENING: VAD-segmented utterance → whisper                    │
  │        30 s silence → goodbye nod ─────────────────────────────┘
  ▼
THINKING (ChatBrain): embed question (BGE query prefix) → hybrid LanceDB
  search (vector + BM25, cosine scores) → context +
  │   "<name>: <question>" appended to the conversation history (reset per
  │   visit) → one LLM call with the Reachy personality prompt. The model
  │   judges relevance: grounded answer naming the demo, or a general-
  │   knowledge answer with a casual not-from-our-docs signal (none for
  │   chit-chat). Follow-ups work via history. Replies STREAM: each
  │   sentence is spoken as it generates (~1s to first sentence).
  │   LLM: local oMLX (`Qwen3.6-35B-A3B-oQ6e-mtp`, thinking off) at LLM_BASE_URL.
  │   Tools (second LLM round-trip, actions only): open_url (browser,
  │   http(s) only; runs only when the user's turn has an explicit
  │   open/show/... word), save_note (remember about this person),
  │   send_message (relay to an enrolled teammate, delivered at a turn
  │   boundary when they next speak), get_weather (Open-Meteo, no key),
  │   get_time, ask_hermes (multi-step tasks → Hermes CLI, `reachy`
  │   profile, read-only + scheduling toolsets; holding line "Let me look
  │   into that.", waits 20 s, else the answer arrives later as a message
  │   from Hermes; one task at a time).
  │   LLM error → spoken apology, keep listening
  ▼
SPEAKING: `say` + nod → back to LISTENING
```

Identity rule (from the parent spec): **never guess.** Voice below
threshold = stranger; borderline (within 0.05 under threshold) or too-short
audio = can't tell, answered anonymously and never written to the store.

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
`MONGODB_URI`, `LLM_API_KEY`) have no prefix. Full reference with defaults and tuning
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
2. **`ROBOT_HOST` + body resilience — done in Phase 4b.** `make_robot` now
   connects to a remote robot (`connection_mode="network"`) when `ROBOT_HOST`
   is set, and a `ReconnectingBody` keeps motions alive across a transient
   daemon/WiFi drop, degrading to a spoken body-less mode only after
   `BODY_RECONNECT_ATTEMPTS`. Still open: **robot media** does not hot-recover
   — on a stream drop the camera/mic go quiet (soft-degrade) until it returns;
   live media→Mac fallback is a later phase.
3. **Barge-in — done in Phase 4c.** Talking over a reply stops the current
   sentence (`Speaker.stop()`), abandons the in-flight LLM stream (partial
   reply kept in history), and makes the interruption the next turn — via a
   `BargeInMonitor` watching the mic and `SpeechInterrupted` in `ChatBrain`.
   Gated by `BARGE_IN`. Caveat: on-robot audio-out `stop()` is best-effort
   (per-sentence buffer); the `say`/local backends cut mid-sentence.
4. **LLM server dependency.** Answers need the local oMLX server
   (`LLM_BASE_URL`), or OpenAI if that is unset; there is no automatic
   fallback between them, so an outage yields a spoken apology. Hermes
   tasks additionally need the `hermes` CLI and its `reachy` profile.
   The Mac↔robot link is also assumed reliable.
5. **Attention is wake-word only.** "Reachy" wakes it, but there is no
   sound localization — it can't turn toward the speaker; multi-party
   turn-taking beyond voice attribution is untested.
6. **Operational polish.** No autostart/service, no metrics beyond
   `data/reachy.log`, and identity management (delete/rename/merge a person,
   consent + retention controls) is not exposed anywhere but the raw store.
   - `selfie()` writes photos to `data/photos/` — pictures of people;
     git-ignored, never committed (same privacy stance as `data/reachy.log`).

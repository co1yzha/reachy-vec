# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team Familiar: an embodied team assistant on a Reachy Mini robot. The robot wakes when someone says "Reachy", recognizes teammates by voice (face recognition was removed), answers questions from a shared knowledge base (RAG), remembers per-person notes, relays messages, looks through its camera on request, and delegates multi-step / live-information tasks to a Hermes agent. The brain runs on a Mac; the robot (or MuJoCo simulator) is the body for motion and, optionally, camera/mic/speaker. Currently mid-Phase 4 (hardware bring-up): 4a on-robot media, 4b `ROBOT_HOST` + reconnect, and 4c barge-in have shipped, plus the local LLM + Hermes work; 4d (autostart + offline degradation) has not.

**Remaining gaps (see `docs/architecture.md` → Known gaps):** `say` TTS has no on-robot output (renders Mac-side; `qwen-tts` and `kokoro` play on the robot); far-field mic gain is untuned; robot media does not hot-recover from a stream drop (soft-degrades until it returns — no live media→Mac fallback). Answers need the local oMLX server (`Qwen3.6-35B-A3B-4bit` at `http://127.0.0.1:8001/v1`), or OpenAI if `LLM_BASE_URL` is unset; there is no fallback between them.

**Current local setup (this Mac's `.env`; code defaults differ):** STT `parakeet`, TTS `kokoro` with `KOKORO_VOICE=bf_alice`, `HERMES_TRANSPORT=gateway`. `skills/` holds Hermes skills (e.g. `skills/web/agent-reach`) for the Hermes `reachy` profile — not Claude Code skills, and not part of the Python package.

## Commands

```bash
uv sync                              # install (Python 3.12+, uv-managed venv)
uv run pytest -q                     # full test suite (no devices/network needed)
uv run pytest tests/test_oracle.py -q             # one test file
uv run pytest tests/test_oracle.py::test_name -q  # one test
uv run ruff check src tests            # lint (add --fix to auto-fix)

uv run reachy-vec doctor             # check the whole setup (LLM, robot, motors, Hermes...) and say what to fix
#   read-only, safe while `run` is running; --fix switches the robot's motors on
uv run reachy-vec chat               # text-only brain check (uses the configured LLM)
uv run reachy-vec sync-mongo         # pull aixlab.demos into LanceDB (needs MONGODB_URI)
uv run reachy-vec ingest <path>      # add .md/.txt docs
uv run reachy-vec record-voice       # record ~10s mic sample for cloned TTS -> data/voice_sample.wav
uv run reachy-vec run                # full Oracle loop (mic; say "Reachy" to start)
#   --source auto|robot|mac (or REACHY_VEC_MEDIA_SOURCE): whose camera/mic/speaker
#   REACHY_VEC_MIC_SOURCE=mac: Mac/USB mic with the robot's speaker
#   REACHY_VEC_ROBOT_HOST=<ip> connects to a remote robot (network mode)
#   TTS: REACHY_VEC_TTS_BACKEND=say (default, macOS) | kokoro (fast presets, KOKORO_VOICE)
#        | qwen-tts (cloned voice, needs REACHY_VEC_VOICE_SAMPLE=<wav>)
#   STT: REACHY_VEC_STT_BACKEND=local (faster-whisper, default) | parakeet | openai
#   --dashboard (or REACHY_VEC_DASHBOARD=true): also serve the dashboard, open its Live page
uv run reachy-vec dashboard          # local web UI: LanceDB store (/) + live run log (/live; --live opens it)

# optional extras (the lock can't resolve them here; install without re-locking):
uv pip install --python .venv/bin/python parakeet-mlx    # STT_BACKEND=parakeet
uv pip install --python .venv/bin/python "misaki[en]"    # TTS_BACKEND=kokoro

# simulator (separate terminal, needed for `run` body motions):
uv run mjpython .venv/bin/reachy-mini-daemon --sim      # with 3D viewer (macOS)
uv run reachy-mini-daemon --sim --headless              # headless
```

Gotchas: after (re)creating `.venv`, re-link libpython for MuJoCo (see README). A crashed daemon can hold port 8000: `pkill -f reachy-mini-daemon`. Secrets (`OPENAI_API_KEY`, `MONGODB_URI`, `REACHY_VEC_LLM_API_KEY`, `TAVILY_API_KEY`, the Hermes gateway key) live in `.env` (template: `.env.example`; `.env` backups are gitignored too). No CI — run `ruff check` + `pytest` locally before pushing; both must pass.

Robot hardware gotchas (wireless Reachy Mini, WiFi-only — USB-C is power-only): robot connected but not moving while every command "succeeds" = wedged onboard daemon — `curl -X POST http://reachy-mini.local:8000/api/daemon/restart` (its dashboard lives at that host:port). Motors off = robot limp; `run` switches them on at connect/wake, `doctor --fix` does it manually, and the LLM has a `fix_body` tool. Only one media client at a time, and constructing a `no_media` SDK connection *releases* the daemon's media, killing another client's live stream (why `ReconnectingBody` is seeded with the initial body, and why `doctor` uses only the REST API). Robot WebRTC camera frames arrive read-only and the first frame takes ~2 s. Pytest fakes can pass while production wiring is broken — hardware-facing changes need the `docs/testing.md` smoke checklist on the real robot.

## Architecture

Full detail in `docs/architecture.md` and `docs/pipelines.md` (models per step, config knobs), `docs/configuration.md` (every setting). Design specs and phase plans: `docs/superpowers/`. `wiki/` is an Obsidian-style project wiki (`wiki/index.md`).

The core pattern: **every heavy dependency sits behind a small Protocol with a test fake** — `Embedder`, `Transcriber`, `Speaker`, `Body`, `SpeakerIdentifier`, `Camera`. The whole Oracle state machine runs in pytest with no devices, models, or network. When adding a capability, keep this shape: protocol in the module, real implementation lazy-loads its model, fake in `tests/conftest.py`.

Data layer: one embedded LanceDB database (`data/lancedb`) with five tables — `docs` (384-dim BGE chunks), `voices` (192-dim ECAPA embeddings, enrolled + passive), `greetings`, `memories` (per-person notes), `messages` (queued relays). All access goes through `store/db.py:Store`; schemas in `store/schemas.py`.

Control flow: `cli/run.py` wires everything and hands it to `brain/oracle.py:OracleLoop`, a synchronous state machine (listen for wake word → greet → listen → think → speak). The mic is always open; a transcript containing a `WAKE_WORDS` entry ("reachy" + common STT mishearings) starts a conversation, and words after it are the first question. Each utterance is voice-identified (`perception/fusion.py:from_voice`) into a per-turn `TurnIdentity`. Assistant first: enrollment (name read by the LLM, spelling fallback, then a few spoken sentences) happens only when someone asks to be remembered, unless `ENROLL_AUTO_OFFER` is on.

Each turn first goes through `brain/router.py` — one short LLM call that picks a route: `hermes` (multi-step tasks and anything needing live information), `open` (open a site, or play music/videos via a YouTube search — the brain opens the router's URL itself, since the local model claims actions without calling tools), `latest_demos`, `fix_body`, `volume` ("speak up"/"too loud": robot speaker volume via the daemon REST API when replies play on the robot, else macOS output volume; `audio/volume.py`), `remember_me`, or `chat` (the default, and the fallback on any failure). `chat` is `brain/chat.py:ChatBrain`: embed question → scored LanceDB search (docs above `RAG_MIN_SCORE` + the turn speaker's memories) → one streaming LLM call (local oMLX `Qwen3.6-35B-A3B-4bit` via `brain/llm.py:make_llm_client`, thinking off via `LLM_DISABLE_THINKING`; OpenAI when `LLM_BASE_URL` is unset) with tools: `open_url` (only if the turn contains an `OPEN_URL_TRIGGER_WORDS` word), `save_note`, `send_message`, `get_weather`, `get_time`, `list_demos`, `fix_body`, `set_volume`; `look`/`selfie`/`look_around` if `CAMERA_TOOLS`; `web_search` if `WEB_SEARCH` + `TAVILY_API_KEY`; `ask_hermes` if `HERMES_ENABLED`. Hermes (`brain/hermes.py`) runs under its `reachy` profile via the CLI or a running gateway (`HERMES_TRANSPORT=cli|gateway`), waits `HERMES_WAIT_S` (20 s; longer for unrecognised speakers, who get no late delivery) with spoken progress updates, then hands the result off as a queued message from "Hermes".

Speech out: a short filler (`ACK_ENABLED`) covers prompt processing; `audio/pipeline.py:SpeechPipeline` synthesizes the next sentence while the current one plays; emojis in replies become gestures, not speech; the head sways with the audio. Barge-in (`BARGE_IN`, default `BARGE_IN_MODE=keyword`): saying "Reachy"/"stop"/"wait" over a reply stops it and the interruption becomes the next turn; `audio/echo.py:EchoGuard` vetoes transcripts that are just the reply heard back (off-robot mics have no echo cancellation). Conversations end on silence (wordless noise doesn't count); `end_conversation()` distills up to 3 memories per enrolled speaker via extra LLM calls.

Identity rule (from the design spec): **never guess.** Voice match below `VOICE_THRESHOLD` = stranger; borderline (within 0.05 under threshold) or too-short audio = can't tell — answered anonymously, not greeted by name. An empty voice store means everyone is a stranger. An anonymous turn is answered but never written to the store. Passive voice banking only takes matches ≥ threshold + 0.05, since there is no second modality to confirm them.

Configuration is all `pydantic-settings` in `config.py`: env vars with the `REACHY_VEC_` prefix, loaded from `.env`. Add new knobs there, not as module constants.

## Conventions

- Modules map to roles: `store/` persistence, `brain/` reasoning (router, chat, Hermes), `perception/` identity + camera, `audio/` I/O, `body/` robot motions, `cli/` one file per command.
- Heavy imports (torch, sentence-transformers, faster-whisper, mlx-audio, parakeet-mlx) are deferred to inside methods/functions so `import reachy_vec` and the test suite stay fast.
- Everything the robot hears and says is logged to `data/reachy.log` (rolls over at midnight to `reachy.log.YYYY-MM-DD`; old days are kept) — privacy-relevant; never commit `data/`.
- New behavior gets a test against the fakes first; see `docs/testing.md` for the manual smoke-test checklist before claiming hardware-facing work done.
- Cross-session task notes / deferred work live in Claude Code's auto-memory, relocated into the repo at `.claude/memory/` via `autoMemoryDirectory` in `.claude/settings.json` (one fact per file; `MEMORY.md` there is the index loaded each session). Note: `.claude/` is gitignored and auto-memory is machine-local — it is **not** committed or shared with teammates. The committed, team-shareable version of that durable knowledge is `docs/project-notes.md`; design specs and implementation plans live in `docs/superpowers/`.

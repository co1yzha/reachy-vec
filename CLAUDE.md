# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team Familiar: an embodied team assistant on a Reachy Mini robot. The robot wakes when someone says "Reachy", recognizes teammates by voice (face recognition was removed; vision tools are disabled for now), answers questions from a shared knowledge base (RAG), remembers per-person notes, and relays messages. The brain runs on a Mac; the robot (or MuJoCo simulator) is a thin body for motion, and (since Phase 4a) optionally for camera/mic/speaker too. Currently mid-Phase 4 (hardware bring-up): 4a on-robot media, 4b `ROBOT_HOST` + reconnect, and 4c barge-in have shipped; 4d (autostart + offline degradation) has not.

**Remaining gaps (see `docs/architecture.md` → Known gaps):** on-robot `say` TTS output still renders Mac-side (`qwen-tts` is the on-robot backend); far-field mic gain is untuned; robot media does not hot-recover from a stream drop (soft-degrades until it returns — no live media→Mac fallback). Answers need the local oMLX server (`Qwen3.6-35B-A3B-oQ6e-mtp` at `http://127.0.0.1:8001/v1`), or OpenAI if `LLM_BASE_URL` is unset; there is no fallback between them.

## Commands

```bash
uv sync                              # install (Python 3.12+, uv-managed venv)
uv run pytest -q                     # full test suite (no devices/network needed)
uv run pytest tests/test_oracle.py -q             # one test file
uv run pytest tests/test_oracle.py::test_name -q  # one test
uv run ruff check src tests            # lint (add --fix to auto-fix)

uv run reachy-vec chat               # text-only brain check (needs OPENAI_API_KEY)
uv run reachy-vec sync-mongo         # pull aixlab.demos into LanceDB (needs MONGODB_URI)
uv run reachy-vec ingest <path>      # add .md/.txt docs
uv run reachy-vec record-voice       # record ~10s mic sample for cloned TTS -> data/voice_sample.wav
uv run reachy-vec run                # full Oracle loop (mic; say "Reachy" to start)
#   --source auto|robot|mac (or REACHY_VEC_MEDIA_SOURCE): whose camera/mic/speaker
#   REACHY_VEC_ROBOT_HOST=<ip> connects to a remote robot (network mode)
#   cloned voice: REACHY_VEC_TTS_BACKEND=qwen-tts + REACHY_VEC_VOICE_SAMPLE=<wav>
#   (Qwen3-TTS via mlx-audio, local; default backend is macOS `say`)
uv run reachy-vec dashboard          # local web UI to browse the LanceDB store

# simulator (separate terminal, needed for `run` body motions):
uv run mjpython .venv/bin/reachy-mini-daemon --sim      # with 3D viewer (macOS)
uv run reachy-mini-daemon --sim --headless              # headless
```

Gotchas: after (re)creating `.venv`, re-link libpython for MuJoCo (see README). A crashed daemon can hold port 8000: `pkill -f reachy-mini-daemon`. Secrets (`OPENAI_API_KEY`, `MONGODB_URI`, and `REACHY_VEC_LLM_API_KEY`) live in `.env` (template: `.env.example`). No CI — run `ruff check` + `pytest` locally before pushing; both must pass.

Robot hardware gotchas (wireless Reachy Mini, WiFi-only — USB-C is power-only): robot connected but not moving while every command "succeeds" = wedged onboard daemon — `curl -X POST http://reachy-mini.local:8000/api/daemon/restart` (its dashboard lives at that host:port). Only one media client at a time, and constructing a `no_media` SDK connection *releases* the daemon's media, killing another client's live stream (why `ReconnectingBody` is seeded with the initial body). Robot WebRTC camera frames arrive read-only and the first frame takes ~2 s. Pytest fakes can pass while production wiring is broken — hardware-facing changes need the `docs/testing.md` smoke checklist on the real robot.

## Architecture

Full detail in `docs/architecture.md` and `docs/pipelines.md` (models per step, config knobs). Design specs and phase plans: `docs/superpowers/`.

The core pattern: **every heavy dependency sits behind a small Protocol with a test fake** — `Embedder`, `Transcriber`, `Speaker`, `Body`, `SpeakerIdentifier`, `Camera`. The whole Oracle state machine runs in pytest with no devices, models, or network. When adding a capability, keep this shape: protocol in the module, real implementation lazy-loads its model, fake in `tests/conftest.py`.

Data layer: one embedded LanceDB database (`data/lancedb`) with five tables — `docs` (384-dim BGE chunks), `voices` (192-dim ECAPA embeddings, enrolled + passive), `greetings`, `memories` (per-person notes), `messages` (queued relays). All access goes through `store/db.py:Store`; schemas in `store/schemas.py`.

Control flow: `cli/run.py` wires everything and hands it to `brain/oracle.py:OracleLoop`, a synchronous state machine (listen for wake word → greet/enroll → listen → think → speak). The mic is always open; a transcript containing a `WAKE_WORDS` entry ("reachy" + common STT mishearings) starts a conversation, and words after it are the first question. Each utterance is voice-identified (`perception/fusion.py:from_voice`) into a per-turn `TurnIdentity`; strangers are offered voice enrollment (name + a few spoken sentences). `brain/chat.py:ChatBrain` does the per-turn work: embed question → scored LanceDB search (docs + the turn speaker's memories) → one streaming LLM call (local oMLX `Qwen3.6-35B-A3B-oQ6e-mtp` via `brain/llm.py:make_llm_client`, thinking off by default via `LLM_DISABLE_THINKING`; OpenAI when `LLM_BASE_URL` is unset) with tools (`open_url` — only runs if the turn contains an `OPEN_URL_TRIGGER_WORDS` word; `save_note`, `send_message`, `get_weather`, `get_time`; `ask_hermes` if `HERMES_ENABLED` — delegates multi-step tasks to the Hermes CLI under its `reachy` profile with read-only + scheduling toolsets, waits `HERMES_WAIT_S` (20 s), then hands the result off as a queued message from "Hermes"; `web_search` if `TAVILY_API_KEY` is set; camera tools `look`/`selfie`/`look_around` exist but are not wired into `run` while vision is disabled); sentences are spoken as they stream. Barge-in (`BARGE_IN`, on by default): talking over a reply stops the sentence and makes the interruption the next turn. Conversations end on silence; `end_conversation()` distills up to 3 memories per enrolled speaker via extra LLM calls.

Identity rule (from the design spec): **never guess.** Voice match below `VOICE_THRESHOLD` = stranger (offered enrollment); borderline (within 0.05 under threshold) or too-short audio = can't tell — answered anonymously, not greeted by name. An anonymous turn is answered but never written to the store. Passive voice banking only takes matches ≥ threshold + 0.05, since there is no second modality to confirm them.

Configuration is all `pydantic-settings` in `config.py`: env vars with the `REACHY_VEC_` prefix, loaded from `.env`. Add new knobs there, not as module constants.

## Conventions

- Modules map to roles: `store/` persistence, `brain/` reasoning, `perception/` identity, `audio/` I/O, `body/` robot motions, `cli/` one file per command.
- Heavy imports (torch, sentence-transformers, faster-whisper, mlx-audio) are deferred to inside methods/functions so `import reachy_vec` and the test suite stay fast.
- Everything the robot hears and says is logged to `data/reachy.log` — privacy-relevant; never commit `data/`.
- New behavior gets a test against the fakes first; see `docs/testing.md` for the manual smoke-test checklist before claiming hardware-facing work done.
- Cross-session task notes / deferred work live in Claude Code's auto-memory, relocated into the repo at `.claude/memory/` via `autoMemoryDirectory` in `.claude/settings.json` (one fact per file; `MEMORY.md` there is the index loaded each session). Note: `.claude/` is gitignored and auto-memory is machine-local — it is **not** committed or shared with teammates. The committed, team-shareable version of that durable knowledge is `docs/project-notes.md`; design specs and implementation plans live in `docs/superpowers/`. As of 2026-07-23 the queued next task is local small-LLM integration (`local-llm-deferred` / see `docs/project-notes.md`).

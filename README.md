# reachy-vec — Team Familiar

An embodied team assistant on a Reachy Mini (wireless). The robot recognizes
teammates by face and voice, answers questions from a shared knowledge base,
remembers who said what, and relays messages — all vectors (docs, memories,
faces, voices) in one embedded LanceDB store.

**Architecture:** the robot (Raspberry Pi 5) is a thin body streaming
audio/video over WiFi; the brain runs on a Mac (Apple silicon) — face ID
(insightface), speaker ID (ECAPA), STT (faster-whisper), RAG via the OpenAI
API, and voice-cloned TTS (fish-speech) back out through the robot's speaker.

See [the design spec](docs/superpowers/specs/2026-07-06-team-familiar-design.md)
for aims, architecture, and the phased roadmap.

## Roadmap

- **Phase 0 — Skeleton & simulator:** text-only RAG loop, doc ingestion, no hardware.
- **Phase 1 — Oracle:** walk up, get greeted by name, ask a question, hear the answer.
- **Phase 2 — Memory keeper:** per-person attribution of spoken notes, recall by person.
- **Phase 3 — Messenger:** "tell Bob…" relayed when Bob is next seen.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run reachy-vec --help
```

## Layout

```
src/reachy_vec/
  config.py        # settings (robot host, model choices, data dir)
  store/           # persistence: LanceDB connection + table schemas
    schemas.py, db.py
  brain/           # reasoning: intent routing, RAG, conversation loop
    intents.py, rag.py, loop.py
  body/            # robot I/O: connection/streaming + motion primitives
    robot.py, motions.py
  perception/      # identity: face ID + speaker ID + fusion
    face.py, voice.py, fusion.py
  audio/           # audio front-end and output
    listen.py, speak.py
  cli/             # entry points, one file per command
    chat.py, ingest.py, enroll.py, run.py
tests/
```

# reachy-vec — Team Familiar

An embodied team assistant on a Reachy Mini (wireless). The robot recognizes
teammates by face and voice, answers questions from a shared knowledge base,
remembers who said what, and relays messages — all vectors (docs, memories,
faces, voices) in one embedded LanceDB store.

**Architecture:** the robot (Raspberry Pi 5) is a thin body streaming
audio/video over WiFi; the brain runs on a Mac (Apple silicon) — face ID
(insightface), speaker ID (ECAPA), STT (faster-whisper), RAG via the OpenAI
API, and voice-cloned TTS (Qwen3-TTS via mlx-audio) back out through the
robot's speaker.

Docs:
- **[Architecture](docs/architecture.md)** — how the robot works: the run
  loop and data flow
- **[Pipelines](docs/pipelines.md)** — each pipeline step by step, and
  which model runs at every step
- **[Configuration](docs/configuration.md)** — every knob, default, and
  tuning note
- **[Testing guide](docs/testing.md)** — automated suite + manual smoke
  tests + troubleshooting
- **[Design spec](docs/superpowers/specs/2026-07-06-team-familiar-design.md)** —
  aims, architecture decisions, and the phased roadmap ([all specs & plans](docs/superpowers/))

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

### macOS one-time fix (simulator + media)

uv-managed virtualenvs don't ship `libpython3.12.dylib` where MuJoCo's
`mjpython` and GStreamer expect it. After (re)creating `.venv`, run:

```bash
ln -sf ~/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/lib/libpython3.12.dylib .venv/lib/
```

### Running the simulator (no robot needed)

```bash
# with the MuJoCo 3D viewer window (macOS requires mjpython for the GUI):
uv run mjpython .venv/bin/reachy-mini-daemon --sim

# headless (tests/CI):
uv run reachy-mini-daemon --sim --headless
```

Dashboard: http://localhost:8000. If the daemon reports a weird state after
a crashed run, a stale process may hold port 8000: `pkill -f reachy-mini-daemon`.

### Cloned voice (optional)

By default the robot uses the macOS `say` voice. To have it speak in a cloned
voice (fully local, Qwen3-TTS on MLX):

1. Record ~10 seconds of clean speech, e.g.
   `sox -d -r 24000 -c 1 data/voice_sample.wav trim 0 10`
   (or QuickTime → export WAV). Only clone voices with the speaker's consent.
2. In `.env`, set `REACHY_VEC_TTS_BACKEND=qwen-tts` and
   `REACHY_VEC_VOICE_SAMPLE=data/voice_sample.wav`. Optionally set
   `REACHY_VEC_VOICE_SAMPLE_TEXT` to the sample's transcript to skip a
   one-off auto-transcription.
3. `uv run reachy-vec run --preview` — the first run downloads the model
   (~1.5 GB); expect 1–3 s of synthesis per sentence.

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

# reachy-vec — Team Familiar

An embodied team assistant on a Reachy Mini (wireless). Say "Reachy" to start a
conversation; the robot recognizes teammates by voice, answers questions from a shared knowledge base,
remembers who said what, and relays messages — all vectors (docs, memories,
voices) in one embedded LanceDB store.

**Architecture:** the robot (Raspberry Pi 5) is a thin body streaming
audio/video over WiFi; the brain runs on a Mac (Apple silicon) — speaker ID (ECAPA), STT (Parakeet on Apple GPU), RAG via the local oMLX server
(`Qwen3.6-35B-A3B-4bit`), and voice via Kokoro preset voices (bf_alice) through the
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
- **[Voice cloning](docs/voice-cloning.md)** — make the robot speak in your
  own voice (qwen-tts backend)
- **[Design spec](docs/superpowers/specs/2026-07-06-team-familiar-design.md)** —
  aims, architecture decisions, and the phased roadmap ([all specs & plans](docs/superpowers/))

## Roadmap

- **Phase 0 — Skeleton & simulator:** text-only RAG loop, doc ingestion, no hardware.
- **Phase 1 — Oracle:** walk up, get greeted by name, ask a question, hear the answer.
- **Phase 2 — Memory keeper:** per-person attribution of spoken notes, recall by person.
- **Phase 3 — Messenger:** "tell Bob…" relayed when Bob is next seen.
- **Phase 4 — On-robot deployment** *(in progress)*: 4a on-robot media (`--source robot`), 4b `ROBOT_HOST` + reconnect, and 4c barge-in have shipped; 4d autostart + offline degradation remains. See the [bring-up spec](docs/superpowers/specs/2026-07-09-phase4-hardware-bringup-design.md).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run reachy-vec --help
```

### macOS one-time fix (simulator + media)

uv-managed virtualenvs don't ship `libpython3.12.dylib` where MuJoCo's
`mjpython` and GStreamer expect it. After (re)creating `.venv`, run:

```bash
ln -sf "$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.dylib" .venv/lib/
```

This resolves the library from the venv's own interpreter, so it works whatever
3.12.x patch release uv installed.

### Running the simulator (no robot needed)

```bash
# with the MuJoCo 3D viewer window (macOS requires mjpython for the GUI):
uv run mjpython .venv/bin/reachy-mini-daemon --sim

# headless (tests/CI):
uv run reachy-mini-daemon --sim --headless
```

Daemon dashboard: http://localhost:8000 (robot state — not to be confused
with the LanceDB store dashboard below). If the daemon reports a weird state
after a crashed run, a stale process may hold port 8000:
`pkill -f reachy-mini-daemon`.

### Running on the real robot (wireless)

The wireless Reachy Mini is controlled entirely over WiFi — its USB-C port
is **power-only**. One-time setup: power it on, join the `reachy-mini-ap`
hotspot it broadcasts, and use the Reachy Mini Control app ("First time
connecting…") to hand it your WiFi credentials. It then appears on your
network as `reachy-mini.local` (mDNS), with the daemon dashboard at
http://reachy-mini.local:8000. Enterprise WiFi (eduroam etc.) usually
rejects embedded devices — use a phone/Mac hotspot or a personal router,
with the Mac on the same network.

```bash
# .env
REACHY_VEC_ROBOT_HOST=reachy-mini.local   # or the robot's IP

# robot moves, Mac supplies camera/mic/speaker (desk workflow):
uv run reachy-vec run --source mac

# everything on the robot: its camera, mics, and speaker
# (set REACHY_VEC_TTS_BACKEND=qwen-tts so replies play on the robot):
uv run reachy-vec run --source robot
```

Troubleshooting:

- **Robot connected but not moving** (commands succeed, head stays still):
  the onboard daemon can wedge at boot —
  `curl -X POST http://reachy-mini.local:8000/api/daemon/restart`, or
  power-cycle the robot.
- **One media client at a time.** Don't run two `reachy-vec run` instances
  against the robot; the second steals or kills the first's stream.
- **Keep SDK and daemon versions matched** (dashboard → System Updates);
  mismatches cause subtle media bring-up failures.

### Cloned voice (optional)

By default the robot uses the macOS `say` voice. To have it speak in a cloned
voice (fully local, Qwen3-TTS on MLX):

1. Record a reference sample: `uv run reachy-vec record-voice` prompts you
   to read a sentence, captures ~10 s, and prints the exact `.env` lines to
   paste. (Manual alternative: `sox -d -r 24000 -c 1 data/voice_sample.wav
   trim 0 10` or QuickTime → export WAV.) Only clone voices with the
   speaker's consent.
2. Paste the printed lines into `.env` — `REACHY_VEC_TTS_BACKEND=qwen-tts`,
   `REACHY_VEC_VOICE_SAMPLE=data/voice_sample.wav`, and
   `REACHY_VEC_VOICE_SAMPLE_TEXT` (the sentence you read, so no
   auto-transcription is needed).
3. `uv run reachy-vec run` — the first run downloads the model
   (~1.5 GB); expect 1–3 s of synthesis per sentence.

### Kokoro preset voices (alternative)

Kokoro-82M provides preset voices (no cloning needed) at ~25× real-time speed.
Install with `uv pip install --python .venv/bin/python "misaki[en]"`, then set:

```bash
REACHY_VEC_TTS_BACKEND=kokoro
REACHY_VEC_KOKORO_VOICE=bf_alice   # British female; try bf_emma, am_michael, etc.
```

Voice prefixes: `a…` = US, `b…` = UK. See `docs/configuration.md` for the full list.

## Usage

```bash
uv run reachy-vec chat               # text-only brain check (needs OPENAI_API_KEY)
uv run reachy-vec sync-mongo         # pull aixlab.demos into LanceDB (needs MONGODB_URI)
uv run reachy-vec ingest <path>      # add .md/.txt docs to the knowledge base
uv run reachy-vec record-voice       # record a voice sample for the cloned TTS
uv run reachy-vec run                # full Oracle loop: say "Reachy" to start (mic + robot/sim)
#   --source auto|robot|mac: whose camera/mic/speaker (see "Running on the real robot")
#   --dashboard: also open the Live view (or REACHY_VEC_DASHBOARD=true in .env)
uv run reachy-vec dashboard          # browse the LanceDB store in the browser
#   --live: open the Live conversation page instead
```

### Dashboard (store + live conversation)

`uv run reachy-vec dashboard` serves a local web UI at http://127.0.0.1:8400
and opens it in the browser (`--port` / `--no-open` to override). It has two
pages:

- **Store** (`/`) — live contents of every LanceDB table: docs, people,
  voices, greetings, memories, and messages. Handy for checking what the robot
  ingested, who is enrolled, and which memories/messages it has stored.
- **Live** (`/live`) — follows the current `reachy-vec run` (it starts at
  that run's "run started" line, not the whole history) by tailing
  `data/reachy.log`: what it heard, the route it chose, its reply, opened
  pages, Hermes tasks, barge-ins, and errors, with each step's latency since
  the words were heard. Conversation ends and sleep appear as full-width
  banners ("Conversation ended", a dark "Asleep" bar). A status bar shows listening / thinking / asleep, turns
  red when a turn gets no reply for 30 s (a stuck turn), and shows **which mic
  is listening** (e.g. `Mac input 'Wireless Microphone RX'` or `robot mic`).
  It only reads the log, so it can't slow down or break the robot, and it can
  be started before or after `run`.
- **Ports** (a panel on the Live page) — the services Reachy depends on, each
  with its host:port and whether it answers: robot daemon (8000), LLM / oMLX
  (8001), Hermes gateway (8642), and the dashboard itself. "Everything
  listening on this Mac" lists every listening TCP port with its process.
  Each local process has a **Kill** button — handy for a crashed daemon or
  server still holding its port. It asks first, sends a normal stop
  (SIGTERM, not `kill -9`), and only works on a process `lsof` shows
  listening on that port right now. It can't stop the robot's own daemon (use
  its restart endpoint) or the dashboard itself, and requests from other
  websites are rejected.

Ways to get to the Live page:

```bash
uv run reachy-vec run --dashboard    # run + Live view in one command
uv run reachy-vec dashboard --live   # separate terminal, opens /live directly
```

To always start it with `run`, set `REACHY_VEC_DASHBOARD=true` in `.env`
(`REACHY_VEC_DASHBOARD_PORT` changes the port, default 8400). If the port is
already taken (say, a dashboard is open in another terminal), `run` simply
carries on without starting a second one.

Localhost-only by default: the store holds transcripts and embeddings of
enrolled teammates, and the log holds everything said near the robot — keep it
that way.

## Layout

```
src/reachy_vec/
  config.py        # settings (robot host, model choices, data dir)
  store/           # persistence: LanceDB connection + table schemas
    schemas.py, db.py, embeddings.py, ingestion.py, mongo_sync.py
  brain/           # reasoning: Oracle state machine, RAG chat, intents
    oracle.py, chat.py, intents.py, loop.py
  body/            # robot I/O: connection/streaming + motion primitives
    robot.py, motions.py
  perception/      # identity: speaker ID (voice only)
    voice.py, fusion.py; camera.py, vision.py, occlusion.py (disabled for now)
  audio/           # audio front-end and output (say / qwen-tts voice clone)
    listen.py, speak.py
  cli/             # entry points, one file per command
    chat.py, ingest.py, run.py, sync.py, dashboard.py (store + Live pages), record_voice.py
tests/
```

# Testing guide

## Automated tests

```bash
uv run pytest -q
```

Everything runs on fakes — no robot, devices, models, or network. Suite
covers the store, ingestion, mongo sync, RAG gating, chat loop, motions,
STT segmentation, transcriber selection, face enrollment, and all Oracle
state-machine scenarios.

## Manual smoke tests (in order of dependency)

### 0. Refresh the knowledge base

```bash
uv run reachy-vec sync-mongo          # pull latest aixlab.demos
uv run reachy-vec ingest <path>       # optional extra .md/.txt docs
```

### 1. Text-only brain check (no devices)

```bash
uv run reachy-vec chat
```

- "what demos do we have about food insecurity?" → grounded answer citing
  `demo: Liverpool City Food Mapping` etc.
- "what is the capital of France?" → answers with a light signal that it's
  outside the team library (e.g. "off the top of my head...").

### 2. Voice echo (mic + speaker)

```bash
uv run python -c "
from reachy_vec.audio.speak import make_speaker
from reachy_vec.audio.listen import MicTranscriber
s = make_speaker(); s.speak('Say something now.')
t = MicTranscriber(); text = t.listen_once(10)
s.speak(f'I heard: {text}' if text else 'I heard nothing.'); print(text)
"
```

First run triggers the macOS microphone permission prompt — allow it.

### 3. Face enrollment (webcam)

```bash
uv run reachy-vec enroll "YourName"   # five guided captures
```

First run triggers the camera permission prompt.

### 4. The full Oracle (two terminals)

```bash
# terminal 1 — simulated robot with 3D viewer:
uv run mjpython .venv/bin/reachy-mini-daemon --sim

# terminal 2 — the brain:
uv run reachy-vec run --preview   # --preview opens a "Reachy sees" window
```

The preview window shows the webcam feed with a box around the detected
face: green = recognized (name + score), orange = unknown, gray =
borderline. It only refreshes while the robot is scanning for faces —
it freezes during listening/speaking; that's normal.

Walk through the checklist:

| You do | Expected |
|---|---|
| Walk into webcam frame | Spoken "Hi <name>!" + greet motion in the viewer |
| Ask about a demo | Spoken grounded answer, nod |
| Ask something off-library | Helpful answer with a casual not-from-our-docs signal |
| Stay quiet 30 s | Goodbye nod, back to idle |
| Return immediately | Silent head-turn only (greeting cooldown) |
| Un-enrolled person, ~3 s in frame | Enrollment offer → yes → name → confirm → captures |
| Say "remember that I ..." | "Noted" — stored; verify it survives to the next visit |
| Leave, return later, ask "what was I working on?" | Recalls memories from earlier visits |
| Nobody in frame for 5 min | Robot slumps to sleep; wakes when you appear |
| Say "tell <enrolled teammate> ..." | Queued; spoken to them when they next appear |
| Ask "what's the weather like?" | Live conditions for the lab location (Open-Meteo) |

## Troubleshooting

- **What did it hear/say?** `reachy-vec run` writes every transcribed
  utterance, reply, and opened URL to `data/reachy.log` (also echoed to the
  console). Note this means transcripts of everyone who talks to the robot
  persist there — delete the file to forget.

- **Sim viewer won't open / `launch_passive` error:** the GUI needs
  `mjpython` (see README macOS setup; re-create the libpython symlink after
  rebuilding `.venv`).
- **Daemon in a weird state:** a crashed run may hold port 8000 —
  `pkill -f reachy-mini-daemon` and relaunch.
- **Slow or wrong transcription:** try `REACHY_VEC_STT_BACKEND=openai`
  (accuracy) or `REACHY_VEC_STT_MODEL=small.en` (accuracy, slower);
  `base.en` is the speed default.
- **Not recognized / greeted as unknown:** lower `REACHY_VEC_FACE_THRESHOLD`
  slightly (e.g. 0.40) or re-enroll in better lighting.
- **Fallback fires on questions the docs do cover:** lower
  `REACHY_VEC_RAG_MIN_SCORE` (e.g. 0.45).
- **Inspect what's in LanceDB:**
  ```bash
  uv run python -c "
  import lancedb
  db = lancedb.connect('data/lancedb')
  print('tables:', db.list_tables().tables)
  t = db.open_table('docs')
  print('docs rows:', t.count_rows())
  for r in t.to_arrow().to_pylist()[:5]:
      print('-', r['source'])
  "
  ```

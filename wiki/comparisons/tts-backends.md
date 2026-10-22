---
title: TTS backends (say vs Qwen3-TTS clone)
created: 2026-10-19
updated: 2026-10-19
type: comparison
tags: [tts]
sources: [raw/docs/pipelines.md, raw/docs/readme.md]
---

Two TTS backends, selected by `REACHY_VEC_TTS_BACKEND`:

| | `say` (default) | `qwen-tts` (clone) |
|---|---|---|
| Model | macOS `say` | [[qwen3-tts]] 0.6B via mlx-audio |
| Voice | system voice | a **cloned** teammate's voice |
| Where | local (macOS) | local (MLX) |
| Cost | none | ~1.5 GB model download; 1–3 s/sentence |
| Post-processing | none | loudness/clarity (RMS boost, presence shelf) |

- **On-robot audio out (`--source robot`) requires `qwen-tts`** — the robot's
  `RobotAudioSink` uses the cloned-voice backend.
- Clone only with the speaker's consent. See [[barge-in]] for interruption
  behaviour differences (on-robot stop is best-effort; `say` cuts mid-sentence).


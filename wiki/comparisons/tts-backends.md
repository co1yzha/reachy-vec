---
title: TTS backends (say vs Qwen3-TTS clone vs Kokoro)
created: 2026-10-19
updated: 2026-10-22
type: comparison
tags: [tts]
sources: [raw/docs/pipelines.md, raw/docs/readme.md]
---

Three TTS backends, selected by `REACHY_VEC_TTS_BACKEND`:

| | `say` (default) | `qwen-tts` (clone) | `kokoro` |
|---|---|---|---|
| Model | macOS `say` | [[qwen3-tts]] 0.6B via mlx-audio | Kokoro-82M via mlx-audio |
| Voice | system voice | a **cloned** teammate's voice | preset voices (no cloning) |
| Where | local (macOS) | local (MLX) | local (MLX) |
| Cost | none | ~1.5 GB model download; 1–3 s/sentence | ~25x faster than real time; needs `misaki[en]` |
| Post-processing | none | loudness/clarity (RMS boost, presence shelf) | real tempo change (pitch kept); `TTS_SPEED` |
| Config | default | `REACHY_VEC_TTS_BACKEND=qwen-tts`, `REACHY_VEC_VOICE_SAMPLE` | `REACHY_VEC_TTS_BACKEND=kokoro`, `REACHY_VEC_KOKORO_VOICE` |

**On-robot audio out (`--source robot`)** works with `qwen-tts` and `kokoro` — the robot's `RobotAudioSink` uses these backends. The `say` backend still plays on the Mac.

**Kokoro voices** (prefix sets accent): `a…` = US, `b…` = UK. Examples: `af_heart` (warm US female), `bf_alice` (crisp British female), `bf_emma` (warm British female), `am_michael` (deep US male), `bm_george` (British male). Reachy patches an mlx-audio 0.4.4 length bug that crashed ~1 in 6 sentences.

**Clone only with the speaker's consent.** See [[barge-in]] for interruption behaviour differences (on-robot stop is best-effort; `say` cuts mid-sentence).

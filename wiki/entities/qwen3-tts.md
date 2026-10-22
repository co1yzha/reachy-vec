---
title: Qwen3-TTS (voice-clone TTS)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [tts]
sources: [raw/docs/pipelines.md, raw/docs/readme.md]
---

Qwen3-TTS 0.6B (via mlx-audio, fully local) is the **voice-clone** TTS
backend — see [[tts-backends]]; the default backend is macOS `say`.

- Configured via `REACHY_VEC_TTS_BACKEND=qwen-tts` + `REACHY_VEC_VOICE_SAMPLE`
  (~10 s reference WAV). Only clone voices with the speaker's consent.
- First run downloads the model (~1.5 GB); expect 1–3 s synthesis per sentence.
- **Clone clarity/loudness** (`audio/speak.py:_process_loudness`): the clone's
  level/brightness track the usually-quiet reference sample, so each sentence is
  post-processed before playback — a presence high-shelf
  (`TTS_PRESENCE_DB` dB above `TTS_PRESENCE_HZ`), then a boost toward
  `TTS_TARGET_RMS` with a tanh soft limiter to avoid clipping. Any set to 0
  disables that step.
- **On-robot audio out (`--source robot`) requires the `qwen-tts` backend** — see [[media-source]].


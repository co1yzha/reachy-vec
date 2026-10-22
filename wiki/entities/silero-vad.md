---
title: silero-vad (voice activity)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [stt]
sources: [raw/docs/pipelines.md]
---

silero-vad (16 kHz, 512-sample frames, local) in `audio/listen.py:_AudioCapture`
splits the mic stream into utterances.

- Records 32 ms frames at 16 kHz from the default mic; scores each frame;
  speech = probability > 0.5.
- `collect_utterance` accumulates from first speech until **0.8 s of quiet**
  (or `SILENCE_TIMEOUT_S` with no speech at all → the conversation ends).
- Not configurable (no config knob). Feeds [[faster-whisper]] for transcription. The mic source itself is selectable — see [[media-source]].


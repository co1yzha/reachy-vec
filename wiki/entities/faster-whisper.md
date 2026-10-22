---
title: faster-whisper (STT)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [stt]
sources: [raw/docs/pipelines.md]
---

faster-whisper `base.en` (int8, English-only, local) is the **default** STT
backend, wrapped in `audio/listen.py:MicTranscriber`.

- Configured via `REACHY_VEC_STT_MODEL`; backend via `REACHY_VEC_STT_BACKEND`.
- Utterances arrive already segmented by [[silero-vad]] and are primed with the
  demo titles as an initial vocabulary so demo names transcribe correctly.
- Alternative: OpenAI `gpt-4o-transcribe` (more accurate, ~1 s slower) — see
  [[stt-backends]].


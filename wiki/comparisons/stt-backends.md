---
title: STT backends (local vs OpenAI)
created: 2026-10-19
updated: 2026-10-19
type: comparison
tags: [stt]
sources: [raw/docs/pipelines.md]
---

Two STT backends, selected by `REACHY_VEC_STT_BACKEND`:

| | local (default) | openai |
|---|---|---|
| Model | [[faster-whisper]] `base.en`, int8 | `gpt-4o-transcribe` |
| Where | local | cloud (OpenAI API) |
| Accuracy | good, English-only | more accurate |
| Latency | faster | ~1 s slower |
| Config | `REACHY_VEC_STT_MODEL` | `REACHY_VEC_STT_BACKEND=openai` |

Both receive VAD-segmented utterances from [[silero-vad]] and are primed with
the demo titles as an initial vocabulary prompt. **Verdict:** default to local
for speed/privacy; switch to OpenAI for accuracy-critical or non-English use.


---
title: STT backends (local vs OpenAI vs Parakeet)
created: 2026-10-19
updated: 2026-10-22
type: comparison
tags: [stt]
sources: [raw/docs/pipelines.md]
---

Three STT backends, selected by `REACHY_VEC_STT_BACKEND`:

| | local (default) | openai | parakeet |
|---|---|---|---|
| Model | [[faster-whisper]] `base.en`, int8 | `gpt-4o-transcribe` | NVIDIA Parakeet TDT `mlx-community/parakeet-tdt-0.6b-v2` |
| Where | local | cloud (OpenAI API) | local (Apple GPU via parakeet-mlx) |
| Accuracy | good, English-only | more accurate | good, English |
| Latency | fastest | ~1 s slower | fast (on-GPU) |
| Install | bundled | none extra | `uv pip install --python .venv/bin/python parakeet-mlx` |
| Config | `REACHY_VEC_STT_MODEL` | `REACHY_VEC_STT_BACKEND=openai` | `REACHY_VEC_STT_BACKEND=parakeet` |

Both local backends receive VAD-segmented utterances from [[silero-vad]] and are primed with the demo titles as an initial vocabulary prompt. **Verdict:** default to local (faster-whisper) for speed/privacy; switch to OpenAI for accuracy-critical or non-English use; use Parakeet when on-GPU speed is preferred over the bundled faster-whisper.

Eval scripts: `scripts/stt_eval.py`.

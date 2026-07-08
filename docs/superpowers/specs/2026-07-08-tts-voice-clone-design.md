# TTS voice cloning — Qwen3-TTS backend

**Date:** 2026-07-08 · **Status:** approved in conversation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (supersedes
its TTS decision: fish-speech was chosen before Qwen3-TTS existed).

## Goal

The robot speaks in a cloned voice — the user's own — instead of the macOS
`say` voice. Synthesis stays fully local (privacy stance unchanged: nothing
the robot says leaves the Mac). Per-sentence latency of 1–3 s is acceptable;
the Oracle's sentence-streaming pattern hides most of it behind playback of
the previous sentence.

## Decisions

1. **Backend = Qwen3-TTS 0.6B via mlx-audio.** MLX is native to Apple
   silicon (faster and cooler than PyTorch MPS), weights are Apache 2.0,
   and zero-shot cloning needs only a few seconds of reference audio.
   Supersedes the parent spec's fish-speech pick (PyTorch-MPS latency risk,
   CC-BY-NC-SA weights) and CosyVoice (no MLX path, weaker clone
   benchmarks).
2. **Reference sample is a file, recorded manually.** `settings.voice_sample`
   (already in config) points to a ~10 s clean WAV of the target voice.
   No `record-voice` CLI command for now — QuickTime or `sox` does the job
   (documented in README). Consent stays explicit: you record your own clip.
3. **`say` remains the default backend.** Tests, CI, and dev flows are
   untouched; opting in is two `.env` lines.
4. **Fail fast on misconfiguration.** Selecting `qwen-tts` without a
   readable `voice_sample` raises at `make_speaker()` time with a clear
   message — not mid-conversation.

## Architecture

```
ChatBrain sentence stream ──► Speaker.speak(text)          (unchanged)
                                   │
                     ┌─────────────┴──────────────┐
                 SaySpeaker                 QwenTTSSpeaker (new)
              (macOS say, dev)                    │
                                    lazy: import mlx_audio, load
                                    Qwen3-TTS 0.6B + clone prompt
                                    from settings.voice_sample
                                                  │
                                    synthesize → play WAV, block
```

- `QwenTTSSpeaker` implements the existing `Speaker` protocol
  (`speak(text) -> None`, blocking). No changes to `OracleLoop` or
  `ChatBrain`.
- Model + voice-clone conditioning load lazily on first `speak()` and are
  cached for the process lifetime (repo convention: heavy imports deferred).
- Playback uses `sounddevice` (already a dependency for the mic path),
  blocking until the sentence finishes — same contract as `SaySpeaker`.
- `make_speaker()` gains the `"qwen-tts"` branch; the stale
  `fish-speech`/`openvoice` names in `config.py` comments are updated.

## Configuration

| Setting (env var) | Value |
|---|---|
| `REACHY_VEC_TTS_BACKEND` | `say` (default) \| `qwen-tts` |
| `REACHY_VEC_VOICE_SAMPLE` | path to reference WAV (required for `qwen-tts`) |
| `REACHY_VEC_TTS_MODEL` | mlx-audio model id, default Qwen3-TTS 0.6B (new knob, pinned default) |
| `REACHY_VEC_VOICE_SAMPLE_TEXT` | transcript of the sample (optional; omitted → mlx-audio auto-transcribes once via Whisper) |

`.env.example` gains the two opt-in lines with a comment on recording the
sample.

## Error handling

- Missing/unreadable `voice_sample` with `qwen-tts` selected → `ValueError`
  at wiring time.
- Synthesis failure mid-conversation → log the exception, skip the sentence,
  keep the loop alive (the robot going silent for one sentence beats a
  crash).
- Empty/whitespace text → no-op (same as `SaySpeaker`).

## Testing

- Unit tests drive `QwenTTSSpeaker` with injected fake synthesize/play
  callables (constructor injection, mirroring `SaySpeaker(run=...)`): text
  passed through, empty text skipped, synthesis error swallowed and logged,
  `make_speaker()` branch + misconfiguration error.
- No model download or audio device in the test suite.
- Manual smoke test (per `docs/testing.md`): record sample, set the two env
  vars, `uv run reachy-vec run --preview`, confirm cloned voice and note
  real per-sentence latency; fall back to 1.7B model or shorter responses
  only if quality/latency disappoints.

## Out of scope

- Voice design (text-described synthetic voices), multi-voice per person,
  emotion control.
- A `record-voice` CLI command.
- Cloud TTS of any kind.

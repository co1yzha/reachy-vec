# `record-voice` CLI — capture a clone reference sample

**Date:** 2026-07-09 · **Status:** approved in conversation
**Parent:** [TTS voice clone](2026-07-08-tts-voice-clone-design.md) (delivers
the reference-sample recording deferred there as YAGNI).

## Goal

A one-command way to record a clean ~10 s voice sample for the `qwen-tts`
clone backend, so the robot can speak in the user's own voice without
fiddling with `sox`/QuickTime. Repeatable and foolproof; produces the WAV
plus the exact `.env` lines to enable cloning.

## Decisions

1. **Fixed 10 s recording** with a countdown — no press-to-stop. Simplest,
   and 10 s is ample for zero-shot cloning.
2. **Read a known sentence.** The command prints a fixed, phonetically
   varied line; because we know what was read, we can emit
   `REACHY_VEC_VOICE_SAMPLE_TEXT` directly — no Whisper pass needed.
3. **App never writes `.env`.** Config is env-driven (repo convention), so
   the command echoes the three lines to paste, it doesn't edit `.env`.
4. **Record at 24 kHz mono.** Matches the README; mlx-audio resamples
   anyway.
5. **New `audio/record.py`.** `listen.py` is VAD-segmented utterances +
   transcription; a fixed-duration raw recorder is a separate concern.

## Architecture

```
uv run reachy-vec record-voice [--out data/voice_sample.wav]
        │
        ├─ print sentence to read + 3-2-1 countdown
        ├─ record_sample(10.0, 24000)  ──► np.ndarray   (audio/record.py)
        ├─ write_wav(out, audio, 24000)                 (audio/record.py, pure)
        └─ echo .env lines (TTS_BACKEND, VOICE_SAMPLE, VOICE_SAMPLE_TEXT)
```

- `record_sample(duration_s, sample_rate, record=None)` — records a fixed
  block from the mic via sounddevice; `record` is injectable for tests.
- `write_wav(path, audio, sample_rate)` — pure; writes mono 16-bit PCM WAV.
- `cli/record_voice.py:record_voice(out: Path = data/voice_sample.wav)`
  orchestrates; registered in `cli/__init__.py`.

## Error handling

- Empty/failed capture (all-zero or None) → message + non-zero exit.
- `--out` parent dir created if missing.

## Testing

- `write_wav` round-trip: read back with `wave`, assert 1 channel,
  24000 Hz, 16-bit, sample count ≈ duration.
- CLI test (`CliRunner`) with an injected fake recorder returning a numpy
  array: asserts the WAV is written and the three `.env` lines (with the
  sentence) appear in output. No mic, no model.

## Out of scope

- Press-to-stop / variable length, multiple takes, playback/preview,
  storing the sample in LanceDB, writing `.env` automatically.

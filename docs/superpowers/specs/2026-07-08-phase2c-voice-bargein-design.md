# Phase 2c — Cloned voice (Qwen3-TTS) + barge-in + debug configs

**Date:** 2026-07-08 · **Status:** approved in conversation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (delivers
the "TTS voice cloning" line; fish-speech is superseded by Qwen3-TTS via
mlx-audio as the local backend — same protocol slot, better fit for Apple
silicon).

## Goal

1. Reachy speaks in a cloned voice (yours, from a short sample) with local,
   Apple-silicon-native synthesis.
2. You can interrupt Reachy mid-reply by talking over it ("barge-in"); it
   stops immediately and listens.
3. F5 in VS Code runs/debugs the demo without manual terminal choreography.

## Decisions

1. **TTS backend: Qwen3-TTS-0.6B (quantized) via `mlx-audio`** — a Python
   library, local, fast on M-series, clones from a ~5 s reference sample.
   `say` stays the default and the fallback; select with
   `REACHY_VEC_TTS_BACKEND=qwen3`. Model id is a knob so Voxtral TTS (also
   in mlx-audio) can be A/B-tested with the same voice sample.
   Ubuntu note: MLX is Apple-only; on a future Linux box the backend swaps
   behind the same `Speaker` protocol — the voice sample and everything
   else stay.
2. **Barge-in detection: sustained-speech gate.** While Reachy speaks, a
   monitor thread watches the mic with the existing silero-VAD and fires
   only after ~0.7 s of continuous speech — that hysteresis is what stops
   Reachy's own voice (heard through the mic) from triggering it. No echo
   cancellation, no speaker-ID gate (rejected: adds decision lag and blocks
   strangers from interrupting).
3. **Interrupt semantics: abort the reply.** On barge-in the speaker stops
   mid-word, the in-flight LLM stream is abandoned, and the partial reply
   is kept in history so the conversation stays coherent. The interrupting
   utterance becomes the next turn.

## Architecture

```
ChatBrain._complete_streaming ──sentence──► oracle on_sentence wrapper
        ▲                                   │ monitor fired? ──yes──► raise
        │ catches SpeechInterrupted         ▼                SpeechInterrupted
        │ keeps partial reply       speaker.speak(sentence)
        │                                   ▲
BargeInMonitor (thread, silero-VAD) ──fire──► speaker.stop()
```

### `audio/speak.py` — stoppable speakers

- `Speaker` protocol gains `stop() -> None` (interrupt current playback;
  no-op when idle).
- `SaySpeaker.stop()`: kill the running `say` subprocess (track it via
  `subprocess.Popen` instead of `run`).
- New `QwenSpeaker`:
  - Lazy-loads the mlx-audio TTS model (`settings.tts_model`, default a
    quantized Qwen3-TTS-0.6B community build) on first `speak()`.
  - Reference voice from `settings.voice_sample` (WAV, ~5–10 s); unset →
    the model's default voice.
  - `speak(text)`: synthesize to a numpy array, play via
    `sounddevice.OutputStream` in chunks, blocking until done — but
    checking a stop flag between chunks so `stop()` (called from the
    monitor thread) halts within ~50 ms.
  - Synthesis failure → log once, fall back to `say` for the rest of the
    session (never crash mid-conversation).
- `make_speaker()`: `qwen3` → `QwenSpeaker`; import/load errors → warn and
  return `SaySpeaker`. Unknown backend still raises.

### `audio/listen.py` — BargeInMonitor

- `BargeInMonitor(min_speech_s, sample_rate)` sharing `_AudioCapture`'s VAD:
  - `start()` spawns a daemon thread reading 32 ms mic chunks; a run of
    `min_speech_s / CHUNK_S` consecutive speech-probability > 0.5 chunks
    sets `fired = True` and invokes an `on_fire` callback (used to call
    `speaker.stop()`); the thread then exits.
  - `stop()` ends the thread (called when Reachy finishes speaking
    normally). Restartable per reply.
  - Any exception in the thread: log, set a `broken` flag; the Oracle stops
    arming it for the session (barge-in off, robot fine).
- Mic sharing: the monitor only runs while the robot is speaking and the
  transcriber only records while it isn't, so the device is never opened
  twice concurrently.

### `brain/chat.py` — SpeechInterrupted

- New exception `SpeechInterrupted` (in `chat.py`).
- `_complete_streaming`: wrap the sentence-emitting calls; on
  `SpeechInterrupted` close/abandon the stream and return the partial
  content collected so far, marked so `respond` skips any pending tool
  round-trip.
- `respond`: partial reply is appended to history with a trailing marker
  `" -- (interrupted)"` so the model knows the user cut it off; returns the
  partial text. Exchanges counter still increments.

### `brain/oracle.py` — wiring

Per reply: arm the monitor (`on_fire=speaker.stop`) before calling
`brain.respond`; the `on_sentence` wrapper checks `monitor.fired` before
each sentence and raises `SpeechInterrupted` instead of speaking; after
`respond` returns (normally or interrupted), disarm the monitor. On
interrupt, skip the "nod" motion and loop straight back to listening — the
user is already talking. Constructor takes an optional
`barge_in_factory` (None = feature off) so tests inject a scripted monitor.

### `.vscode/launch.json` + `tasks.json` (local-only; `.vscode/` is gitignored)

- launch: "Oracle (preview)" (`module: reachy_vec.cli`, args
  `["run", "--preview"]`), "Chat" (`["chat"]`), "Enroll" (`["enroll",
  "Yang"]`); all `justMyCode: false`, `.env` loaded via `envFile`.
- tasks: "sim daemon (headless)" background task running
  `uv run reachy-mini-daemon --sim --headless`; the Oracle config lists it
  as `preLaunchTask` with a problem-matcher that returns once the daemon
  is listening. The 3D-viewer variant stays manual (`mjpython` GUI can't
  run as a background task reliably).

## Configuration (new)

| Setting | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `say` | `say` \| `qwen3` |
| `TTS_MODEL` | `mlx-community/Qwen3-TTS-0.6B-4bit` (exact id pinned at implementation) | mlx-audio model id; swap for Voxtral to A/B |
| `VOICE_SAMPLE` | unset | existing knob, now used: reference WAV for cloning |
| `BARGE_IN` | `true` | master switch for interruption |
| `BARGE_IN_MIN_SPEECH_S` | `0.7` | sustained speech needed to fire (raise if false triggers) |

## Error handling

- mlx-audio missing/broken → `say` fallback, logged once.
- Monitor thread exception → barge-in disabled for the session, logged.
- `SpeechInterrupted` never escapes `respond`; tool calls in flight are
  dropped (side effects that already executed stay executed — acceptable:
  tools run before the final spoken sentence).
- `say` backend + barge-in: `stop()` kills the process mid-word; works but
  choppier than the streamed backend.

## Testing

- Fakes: `FakeBargeInMonitor` (scripted fire-after-N-sentences),
  `FakeSpeaker` gains `stop()` recording.
- ChatBrain: interrupt after sentence 1 of 3 → partial history with
  interrupted marker, no tool round-trip, next turn works.
- Oracle: fired monitor → speaker.stop called, no nod, loop listens again;
  broken monitor → conversation proceeds as if feature off.
- Speak: QwenSpeaker chunked playback respects stop flag (fake stream);
  make_speaker fallback path.
- Manual smoke rows (testing.md): interrupt mid-answer → stops within a
  beat and answers the new question; brief "mm-hm" while it talks → no
  trigger; `REACHY_VEC_BARGE_IN=false` → old behavior.

## Known limitations

- No echo cancellation: a loud TV or nearby conversation can still trigger
  a false interrupt; tune `BARGE_IN_MIN_SPEECH_S` up if it happens.
- The robot stops listening for barge-in the moment it finishes speaking
  (normal listening takes over) — interruptions in the last ~0.3 s of a
  reply may land as the next utterance instead. Fine in practice.
- Qwen3-TTS synthesis adds ~0.3–1 s per sentence vs `say`; the streamed
  sentence pipeline hides most of it.

## Out of scope

Echo cancellation, speaker-ID-gated interruption, on-robot audio output,
fish-speech/Ubuntu backends, committing `.vscode/` to the repo.

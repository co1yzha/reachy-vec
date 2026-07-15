# Motion while speaking — design

**Date:** 2026-07-15
**Status:** approved

## Problem

The robot is completely still while replying — head frozen mid-pose for the
length of every answer. It should look alive while it talks.

## Decision

Two paths, matching where the reply audio actually plays:

1. **Robot speaker (`--source robot`) — SDK audio-synced wobble.** The
   `reachy_mini` SDK ships a PTS-driven `HeadWobbler`: audio played by the
   daemon (including incoming WebRTC audio, which is how our qwen-tts replies
   arrive) is analysed and converted into subtle 6-DOF head sway, composed
   with the current target pose daemon-side. Enabling it is one call:
   `mini.enable_wobbling()`. Wire it in `run` after a media-connected robot
   is up, gated by a new knob `speech_wobble: bool = True`
   (`REACHY_VEC_SPEECH_WOBBLE`).

2. **Mac speaker (`--source mac` desk workflow) — gentle sway fallback.**
   No robot-side audio to analyse, so fake it: `SpeakingSway` (new
   `body/sway.py`) runs a background thread looping a soft keyframe sway
   (small alternating yaw/pitch, slight antenna lift, ~1.2 s per cycle)
   through the existing `body.perform`. A thin `Speaker` wrapper in `run.py`
   starts the sway on each `speak()` call and stops it (thread joined)
   before that call returns — replies stream sentence-by-sentence, so sway
   runs per sentence with negligible gaps; `stop()` (barge-in) also halts
   it immediately. With `NullBody` (no robot) it is a harmless no-op.

## Concurrency

During speech the Oracle commands no other motions, so the sway thread is
the body's sole user while active. It joins before `speak()` returns —
normal motions never overlap with sway. The SDK wobble composes with target
poses daemon-side, so it coexists with keyframe motions by design.

## Alternatives rejected

- **Generic sway on both paths:** discards the free, better, voice-synced
  wobble the SDK already provides on the robot path.
- **Per-sentence gestures only:** robot still frozen *while* talking, which
  is the complaint.

## Error handling

`enable_wobbling` failure (old daemon, no media) logs a warning and
continues — motion is decoration, never fatal. Sway thread exceptions are
caught and end the sway (body may be mid-degrade; `ReconnectingBody`
already handles per-motion failures).

## Testing

Against the fakes, test-first:

- `SpeakingSway` performs sway motions on `FakeBody` only between start and
  stop; stop joins the thread.
- The speaker wrapper starts sway on `speak()`, stops it after, and
  `stop()` halts it (barge-in path).
- `run` wiring: `enable_wobbling` called exactly once when robot media is
  active and `speech_wobble` is on; never called when off or media is None.

Wobble feel (amplitude, latency) is a manual smoke check on the robot.

## Scope

New `body/sway.py`, `run.py` wiring, one config knob, tests. Out of scope:
wake-word ("Reachy") voice wake — queued as its own spec.

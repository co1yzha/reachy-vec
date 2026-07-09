# Phase 4 — On-robot deployment (hardware bring-up)

**Date:** 2026-07-09 · **Project:** reachy-vec · **Status:** Draft, pre-implementation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (delivers the
Phase 4 "Hardening / on-robot fallback" line, and the parent's core promise that
"the robot streams audio/video to the Mac" — which no phase has actually wired yet).
**Related:** [phase-2c spec](2026-07-08-phase2c-voice-bargein-design.md) (barge-in,
built here in 4c); [architecture.md → Known gaps](../../architecture.md#known-gaps--toward-a-real-robot-deploy).

## Problem

Everything shipped through Phase 3 runs the perception + speech loop on the
**Mac's own devices**; the robot is motion-only (`body/robot.py` connects with
`media_backend="no_media"`). That's the gap between "demo at my desk" and "a
robot a teammate can walk up to." This phase closes it, in four independently
shippable sub-phases, each keeping the "every heavy dep behind a Protocol with a
test fake" invariant so the Oracle state machine and its pytest suite never change.

The whole point: **the existing seams already fit.** The Reachy Mini SDK's
`MediaManager` (`mini.media`) exposes `get_frame() -> BGR ndarray`,
`get_audio_sample() -> float32 ndarray`, and `push_audio_sample(float32)` /
`play_sound(path)` — a near-1:1 match for our `Camera.read()`, the mic frame loop
in `audio/listen.py:_AudioCapture`, and the `Speaker` protocol. Bring-up is mostly
new Protocol implementations, not new architecture.

## Sub-phases (ship order)

| # | Deliverable | Milestone |
|---|---|---|
| **4a** | On-robot camera, mic, speaker behind existing protocols | Teammate is seen by the *robot's* eye, heard by the *robot's* mic, answered through the *robot's* speaker — brain still on the Mac |
| **4b** | `ROBOT_HOST` wiring + connection resilience | Robot on WiFi (not the local daemon); a dropped link recovers without killing the visit |
| **4c** | Barge-in (implements the phase-2c spec) | Talk over a reply → it stops and listens |
| **4d** | Autostart + offline degradation | Survives a reboot and an OpenAI/WiFi outage without a human at the terminal |

Each milestone is a valid stopping point. 4a is the load-bearing one.

---

## 4a — On-robot media

### Decisions

1. **Use `media_backend="default"`, not `no_media`.** The SDK picks GStreamer when
   the brain runs on the daemon host and WebRTC when it's remote — transparent to
   us. `make_body` calls `mini.acquire_media()` after connecting and
   `mini.release_media()` on the atexit disconnect.
2. **New Protocol implementations, existing protocols unchanged:**
   - `perception/camera.py`: `RobotCamera` wraps `mini.media.get_frame()` (already
     returns a BGR ndarray, so `InsightFaceMatcher` and `PreviewSight` are
     untouched). `WebcamCamera` stays as the dev/no-robot fallback.
   - `audio/listen.py`: factor the raw-frame source out of `_AudioCapture` into a
     tiny `AudioSource` protocol (`read() -> float32 frame | None`). `MicSource`
     (current sounddevice path) and new `RobotAudioSource` (pulls
     `mini.media.get_audio_sample()`). VAD, `collect_utterance`, and STT sit on
     top unchanged.
   - `audio/speak.py`: `Speaker` gains an optional sink. `QwenTTSSpeaker` already
     produces a `(float32 array, sample_rate)` — route it to
     `mini.media.push_audio_sample()` instead of `sounddevice` when a robot sink is
     injected. `SaySpeaker` on-robot renders via `say -o <tempfile>` →
     `mini.media.play_sound(tempfile)` (macOS-only; the array-producing qwen-tts is
     the recommended on-robot backend).
3. **Resampling is explicit.** silero-VAD, whisper, and ECAPA all expect 16 kHz
   mono; the robot mic rate comes from `mini.media.get_input_audio_samplerate()`
   and channels from `get_input_channels()`. `RobotAudioSource` downmixes to mono
   and resamples to 16 kHz. TTS output is resampled to
   `get_output_audio_samplerate()` before `push_audio_sample`. One resample helper,
   unit-tested on arrays (no device).
4. **`--source` picks the world.** `reachy-vec run --source robot|mac` (default
   `robot` when a daemon with media is reachable, else `mac`). Keeps the desk
   workflow alive and makes the smoke test explicit.

### Config (4a)

| Setting | Default | Meaning |
|---|---|---|
| `MEDIA_SOURCE` | `auto` | `auto` \| `robot` \| `mac`; `auto` = robot media if the daemon offers it, else Mac devices (overridable by `--source`) |
| `AUDIO_INPUT_RATE` | `16000` | target rate fed to VAD/STT/ECAPA; the robot source resamples to it |

### Stretch (nice, not required for 4a)

- **Orient to the speaker.** insightface already returns the face bbox; feed its
  centre to `mini.look_at_image(u, v)` so the head turns to whoever it greets.
  Cheap, high-impact embodiment win.

---

## 4b — `ROBOT_HOST` + resilience

### Decisions

1. **Wire the declared knob.** `make_body` passes `host=settings.robot_host` and
   `connection_mode="network"` when `ROBOT_HOST` is set; unset keeps today's
   `connection_mode="auto"` (local daemon). Documented in configuration.md as
   *reserved* today — this makes it real.
2. **A reconnecting body wrapper.** Wrap `RobotBody` so SDK connection errors
   during `perform()` are caught, logged, and retried with backoff on the next
   motion; a persistently-down link degrades to `NullBody` for the rest of the
   session (matches today's fail-soft, but no longer silent — surfaced in the log
   and via one spoken "I've lost my body but I can still hear you"). Media loss is
   handled the same way: fall back to Mac devices if the robot stream dies mid-visit.
3. **No new state-machine branches.** Resilience lives in the body/media adapters,
   not `oracle.py` — the loop keeps calling `perform()` / `sight()` / `speak()`
   and they no-op-or-degrade on failure.

---

## 4c — Barge-in

Implements the already-approved [phase-2c spec](2026-07-08-phase2c-voice-bargein-design.md)
verbatim (`Speaker.stop()`, `BargeInMonitor`, `SpeechInterrupted`, the
`barge_in_factory` Oracle hook, `BARGE_IN` / `BARGE_IN_MIN_SPEECH_S` config). Only
the cloned-voice half of that spec shipped; this is the other half. Two on-robot
adjustments to that spec:

1. `stop()` must halt `push_audio_sample` playback (a stop flag between pushed
   chunks), not just kill a `say` subprocess.
2. The barge-in monitor reads from the same `AudioSource` (4a), so on-robot it
   watches the robot mic. No echo cancellation still (phase-2c's known limitation
   stands, and matters more with the speaker physically near the mic — expect to
   tune `BARGE_IN_MIN_SPEECH_S` up on hardware).

---

## 4d — Autostart + offline degradation

### Decisions

1. **Supervised launch.** A macOS `launchd` plist (documented, not committed —
   like `.vscode/`) starts the sim/robot daemon and `reachy-vec run` on login and
   restarts on crash. README gets a "run it headless / on boot" section.
2. **Graceful brain outage.** OpenAI/network failure already yields a spoken
   apology; add a fast health check at startup and a distinct spoken state ("I
   can't reach my brain right now — try me in a moment") vs. today's mid-turn
   apology, and keep face greetings + message delivery working offline (they don't
   need the LLM). A true **local-LLM fallback** (e.g. an Ollama backend behind the
   existing OpenAI client seam) is attractive but **out of scope** for 4d — noted
   as the natural follow-on.
3. **Ops visibility.** Beyond `data/reachy.log`, emit a one-line periodic heartbeat
   (faces seen, turns handled, last error) so a headless robot's health is legible
   without attaching a terminal.

---

## Error handling (whole phase)

- Robot media unavailable at startup → fall back to Mac devices (`auto`) or hard
  error (`--source robot`), logged clearly.
- Media stream drops mid-visit → degrade to Mac devices if present, else pause
  perception and keep the connection-retry loop running; never crash the loop.
- Resample/format mismatch → log once, drop the frame/sample (a dropped mic frame
  is a silence frame; a dropped TTS chunk is a skipped sentence, as today).
- All failures fail *soft*: a robot that loses its ears should still greet by face
  and deliver queued messages.

## Testing

Fakes keep the suite device-free:

- `FakeCamera` already exists; add `FakeAudioSource` (scripted float32 frames at a
  configurable rate) and `FakeMediaSpeaker` (records pushed arrays). The resample
  helper is pure-function unit-tested (rate in → rate out, mono downmix).
- Adapter tests: `RobotCamera`/`RobotAudioSource`/robot-sink `Speaker` against a
  fake `mini.media` object — no SDK, no hardware.
- Resilience: fake body whose `perform()` raises N times then recovers → wrapper
  retries then degrades; media-drop → source swaps to Mac fake.
- Barge-in: reuse phase-2c's `FakeBargeInMonitor` scenarios.
- **Manual smoke rows** added to testing.md, on real hardware: robot sees/hears/
  speaks (`--source robot`); walk out of WiFi range mid-visit → recovers; pull the
  daemon → falls back to Mac; interrupt mid-answer; reboot the Mac → comes back up
  on its own.

## Out of scope

Local-LLM fallback (4d follow-on), 4-mic sound-source localization / DOA beyond the
single `look_at_image` orient trick, echo cancellation, on-robot inference (brain
stays on the Mac per the parent spec), Linux/Ubuntu backend for TTS, committing
`launchd`/`.vscode` files to the repo.

## Open questions

1. WebRTC audio round-trip latency (robot mic → Mac → robot speaker) vs. today's
   Mac-local ~1 s-to-first-sentence — measure in 4a before committing to the
   streamed-sentence UX; may need a shorter first sentence or a "thinking" filler.
2. Media ownership when the daemon also runs its own apps — does `acquire_media`
   from the brain conflict with daemon-side `play_sound` (used for wobbling)? Verify
   on hardware.
3. Does the robot mic's far-field pickup need a gain/AGC stage before VAD, or is
   silero robust enough at the robot's default levels?

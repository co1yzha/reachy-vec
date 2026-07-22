# Look-around: turn the base to observe and report surroundings

**Date:** 2026-07-22
**Status:** Design approved, pending spec review

## Goal

Give the robot a `look_around` capability: on request ("have a look around",
"what's around you", "what's behind you"), rotate the base to a few directions,
capture the camera at each, and describe what it saw in one or two spoken
sentences. Builds on the existing `look()` vision tool and the base-rotation
motor (`body_yaw`).

## Background (current state)

- `RobotBody.perform(motion)` (`body/robot.py`) maps named `Keyframe` lists to
  `mini.goto_target(head=, antennas=, duration=)`. The SDK's `goto_target`
  signature defaults `body_yaw=0.0`, so **every motion already resets the base
  to front**; keyframes carry no yaw today.
- Base yaw joint range is ±2.79 rad (≈ ±160°); ±120° (2.09 rad) is a safe
  working sweep.
- The `Body` protocol (`body/robot.py`) is deliberately minimal — just
  `perform(motion: str) -> None`. `NullBody`, `ReconnectingBody`, and all test
  fakes implement only that. We keep it that way.
- `make_look_fn(camera, client, model, max_px, body)` (`perception/vision.py`)
  is the reusable pattern: perform a gesture, `camera.read()`, one vision call,
  return a short spoken string; every side-effect is best-effort.
- Camera tools (`look`, `selfie`) are injected into `ChatBrain` only by `run`
  (`cli/run.py`), registered as OpenAI tools and dispatched in `chat.py`.
- Vision API calls accept multiple images in a single message.

## Approach (chosen: A)

Orchestrate the turn-capture-describe sequence in a closure, using new
directional "hold" motions. The `Body` protocol stays `perform(str)` only.
Rejected alternatives: adding a `turn(yaw)` method to the protocol (widens the
interface and forces edits to every fake) and a camera-less gesture (not the
requested behavior).

## Design

### 1. Motions (`body/motions.py`)

- `Keyframe` gains `body_yaw: float | None = None` (radians; `None` = keep
  current yaw per the SDK).
- Module constant `LOOK_AROUND_YAW_RAD` (≈ 2.09, i.e. 120°) defines the sweep
  angle. Kept as a module constant, not a config knob: `MOTIONS` is a static
  module-level dict built at import, and reaching into runtime settings there
  would be an import-time side effect. Edit the constant to retune.
- New named motions that rotate the base and **hold** (no return-to-neutral
  keyframe):
  - `turn_left`  — `body_yaw = +LOOK_AROUND_YAW_RAD`, antennas perked.
  - `turn_right` — `body_yaw = -LOOK_AROUND_YAW_RAD`, antennas perked.
  - `turn_front` — `body_yaw = 0.0`, antennas neutral (recenters).

### 2. perform pass-through (`body/robot.py`)

`RobotBody.perform` passes `body_yaw=kf.body_yaw` to `goto_target` when set.
Existing motions (no `body_yaw`) keep today's behavior — the SDK default resets
to front, which is the current, unchanged effect.

### 3. Orchestrator (`perception/vision.py`)

`make_look_around_fn(camera, client, model, max_px, body, *, settle_s)` returns
a `look_around()` closure:

1. `body.perform("turn_left")`; sleep `settle_s`; `frame_left = camera.read()`.
2. `body.perform("turn_right")`; sleep `settle_s`; `frame_right = camera.read()`.
3. `body.perform("turn_front")` (recenter).
4. One vision call: system prompt = the robot's eyes; user message carries both
   frames labelled "to my left" / "to my right" → one or two spoken sentences.
5. Return the spoken string.

Best-effort throughout, mirroring `look`:
- No `body` → skip turns, capture once forward (degenerate but functional).
- A `None` frame from a direction → that direction is omitted from the vision
  call; if **both** are `None`, return "I can't see anything right now …".
- Vision-call failure → "I had trouble seeing just now." Always attempt
  `turn_front` recenter even on failure.

Reuse `encode_frame_jpeg` and the `VISION_SYSTEM` constant already in the module.

### 4. Tool registration (`brain/chat.py`)

- Add a `look_around` tool schema (no required args) with a description steering
  the brain to call it for "have a look around", "what's around/behind you",
  "check the room".
- `ChatBrain.__init__` accepts `look_around_fn` (like `look_fn`/`selfie_fn`);
  the tool is included in `_active_tools()` only when the closure is present.
- Dispatch `_tool_look_around(args)` → calls the closure, guarded like
  `_tool_look`.

### 5. Wiring (`cli/run.py`)

Build `look_around_fn = make_look_around_fn(camera, client, model=…,
max_px=…, body=body, settle_s=settings.look_around_settle_s)` alongside
`look_fn`, and pass it into `ChatBrain`.

### 6. Config (`config.py`)

- `look_around_settle_s: float = 0.8` — pause after a turn before capture.
  `REACHY_VEC_`-prefixed via the existing pydantic-settings model.

The sweep angle is the `LOOK_AROUND_YAW_RAD` module constant in
`body/motions.py` (see §1), not a config knob.

## Testing (fakes only, no devices/network)

- `FakeBody` recording `perform` calls; `FakeCamera` returning canned frames;
  fake OpenAI client capturing the messages.
- Assert perform ordering is `turn_left` → `turn_right` → `turn_front`.
- Assert exactly one vision call is made and it carries both images.
- `body=None` path captures once and still returns a description.
- A `None` frame for one direction drops that image; both `None` → the
  no-camera message.
- Tool wiring: `look_around` appears in `_active_tools()` only when the closure
  is provided; dispatch routes to it.

## Out of scope (YAGNI)

- Directions fixed at left / right / front — no configurable count, no full
  360° panorama, no per-direction separate utterances.
- No new `Body` protocol method; no changes to `NullBody`/`ReconnectingBody`
  interface beyond the shared `perform` pass-through already covered by (2).
```

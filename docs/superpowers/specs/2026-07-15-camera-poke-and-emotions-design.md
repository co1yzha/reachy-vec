# Camera-poke reaction + built-in emotion moves — design

**Date:** 2026-07-15
**Status:** Approved, ready for implementation planning

## Summary

Two related expressiveness features:

1. **Camera-poke reaction.** When someone covers or pokes the camera (the frame
   goes suddenly dark), the robot startles: it wakes if asleep, plays a startle
   emote, and says a short line ("Hey — I can't see!"). Covering the lens
   thereby becomes a hands-on wake gesture, complementing the existing
   face-triggered wake.
2. **Built-in emotion moves in conversation.** The Reachy Mini SDK ships 81
   recorded expressive moves (`pollen-robotics/reachy-mini-emotions-library`,
   played via `mini.play_move()`). The brain gets a `react(emotion)` tool with a
   curated ~16-move subset so replies can carry situational body language, and
   two fixed cues upgrade from hand-made keyframes to recorded moves
   (`greet` → `welcoming1`, the new `startle` → `scared1`).

## Part 1 — Camera-poke reaction

### Detection: brightness-drop, on frames we already capture

`perception/occlusion.py` gains an `OcclusionDetector` with a single method
`update(frame) -> bool`, pure numpy, no models:

- Maintains an exponential moving average of mean frame brightness while
  unoccluded (the baseline).
- Fires (returns `True` once) when the current frame's mean brightness is below
  **both** an absolute dark floor and a fraction of the baseline, for N
  consecutive polls (debounce).
- After firing it is **disarmed** until brightness recovers above the re-arm
  level, and additionally respects a cooldown, so a hand held over the lens
  produces one reaction, not a loop.
- A `None` frame is ignored (camera hiccup, not occlusion).

Config knobs (`config.py`, `REACHY_VEC_` prefix):

| knob | default | meaning |
|---|---|---|
| `occlusion_dark_floor` | 28.0 | absolute mean-brightness ceiling for "dark" (0–255) |
| `occlusion_drop_ratio` | 0.3 | dark also means below this fraction of baseline |
| `occlusion_polls` | 2 | consecutive dark polls needed to fire |
| `occlusion_cooldown_s` | 10.0 | minimum time between reactions |

Rejected alternatives: frame-diff / face-size heuristics (false-positives when
people lean in); IMU/touch sensing (not exposed through the `Body` seam).

### Reaction: wake + startle + spoken line

`OracleLoop` accepts an optional `occlusion` collaborator (default `None` =
feature off, as with `speaker_id` and `barge_in_factory`). In the `run_once`
poll loop — the only place frames are polled (idle and asleep) — a trip:

1. wakes the robot if asleep (reuses `_note_presence`-style wake, without
   requiring a face),
2. performs `"startle"`,
3. speaks a random line from a small `STARTLE_LINES` set,
4. continues polling (no conversation is started; if a face is there, the
   normal flow takes over on the next poll).

Out of scope (YAGNI): detection mid-conversation (frames are not polled there);
"poking near" without covering.

### Wiring

`cli/run.py` composes the detector into the existing sight path so the same
frame feeds face matching and occlusion — both the plain lambda and
`PreviewSight` variants — and passes the detector to `OracleLoop`.

## Part 2 — Built-in emotion moves

### Body seam stays a single `perform(name)`

`RobotBody.perform` resolves a motion name in order:

1. local keyframe `MOTIONS` (fast, hand-tuned, works offline),
2. the recorded-moves library, lazy-loaded on first use.

Library load failure (offline with a cold HF cache) logs once and permanently
falls back — unknown names become no-ops, exactly like today's unknown-motion
behavior. `NullBody` and the test `FakeBody` are unchanged: emotion names are
just strings flowing through the same seam, so the whole suite stays
device-free.

Recorded moves may carry sidecar sounds; `play_move(..., sound=...)` is driven
by a new `emotion_sound: bool = False` knob — the daemon allows one media
client and we hold it for TTS/mic, so move audio stays off unless explicitly
enabled.

`play_move` blocks for the move's duration (2–6 s), consistent with the loop's
synchronous design; a mid-reply emote reads as a deliberate pause.

### The `react` tool

`brain/chat.py` gains a `react(emotion)` tool, injected by `run` the same way
as `look`/`selfie` (so `chat` CLI and tests without a body don't see it). The
enum maps friendly names to library moves:

curious, laughing, surprised, thoughtful, yes, no, oops, proud, sad, grateful,
confused, enthusiastic, calming, attentive, welcoming, scared → `curious1`,
`laughing1`, `surprised1`, `thoughtful1`, `yes1`, `no1`, `oops1`, `proud1`,
`sad1`, `grateful1`, `confused1`, `enthusiastic1`, `calming1`, `attentive1`,
`welcoming1`, `scared1`.

Tool description instructs the model to emote **sparingly**, when it genuinely
fits the moment, at most once per reply.

### Fixed upgrades

- `greet` → `welcoming1`
- `startle` → `scared1` (with a keyframe `startle` fallback in `MOTIONS` for
  offline/sim-without-library)
- `nod` deliberately stays a fast keyframe — it fires every turn and a
  multi-second recorded move would drag the conversation.

The upgrade map lives next to `MOTIONS` so keyframe fallbacks and upgrades are
defined in one place.

## Testing

- `OcclusionDetector` unit tests with synthetic frames: fire on sustained
  darkness, debounce (one dark poll is not enough), re-arm only after
  recovery, cooldown, `None` frames ignored.
- Oracle tests with a scripted occlusion fake: asleep → wakes + startles +
  speaks; awake → startles without re-waking; no trip → no reaction.
- Body tests: keyframe names resolve before library names; library load
  failure degrades to no-op; `emotion_sound` flag reaches `play_move`.
- Chat tests: `react` tool call performs the mapped move (via `FakeBody`) and
  the reply continues normally; unknown emotion name is rejected by the enum.
- Hardware-facing pieces (real `play_move`, real camera darkness) go on the
  `docs/testing.md` smoke checklist.

# Wake-up flourish — design

**Date:** 2026-07-15
**Status:** approved

## Problem

Waking from sleep is easy to miss. `OracleLoop._maybe_wake` calls
`body.perform("wake")`, which maps to the SDK's `wake_up()`: a return to the
neutral head pose, a small roll, and an on-robot "toudoum" sound. In body-only
mode (`--source mac`, `media_backend="no_media"`) the sound is skipped
entirely ("Audio system is not initialized"), so a wake is a barely visible
head twitch. The user wants an unmistakable "I'm here with you" signal every
time Reachy wakes after sleep.

## Decision

App-level motion + spoken line (not SDK sounds, not body-layer speech):

1. **`wakeup` motion** — new entry in `body/motions.py:MOTIONS`, same
   `Keyframe` format as the existing motions. A noticeable ~3–4 s flourish:
   head stretches up, looks left then right (checking who's there), antennas
   wiggle, settle to neutral. Plays identically on sim and real robot.
2. **`WAKE_LINES`** — small module-level tuple in `brain/oracle.py` (~4 short
   lines, e.g. "Mm — I'm up!", "Oh! Hello again."), picked with
   `random.choice`. Spoken through the loop's existing `Speaker`, so it is
   audible on whichever speaker is active (Mac, robot, or fake).
3. **Wiring** — in `_maybe_wake`, after `perform("wake")`:
   `perform("wakeup")`, then `speaker.speak(random.choice(WAKE_LINES))`.
   The normal named greeting then follows unchanged (wake line first, greet
   second — chosen over merging them to keep greeting logic untouched).

## Alternatives rejected

- **Inside `RobotBody.perform("wake")`:** the body layer would need a
  speaker; breaks the module-role convention (`body/` motions, `audio/` I/O).
- **SDK `play_sound` emote:** only works when robot media is initialized;
  silent in `--source mac` mode — exactly the current problem.

## Error handling

Nothing new. Motions already degrade via `NullBody` / `ReconnectingBody`
(skip, reconnect, or announce-once); a speak failure follows existing
speaker behavior.

## Testing

Extend the oracle wake scenario against the fakes (test-first):

- waking from sleep records `wake` then `wakeup` on the fake body;
- exactly one wake line is spoken and it is a member of `WAKE_LINES`;
- the normal named greeting still follows.

## Scope

Two source files (`body/motions.py`, `brain/oracle.py`) plus tests. No new
config knob — a `REACHY_VEC_WAKE_LINES` override can be added later if
customization is wanted (YAGNI now).

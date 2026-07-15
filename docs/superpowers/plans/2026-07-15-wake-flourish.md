# Wake-up Flourish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Reachy wakes from idle sleep, play a noticeable ~3.5 s "wakeup" flourish motion and speak one short wake line, so the user can't miss that it's awake.

**Architecture:** Pure keyframe data added to `body/motions.py:MOTIONS` (same `Keyframe` format as the existing eight motions), wired in `brain/oracle.py:OracleLoop._note_presence` right after the existing `perform("wake")`, with a spoken line chosen by `random.choice` from a new module-level `WAKE_LINES` tuple via the loop's existing `Speaker`. No body-layer speech, no SDK sounds (spec: `docs/superpowers/specs/2026-07-15-wake-flourish-design.md`).

**Tech Stack:** Python 3.12, pytest against the fakes in `tests/conftest.py` (`FakeBody` records motion names; `FakeSpeaker` records spoken lines). No devices or network needed.

## Global Constraints

- Run `uv run pytest -q` and `uv run ruff check src tests` before every commit; both must pass (repo has no CI).
- Test-first: write the failing test, see it fail, then implement (repo convention).
- Head kwargs in `Keyframe` are degrees for `roll`/`pitch`/`yaw`; only use keys from `{"x", "y", "z", "roll", "pitch", "yaw"}` — existing motions use rotations only, do the same.
- No new config knobs (spec explicitly defers a `REACHY_VEC_WAKE_LINES` override — YAGNI).

---

### Task 1: `wakeup` motion keyframes

**Files:**
- Modify: `src/reachy_vec/body/motions.py` (add one entry to `MOTIONS`, after `"pose"`)
- Test: `tests/test_body.py` (update `EXPECTED`, add one test)

**Interfaces:**
- Consumes: `Keyframe`, `NEUTRAL`, `MOTIONS` already defined in `src/reachy_vec/body/motions.py:11-19`.
- Produces: `MOTIONS["wakeup"]: list[Keyframe]` — Task 2's oracle wiring calls `body.perform("wakeup")`, which resolves through this dict (`RobotBody.perform`, `src/reachy_vec/body/robot.py:30`). Unknown names only log a warning, so Task 2 technically runs without this — but the robot would not move.

- [ ] **Step 1: Write the failing test**

In `tests/test_body.py`, change line 5 to include the new name:

```python
EXPECTED = {"greet", "nod", "listen", "idle", "acknowledge", "goodbye", "look", "pose", "wakeup"}
```

and add after `test_look_and_pose_motions_exist_and_are_valid` (line 27):

```python
def test_wakeup_motion_is_long_noticeable_and_ends_neutral():
    frames = MOTIONS["wakeup"]
    assert sum(kf.duration for kf in frames) >= 3.0          # noticeable, not a twitch
    assert any(kf.head.get("yaw", 0) > 0 for kf in frames)   # looks left...
    assert any(kf.head.get("yaw", 0) < 0 for kf in frames)   # ...and right
    assert frames[-1].head == {} and frames[-1].antennas == (0.0, 0.0)  # ends neutral
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -q`
Expected: 2 failures — `test_all_motions_defined_and_well_formed` (set mismatch: `wakeup` missing) and `test_wakeup_motion_is_long_noticeable_and_ends_neutral` (`KeyError: 'wakeup'`).

- [ ] **Step 3: Implement the motion**

In `src/reachy_vec/body/motions.py`, add to `MOTIONS` after the `"pose"` entry (keep the closing `}` of the dict after it):

```python
    "wakeup": [
        # stretch up, antennas high
        Keyframe(head={"pitch": -20}, antennas=(0.9, 0.9), duration=0.8),
        # look left, then right — "who's there?"
        Keyframe(head={"pitch": -10, "yaw": 25}, antennas=(0.6, -0.6), duration=0.6),
        Keyframe(head={"pitch": -10, "yaw": -25}, antennas=(-0.6, 0.6), duration=0.6),
        # quick antenna wiggle
        Keyframe(head={"pitch": -5}, antennas=(0.8, -0.8), duration=0.3),
        Keyframe(head={"pitch": -5}, antennas=(-0.8, 0.8), duration=0.3),
        NEUTRAL,
    ],
```

(Total 3.0 s + `NEUTRAL`'s 0.4 s = 3.4 s.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check src tests
uv run pytest -q
git add src/reachy_vec/body/motions.py tests/test_body.py
git commit -m "feat: add 'wakeup' flourish motion (stretch, look around, antenna wiggle)"
```

---

### Task 2: speak a wake line and play the flourish on wake

**Files:**
- Modify: `src/reachy_vec/brain/oracle.py` (add `WAKE_LINES` constant near the top after the imports; extend `_note_presence`, currently lines 241-246)
- Test: `tests/test_oracle.py` (extend `test_sleeps_after_idle_and_wakes_on_face`, lines 170-194)

**Interfaces:**
- Consumes: `MOTIONS["wakeup"]` from Task 1 (via `self._body.perform("wakeup")`); `self._speaker.speak(text: str)` and `self._body.perform(motion: str)` already on `OracleLoop`.
- Produces: `WAKE_LINES: tuple[str, ...]` module-level in `reachy_vec.brain.oracle` — the test imports it to assert membership.

- [ ] **Step 1: Write the failing test**

In `tests/test_oracle.py`, replace the last three lines of `test_sleeps_after_idle_and_wakes_on_face` (lines 192-194):

```python
    sights.append(ALICE)                         # someone walks up
    assert loop.run_once() == "conversation"
    assert body.motions[1] == "wake"             # woke before greeting
```

with:

```python
    sights.append(ALICE)                         # someone walks up
    assert loop.run_once() == "conversation"
    from reachy_vec.brain.oracle import WAKE_LINES

    assert body.motions[1:3] == ["wake", "wakeup"]   # woke with a flourish, before greeting
    assert speaker.spoken[0] in WAKE_LINES           # announced it's awake, before the greeting
    assert len(speaker.spoken) >= 2                  # normal greeting still follows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py::test_sleeps_after_idle_and_wakes_on_face -q`
Expected: FAIL — `ImportError: cannot import name 'WAKE_LINES'`.

- [ ] **Step 3: Implement**

In `src/reachy_vec/brain/oracle.py`:

Add `import random` to the imports at the top of the file (alphabetical order with the other stdlib imports).

Add a module-level constant after the imports, before the `OracleLoop` class:

```python
# Short lines spoken on waking from idle sleep, so a wake is unmissable.
WAKE_LINES = (
    "Mm — I'm up!",
    "Oh! Hello again.",
    "Yawn... I'm awake now.",
    "Back with you!",
)
```

Change `_note_presence` (currently lines 241-246):

```python
    def _note_presence(self) -> None:
        self._last_face_at = self._clock()
        if self._asleep:
            logger.info("face detected - waking up")
            self._body.perform("wake")
            self._body.perform("wakeup")
            self._speaker.speak(random.choice(WAKE_LINES))
            self._asleep = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle.py -q`
Expected: all pass (the whole file, not just the one test — other scenarios also wake and must still pass).

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check src tests
uv run pytest -q
git add src/reachy_vec/brain/oracle.py tests/test_oracle.py
git commit -m "feat: noticeable wake-up — flourish motion + spoken wake line"
```

---

## Manual smoke test (after both tasks)

Per `docs/testing.md` convention for hardware-facing work — with the robot on WiFi and `REACHY_VEC_ROBOT_HOST=reachy-mini.local` in `.env`:

```bash
uv run reachy-vec run --preview --source mac
```

Walk away (or cover the webcam) for `idle_sleep_s` (default 300 s; export `REACHY_VEC_IDLE_SLEEP_S=20` to shorten the wait), then step back into frame. Expected: robot plays the stretch/look-around flourish and speaks one of the wake lines, then greets you by name as before.

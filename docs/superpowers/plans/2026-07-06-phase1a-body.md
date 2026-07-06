# Phase 1a: Body — Motions in Simulation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Named motion primitives (greet, nod, listen, idle, acknowledge, goodbye) playable on the simulated robot, with graceful degradation when no daemon is running.

**Architecture:** Motions are pure data — lists of keyframes (head pose kwargs + antenna angles + duration) in `body/motions.py`, unit-testable without a robot. `body/robot.py` provides a `Body` protocol with two implementations: `RobotBody` (plays keyframes via the reachy-mini SDK) and `NullBody` (logs and no-ops). A `make_body()` factory tries the daemon and falls back to `NullBody` — body errors never kill a caller.

**Tech Stack:** reachy-mini SDK (`ReachyMini.goto_target`, `reachy_mini.utils.create_head_pose`), pytest.

**Depends on:** nothing (Phase 0 merged). **Unblocks:** 1d.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- Tests never open devices or connect to the daemon: `RobotBody` is exercised only by the manual smoke test.
- Settings come from `reachy_vec.config.settings` (`REACHY_VEC_` env prefix).
- Commit after every green test cycle; conventional-commit messages.

---

### Task 1: Motion keyframes + Body protocol

**Files:**
- Modify: `src/reachy_vec/body/motions.py` (docstring stub)
- Modify: `src/reachy_vec/body/robot.py` (docstring stub)
- Test: `tests/test_body.py`

**Interfaces:**
- Produces: `Keyframe` dataclass `(head: dict[str, float], antennas: tuple[float, float], duration: float)`; `MOTIONS: dict[str, list[Keyframe]]` with keys exactly `{"greet", "nod", "listen", "idle", "acknowledge", "goodbye"}`; `Body` protocol with `perform(motion: str) -> None`; `NullBody` (no-op); `RobotBody(mini)` playing keyframes; `make_body() -> Body` factory. 1d consumes `Body.perform` and `make_body`.

- [x] **Step 1: Write the failing test**

`tests/test_body.py`:

```python
from reachy_vec.body.motions import MOTIONS, Keyframe
from reachy_vec.body.robot import Body, NullBody, RobotBody

EXPECTED = {"greet", "nod", "listen", "idle", "acknowledge", "goodbye"}


def test_all_motions_defined_and_well_formed():
    assert set(MOTIONS) == EXPECTED
    for name, frames in MOTIONS.items():
        assert frames, name
        for kf in frames:
            assert isinstance(kf, Keyframe)
            assert kf.duration > 0
            assert len(kf.antennas) == 2
            assert set(kf.head) <= {"x", "y", "z", "roll", "pitch", "yaw"}


def test_null_body_is_silent_noop():
    body: Body = NullBody()
    body.perform("greet")  # must not raise
    body.perform("nonexistent")  # unknown motions are ignored, not fatal


class RecordingMini:
    def __init__(self):
        self.calls = []

    def goto_target(self, head=None, antennas=None, duration=0.5):
        self.calls.append((head is not None, tuple(antennas), duration))


def test_robot_body_plays_each_keyframe():
    mini = RecordingMini()
    body = RobotBody(mini)
    body.perform("nod")
    assert len(mini.calls) == len(MOTIONS["nod"])
    assert all(duration > 0 for _, _, duration in mini.calls)


def test_robot_body_ignores_unknown_motion():
    mini = RecordingMini()
    RobotBody(mini).perform("moonwalk")
    assert mini.calls == []
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_body.py -v`
Expected: FAIL with `ImportError` (no `MOTIONS`/`Keyframe` yet).

- [x] **Step 3: Write the motions**

Replace `src/reachy_vec/body/motions.py` with:

```python
"""Expressive motion primitives as pure keyframe data.

Head kwargs feed reachy_mini.utils.create_head_pose (degrees, mm=False);
antennas are (left, right) in radians; duration in seconds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Keyframe:
    head: dict[str, float]
    antennas: tuple[float, float]
    duration: float


NEUTRAL = Keyframe(head={}, antennas=(0.0, 0.0), duration=0.4)

MOTIONS: dict[str, list[Keyframe]] = {
    "greet": [
        Keyframe(head={"pitch": -15, "yaw": 10}, antennas=(0.6, -0.6), duration=0.4),
        Keyframe(head={"pitch": -15, "yaw": -10}, antennas=(-0.6, 0.6), duration=0.4),
        NEUTRAL,
    ],
    "nod": [
        Keyframe(head={"pitch": 15}, antennas=(0.0, 0.0), duration=0.3),
        NEUTRAL,
    ],
    "listen": [
        Keyframe(head={"pitch": -8, "roll": 6}, antennas=(0.3, 0.3), duration=0.5),
    ],
    "idle": [
        Keyframe(head={"yaw": 6}, antennas=(0.1, 0.1), duration=1.2),
        Keyframe(head={"yaw": -6}, antennas=(0.1, 0.1), duration=1.2),
        NEUTRAL,
    ],
    "acknowledge": [
        Keyframe(head={"yaw": 12}, antennas=(0.5, 0.5), duration=0.3),
        NEUTRAL,
    ],
    "goodbye": [
        Keyframe(head={"pitch": 18}, antennas=(-0.8, -0.8), duration=0.6),
        NEUTRAL,
    ],
}
```

- [x] **Step 4: Write the bodies**

Replace `src/reachy_vec/body/robot.py` with:

```python
"""Body implementations: real robot/sim via the SDK, or a logging no-op."""

import logging
from typing import Protocol

from reachy_vec.body.motions import MOTIONS

logger = logging.getLogger(__name__)


class Body(Protocol):
    def perform(self, motion: str) -> None: ...


class NullBody:
    """Used when no daemon is reachable; motions become logged no-ops."""

    def perform(self, motion: str) -> None:
        logger.debug("NullBody: skipping motion %r", motion)


class RobotBody:
    """Plays keyframes on a connected ReachyMini (sim or real)."""

    def __init__(self, mini):
        self._mini = mini

    def perform(self, motion: str) -> None:
        frames = MOTIONS.get(motion)
        if frames is None:
            logger.warning("Unknown motion %r", motion)
            return
        from reachy_mini.utils import create_head_pose

        for kf in frames:
            self._mini.goto_target(
                head=create_head_pose(**kf.head),
                antennas=list(kf.antennas),
                duration=kf.duration,
            )


def make_body() -> Body:
    """Connect to the daemon if possible; otherwise degrade to NullBody."""
    try:
        from reachy_mini import ReachyMini

        mini = ReachyMini(media_backend="no_media")
        return RobotBody(mini)
    except Exception as exc:  # daemon down, robot absent, etc.
        logger.warning("No robot/daemon available (%s); running body-less.", exc)
        return NullBody()
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS.

- [x] **Step 6: Manual smoke test (sim viewer)**

```bash
uv run mjpython .venv/bin/reachy-mini-daemon --sim --no-media &   # viewer opens
sleep 8
uv run python -c "
from reachy_vec.body.robot import make_body
import time
body = make_body()
for m in ('greet', 'nod', 'listen', 'acknowledge', 'goodbye', 'idle'):
    print('playing', m); body.perform(m); time.sleep(0.5)
"
pkill -f reachy-mini-daemon
```

Expected: each motion visibly plays in the viewer. Also verify degradation: run the same python snippet with no daemon — it should print warnings and finish without error.

- [x] **Step 7: Commit**

```bash
git add src/reachy_vec/body/motions.py src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: body motion primitives with sim playback and null fallback"
```

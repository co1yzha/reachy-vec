# Motion While Speaking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The robot moves while it talks — SDK audio-synced head wobble when replies play on the robot's speaker, a gentle keyframe sway loop when they play on the Mac.

**Architecture:** Robot path is one SDK call (`mini.enable_wobbling()`) inside `make_robot`, gated by a new `speech_wobble` knob. Mac path is a new `body/sway.py`: `SpeakingSway` (background thread looping a new `"sway"` keyframe motion via `body.perform`) plus `SwayingSpeaker` (a `Speaker` decorator that brackets each `speak()` with sway start/stop), wired in `run.py` only when the chosen media source is not `robot`. Spec: `docs/superpowers/specs/2026-07-15-speech-motion-design.md`.

**Tech Stack:** Python 3.12, `threading` stdlib, pytest against the fakes in `tests/conftest.py` / `tests/test_body.py` (`FakeBody` records motion names; `FakeMini` in `tests/test_body.py:81` fakes the SDK object).

## Global Constraints

- Run `uv run pytest -q` and `uv run ruff check src tests` before every commit; both must pass (no CI).
- Test-first: write the failing test, watch it fail, then implement.
- Motion is decoration, never fatal: every failure path logs/continues (spec "Error handling").
- New config knobs go in `config.py` as `pydantic-settings` fields (`REACHY_VEC_` prefix), never module constants.
- Keyframe head kwargs use only `{"x","y","z","roll","pitch","yaw"}`; existing motions use rotations only — do the same.

---

### Task 1: `speech_wobble` knob + `enable_wobbling` in `make_robot`

**Files:**
- Modify: `src/reachy_vec/config.py` (after `barge_in_min_speech_s`, line ~59)
- Modify: `src/reachy_vec/body/robot.py` (inside `make_robot`, after `mini.acquire_media()`, line ~160)
- Test: `tests/test_body.py`

**Interfaces:**
- Consumes: `FakeMini` (tests/test_body.py:81), `make_robot(with_media, connect, pre_acquire)`, `settings` from `reachy_vec.config`.
- Produces: `settings.speech_wobble: bool` (default True) — Task 3's wiring does NOT read it (wobble is entirely inside make_robot); nothing else depends on this task.

- [ ] **Step 1: Write the failing tests**

In `tests/test_body.py`, add to `FakeMini` (after `release_media`, line 96):

```python
    def enable_wobbling(self):
        self.wobbling = True
```

and add `self.wobbling = False` to `FakeMini.__init__` (after `self.acquired = self.released = False`). Then add after `test_make_robot_preacquire_failure_is_not_fatal`:

```python
def test_make_robot_enables_speech_wobble_with_media(monkeypatch):
    monkeypatch.setattr(_settings, "speech_wobble", True)
    mini = FakeMini()
    make_robot(with_media=True, connect=lambda **kw: mini, pre_acquire=lambda: None)
    assert mini.wobbling


def test_make_robot_skips_wobble_when_knob_off(monkeypatch):
    monkeypatch.setattr(_settings, "speech_wobble", False)
    mini = FakeMini()
    make_robot(with_media=True, connect=lambda **kw: mini, pre_acquire=lambda: None)
    assert not mini.wobbling


def test_make_robot_skips_wobble_without_media(monkeypatch):
    monkeypatch.setattr(_settings, "speech_wobble", True)
    mini = FakeMini()
    make_robot(with_media=False, connect=lambda **kw: mini)
    assert not mini.wobbling


def test_make_robot_wobble_failure_is_not_fatal(monkeypatch):
    monkeypatch.setattr(_settings, "speech_wobble", True)

    class WobbleBoomMini(FakeMini):
        def enable_wobbling(self):
            raise RuntimeError("old daemon")

    body, media = make_robot(
        with_media=True, connect=lambda **kw: WobbleBoomMini(), pre_acquire=lambda: None
    )
    assert not isinstance(body, NullBody)
    assert media is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -q`
Expected: `test_make_robot_enables_speech_wobble_with_media` FAILS (`speech_wobble` attribute missing / wobbling False); the knob-off and without-media tests may pass vacuously; the failure test errors on missing settings attr.

- [ ] **Step 3: Implement**

In `src/reachy_vec/config.py`, after the `barge_in_min_speech_s` line:

```python
    speech_wobble: bool = True  # audio-synced head sway while the robot speaker plays
```

In `src/reachy_vec/body/robot.py`, inside `make_robot`'s `if with_media:` success block, directly after `mini.acquire_media()`:

```python
            if settings.speech_wobble:
                try:
                    mini.enable_wobbling()
                except Exception as exc:
                    logger.warning("could not enable speech wobble (%s); continuing.", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check src tests
uv run pytest -q
git add src/reachy_vec/config.py src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: audio-synced head wobble while the robot speaker plays (speech_wobble knob)"
```

---

### Task 2: `"sway"` motion + `SpeakingSway` background loop

**Files:**
- Modify: `src/reachy_vec/body/motions.py` (add `"sway"` entry after `"wakeup"`)
- Create: `src/reachy_vec/body/sway.py`
- Test: `tests/test_body.py` (EXPECTED set), Create: `tests/test_sway.py`

**Interfaces:**
- Consumes: `Keyframe`, `NEUTRAL`, `MOTIONS` (`body/motions.py`); `Body.perform(motion: str)` protocol.
- Produces: `SpeakingSway(body)` with `.start() -> None` and `.stop() -> None` (idempotent; `stop` joins the thread). `MOTIONS["sway"]`. Task 3 constructs `SpeakingSway(body)` and calls both methods.

- [ ] **Step 1: Write the failing tests**

In `tests/test_body.py` line 5, add `"sway"`:

```python
EXPECTED = {"greet", "nod", "listen", "idle", "acknowledge", "goodbye", "look", "pose", "wakeup", "sway"}
```

Create `tests/test_sway.py`:

```python
import time

from reachy_vec.body.sway import SpeakingSway
from tests.test_body import FlakyBody


class SlowBody:
    """Records motions; each perform takes a beat, like real keyframes."""

    def __init__(self):
        self.motions: list[str] = []

    def perform(self, motion: str) -> None:
        self.motions.append(motion)
        time.sleep(0.01)


def _wait_for_motions(body, deadline_s=2.0):
    deadline = time.time() + deadline_s
    while not body.motions and time.time() < deadline:
        time.sleep(0.005)


def test_speaking_sway_performs_only_between_start_and_stop():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.start()
    _wait_for_motions(body)
    sway.stop()
    count = len(body.motions)
    assert count >= 1
    assert set(body.motions) == {"sway"}
    time.sleep(0.05)
    assert len(body.motions) == count  # nothing after stop (thread joined)


def test_speaking_sway_start_and_stop_are_idempotent():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.stop()  # stop before start: no-op
    sway.start()
    sway.start()  # second start: no second thread
    _wait_for_motions(body)
    sway.stop()
    sway.stop()  # double stop: no error


def test_speaking_sway_ends_quietly_when_body_raises():
    sway = SpeakingSway(FlakyBody(fail=99))  # every perform raises
    sway.start()
    time.sleep(0.05)
    sway.stop()  # thread already dead from the exception; join is clean
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sway.py tests/test_body.py -q`
Expected: `test_sway.py` errors with `ModuleNotFoundError: reachy_vec.body.sway`; `test_all_motions_defined_and_well_formed` fails on the set mismatch.

- [ ] **Step 3: Implement**

In `src/reachy_vec/body/motions.py`, add after the `"wakeup"` entry (inside `MOTIONS`):

```python
    "sway": [
        # soft talking sway - looped by SpeakingSway while a reply plays
        Keyframe(head={"yaw": 4, "pitch": -3}, antennas=(0.2, 0.15), duration=0.6),
        Keyframe(head={"yaw": -4, "pitch": -1}, antennas=(0.15, 0.2), duration=0.6),
    ],
```

(No trailing `NEUTRAL`: the loop repeats seamlessly; the next real motion re-poses the head anyway.)

Create `src/reachy_vec/body/sway.py`:

```python
"""Gentle sway loop while the robot is speaking (Mac-speaker fallback).

When replies play on the robot's own speaker, the SDK's audio-synced
wobble covers this (see make_robot). This module fakes life for the
Mac-speaker path: a background thread loops the 'sway' keyframes until
stopped. Motion is decoration - any body failure just ends the sway.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SpeakingSway:
    """Loops the 'sway' motion on a body between start() and stop()."""

    def __init__(self, body):
        self._body = body
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._body.perform("sway")
            except Exception as exc:
                logger.debug("sway ended on body error: %s", exc)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class SwayingSpeaker:
    """Speaker decorator: sway while each sentence plays; halt on barge-in."""

    def __init__(self, speaker, sway: SpeakingSway):
        self._speaker = speaker
        self._sway = sway

    def speak(self, text: str) -> None:
        self._sway.start()
        try:
            self._speaker.speak(text)
        finally:
            self._sway.stop()

    def stop(self) -> None:
        self._speaker.stop()
        self._sway.stop()
```

(`SwayingSpeaker` is tested in Task 3; it lives here because the two classes change together.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sway.py tests/test_body.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check src tests
uv run pytest -q
git add src/reachy_vec/body/motions.py src/reachy_vec/body/sway.py tests/test_sway.py tests/test_body.py
git commit -m "feat: SpeakingSway - gentle sway loop for the Mac-speaker path"
```

---

### Task 3: `SwayingSpeaker` behavior tests + `run.py` wiring

**Files:**
- Modify: `src/reachy_vec/cli/run.py` (after the `wrap_reconnect` call, line ~163)
- Test: `tests/test_sway.py`

**Interfaces:**
- Consumes: `SpeakingSway`, `SwayingSpeaker` from `reachy_vec.body.sway` (Task 2); `chosen` (`"robot"`/`"mac"`), `speaker`, and the wrapped `body` already in `run()`.
- Produces: nothing downstream; this completes the feature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sway.py`:

```python
from reachy_vec.body.sway import SwayingSpeaker


class RecordingSway:
    def __init__(self):
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


class ScriptedSpeaker:
    def __init__(self, boom=False):
        self.spoken: list[str] = []
        self.stopped = 0
        self._boom = boom

    def speak(self, text: str) -> None:
        if self._boom:
            raise RuntimeError("tts hiccup")
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped += 1


def test_swaying_speaker_brackets_each_sentence():
    sway, inner = RecordingSway(), ScriptedSpeaker()
    speaker = SwayingSpeaker(inner, sway)
    speaker.speak("hello")
    speaker.speak("world")
    assert inner.spoken == ["hello", "world"]
    assert sway.events == ["start", "stop", "start", "stop"]


def test_swaying_speaker_stops_sway_on_speak_error():
    sway = RecordingSway()
    speaker = SwayingSpeaker(ScriptedSpeaker(boom=True), sway)
    try:
        speaker.speak("hello")
    except RuntimeError:
        pass
    assert sway.events == ["start", "stop"]  # finally clause ran


def test_swaying_speaker_stop_halts_sway_too():
    sway, inner = RecordingSway(), ScriptedSpeaker()
    speaker = SwayingSpeaker(inner, sway)
    speaker.stop()  # barge-in path
    assert inner.stopped == 1
    assert sway.events == ["stop"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sway.py -q`
Expected: PASS already if Task 2 shipped `SwayingSpeaker` verbatim — these tests pin its behavior; if any fail, fix `SwayingSpeaker` (the spec behavior is what the tests say).

- [ ] **Step 3: Wire into run.py**

In `src/reachy_vec/cli/run.py`, directly after the `body = wrap_reconnect(...)` call (line ~163), add:

```python
    if chosen != "robot":
        # Robot-speaker replies get the SDK's audio-synced wobble (make_robot);
        # Mac-speaker replies get a gentle sway loop instead.
        from reachy_vec.body.sway import SpeakingSway, SwayingSpeaker

        speaker = SwayingSpeaker(speaker, SpeakingSway(body))
```

Reassigning the `speaker` local is the whole wiring: the `OracleLoop(...)` construction below reads it, so the loop (and the `enroll_person` closure at line ~193) get the wrapped speaker — enrollment speech sways too, which is desired. Note the placement matters: this must come AFTER `wrap_reconnect` so the sway drives the reconnect-protected body, and AFTER `make_barge_in_factory` uses nothing from `speaker` (it doesn't — it takes `chosen` and `media`).

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass, lint clean.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/cli/run.py tests/test_sway.py
git commit -m "feat: sway while speaking on the Mac-speaker path"
```

---

## Manual smoke test (after all tasks)

Robot on and daemon running (`http://reachy-mini.local:8000` shows Running):

```bash
# Mac speaker: head should sway gently exactly while sentences play
uv run reachy-vec run --preview --source mac

# Robot speaker: head should wobble in rhythm with the voice
REACHY_VEC_TTS_BACKEND=qwen-tts uv run reachy-vec run --preview --source robot

# knob off: robot still while talking again
REACHY_VEC_SPEECH_WOBBLE=false uv run reachy-vec run --preview --source robot
```

Also confirm barge-in still cuts speech AND motion together (talk over a long reply).

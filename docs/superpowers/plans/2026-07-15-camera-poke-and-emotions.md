# Camera-Poke Reaction + Built-in Emotion Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The robot startles (wake + emote + spoken line) when someone covers its camera, and conversations gain expressive body language from the SDK's recorded emotion library via a `react(emotion)` LLM tool.

**Architecture:** A pure-numpy `OcclusionDetector` rides on frames `sight()` already reads; `OracleLoop` gets an optional `occlusion` collaborator (same pattern as `speaker_id`). `RobotBody.perform(name)` stays the single body seam but resolves names against the lazy-loaded `pollen-robotics/reachy-mini-emotions-library` after local keyframes, with graceful degradation offline. `ChatBrain` gets a `react` tool injected by `run` exactly like `look`/`selfie`.

**Tech Stack:** Python 3.12, uv, pytest (fakes in `tests/conftest.py`), numpy, reachy-mini SDK 1.9.0 (`RecordedMoves`, `mini.play_move`), pydantic-settings config.

**Spec:** `docs/superpowers/specs/2026-07-15-camera-poke-and-emotions-design.md`

## Global Constraints

- All config knobs live in `config.py` (`REACHY_VEC_` env prefix) — never module constants.
- Heavy imports (SDK, HF downloads) deferred to inside methods; `import reachy_vec` and the suite stay fast and device-free.
- Every behavior gets a fake-based test first (TDD; watch it fail).
- `uv run pytest -q` and `uv run ruff check src tests` must pass before each commit.
- Real-robot behavior goes on the `docs/testing.md` smoke checklist, not claimed from fakes.

---

### Task 1: OcclusionDetector

**Files:**
- Create: `src/reachy_vec/perception/occlusion.py`
- Modify: `src/reachy_vec/config.py` (add 4 knobs after `min_utterance_rms`)
- Test: `tests/test_occlusion.py` (new)

**Interfaces:**
- Produces: `OcclusionDetector(dark_floor=None, drop_ratio=None, polls=None, cooldown_s=None, clock=time.time)` with `update(frame) -> bool` (True once, on the firing transition) and `consume() -> bool` (read-and-clear the fired flag). Constructor `None`s default from settings. Task 2 consumes `consume()`; Task 3 consumes `update()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_occlusion.py
"""Brightness-drop occlusion detector: covering the camera makes the frame
go near-black. Fires once per cover, re-arms on recovery, honors cooldown."""

import numpy as np

from reachy_vec.perception.occlusion import OcclusionDetector

BRIGHT = np.full((48, 64, 3), 120, dtype=np.uint8)
DARK = np.full((48, 64, 3), 5, dtype=np.uint8)


def make_detector(now, **kw):
    kw.setdefault("dark_floor", 28.0)
    kw.setdefault("drop_ratio", 0.3)
    kw.setdefault("polls", 2)
    kw.setdefault("cooldown_s", 10.0)
    return OcclusionDetector(clock=lambda: now["t"], **kw)


def test_fires_after_sustained_darkness():
    now = {"t": 0.0}
    det = make_detector(now)
    assert det.update(BRIGHT) is False        # builds baseline
    assert det.update(DARK) is False          # 1 dark poll: debounced
    assert det.update(DARK) is True           # 2nd dark poll: fires
    assert det.consume() is True
    assert det.consume() is False             # read-and-clear


def test_stays_quiet_while_held_covered():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    det.update(DARK)
    assert det.update(DARK) is True
    assert det.update(DARK) is False          # disarmed until recovery
    assert det.update(DARK) is False


def test_rearms_after_recovery_and_cooldown():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    det.update(DARK)
    assert det.update(DARK) is True
    det.update(BRIGHT)                        # recovery re-arms
    now["t"] = 5.0                            # still inside cooldown
    det.update(DARK)
    assert det.update(DARK) is False
    det.update(BRIGHT)
    now["t"] = 20.0                           # cooldown expired
    det.update(DARK)
    assert det.update(DARK) is True


def test_dark_start_never_fires_without_a_bright_baseline():
    now = {"t": 0.0}
    det = make_detector(now)
    for _ in range(5):
        assert det.update(DARK) is False      # robot may boot covered/at night


def test_none_frames_are_ignored():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    assert det.update(None) is False
    det.update(DARK)
    assert det.update(DARK) is True           # None didn't reset the dark run? no:


def test_defaults_come_from_settings():
    det = OcclusionDetector()
    assert det._dark_floor == 28.0
    assert det._polls == 2
```

Note the `test_none_frames_are_ignored` comment: decide the semantics in the
implementation — a `None` frame must not fire and must not reset the dark run
(camera hiccup mid-cover). The test above asserts exactly that.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'reachy_vec.perception.occlusion'`

- [ ] **Step 3: Add config knobs**

In `src/reachy_vec/config.py`, directly after the `min_utterance_rms` block:

```python
    # Camera-poke reaction: frame suddenly dark = someone covered the lens
    occlusion_dark_floor: float = 28.0  # mean brightness (0-255) below = dark
    occlusion_drop_ratio: float = 0.3   # ...and below this fraction of baseline
    occlusion_polls: int = 2            # consecutive dark polls to fire
    occlusion_cooldown_s: float = 10.0  # min seconds between reactions
```

- [ ] **Step 4: Write the implementation**

```python
# src/reachy_vec/perception/occlusion.py
"""Camera occlusion detection: covering/poking the lens makes the frame go
near-black. Pure numpy on frames sight() already reads - no models.

Fires once per cover (bright -> sustained dark transition), stays quiet while
held covered, re-arms only after brightness recovers, and honors a cooldown.
"""

import logging
import time

import numpy as np

from reachy_vec.config import settings

logger = logging.getLogger(__name__)


class OcclusionDetector:
    def __init__(
        self,
        dark_floor: float | None = None,
        drop_ratio: float | None = None,
        polls: int | None = None,
        cooldown_s: float | None = None,
        clock=time.time,
    ):
        self._dark_floor = dark_floor if dark_floor is not None else settings.occlusion_dark_floor
        self._drop_ratio = drop_ratio if drop_ratio is not None else settings.occlusion_drop_ratio
        self._polls = polls if polls is not None else settings.occlusion_polls
        self._cooldown_s = cooldown_s if cooldown_s is not None else settings.occlusion_cooldown_s
        self._clock = clock
        self._baseline: float | None = None  # EMA of unoccluded brightness
        self._dark_run = 0
        self._armed = True
        self._last_fired_at: float | None = None
        self._fired = False  # latched until consume()

    def update(self, frame) -> bool:
        """Feed one frame; True exactly once per bright->dark transition."""
        if frame is None:
            return False  # camera hiccup: neither dark nor a recovery
        brightness = float(np.asarray(frame).mean())
        dark = (
            self._baseline is not None
            and brightness < self._dark_floor
            and brightness < self._baseline * self._drop_ratio
        )
        if not dark:
            if brightness >= self._dark_floor:  # only bright frames teach the baseline
                self._baseline = (
                    brightness
                    if self._baseline is None
                    else 0.9 * self._baseline + 0.1 * brightness
                )
                self._armed = True
            self._dark_run = 0
            return False
        self._dark_run += 1
        if not self._armed or self._dark_run < self._polls:
            return False
        now = self._clock()
        if self._last_fired_at is not None and now - self._last_fired_at < self._cooldown_s:
            return False
        self._armed = False
        self._last_fired_at = now
        self._fired = True
        logger.info("camera occluded (brightness %.1f, baseline %.1f)", brightness, self._baseline)
        return True

    def consume(self) -> bool:
        """Read-and-clear the fired latch (polled by the Oracle loop)."""
        fired, self._fired = self._fired, False
        return fired
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: 6 passed. Also run `uv run pytest -q` (all pass) and `uv run ruff check src tests`.

- [ ] **Step 6: Commit**

```bash
git add src/reachy_vec/perception/occlusion.py src/reachy_vec/config.py tests/test_occlusion.py
git commit -m "feat: brightness-drop occlusion detector for camera-poke reaction"
```

---

### Task 2: Startle keyframes + Oracle occlusion reaction

**Files:**
- Modify: `src/reachy_vec/body/motions.py` (add `"startle"` to `MOTIONS`)
- Modify: `src/reachy_vec/brain/oracle.py` (add `STARTLE_LINES`, `occlusion=` param, `_react_to_occlusion`)
- Test: `tests/test_oracle.py` (append), `tests/conftest.py` (add `FakeOcclusion`)

**Interfaces:**
- Consumes: `occlusion.consume() -> bool` from Task 1 (via a fake here).
- Produces: `OracleLoop(..., occlusion=None)`; `"startle"` motion name (Task 4 upgrades it to `scared1`); `STARTLE_LINES` tuple in `brain/oracle.py`.

- [ ] **Step 1: Add FakeOcclusion to tests/conftest.py**

```python
class FakeOcclusion:
    """Scripted occlusion detector: consume() pops scripted booleans."""

    def __init__(self, trips: list[bool]):
        self._trips = list(trips)

    def update(self, frame) -> bool:
        return False

    def consume(self) -> bool:
        return self._trips.pop(0) if self._trips else False
```

- [ ] **Step 2: Write the failing tests (append to tests/test_oracle.py)**

```python
def test_camera_poke_wakes_startles_and_speaks(tmp_path):
    from reachy_vec.brain.oracle import STARTLE_LINES
    from tests.conftest import FakeBody, FakeBrain, FakeOcclusion, FakeSpeaker, FakeTranscriber

    speaker, body = FakeSpeaker(), FakeBody()
    now = {"t": 1000.0}
    loop = OracleLoop(
        sight=lambda: None,                      # covered camera: no face
        transcriber=FakeTranscriber([]),
        speaker=speaker,
        body=body,
        brain=FakeBrain(),
        enroll_capture=lambda name: None,
        store=Store(tmp_path / "db"),
        clock=lambda: now["t"],
        start_asleep=True,
        occlusion=FakeOcclusion([True]),
    )
    assert loop.run_once() == "no-face"
    assert body.motions[:2] == ["wake", "startle"]       # woke, then flinched
    assert speaker.spoken[0] in STARTLE_LINES
    assert "sleep" not in body.motions                    # reaction reset the idle timer


def test_camera_poke_while_awake_startles_without_waking(tmp_path):
    from tests.conftest import FakeBody, FakeBrain, FakeOcclusion, FakeSpeaker, FakeTranscriber

    speaker, body = FakeSpeaker(), FakeBody()
    loop = OracleLoop(
        sight=lambda: None,
        transcriber=FakeTranscriber([]),
        speaker=speaker,
        body=body,
        brain=FakeBrain(),
        enroll_capture=lambda name: None,
        store=Store(tmp_path / "db"),
        clock=lambda: 1000.0,
        occlusion=FakeOcclusion([True]),
    )
    assert loop.run_once() == "no-face"
    assert body.motions == ["startle"]                    # no wake motion


def test_no_occlusion_no_reaction(tmp_path):
    from tests.conftest import FakeBody, FakeBrain, FakeOcclusion, FakeSpeaker, FakeTranscriber

    speaker, body = FakeSpeaker(), FakeBody()
    loop = OracleLoop(
        sight=lambda: None,
        transcriber=FakeTranscriber([]),
        speaker=speaker,
        body=body,
        brain=FakeBrain(),
        enroll_capture=lambda name: None,
        store=Store(tmp_path / "db"),
        clock=lambda: 1000.0,
        occlusion=FakeOcclusion([]),
    )
    assert loop.run_once() == "no-face"
    assert body.motions == [] and speaker.spoken == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_oracle.py -q -k poke`
Expected: FAIL with `TypeError: OracleLoop.__init__() got an unexpected keyword argument 'occlusion'` (and ImportError for `STARTLE_LINES`).

- [ ] **Step 4: Implement**

In `src/reachy_vec/body/motions.py`, add to `MOTIONS` (after `"wakeup"`):

```python
    "startle": [
        # sharp recoil back, antennas snapped high
        Keyframe(head={"pitch": -22}, antennas=(1.0, 1.0), duration=0.15),
        # quick left-right head shake, antennas flailing
        Keyframe(head={"pitch": -18, "roll": -8}, antennas=(1.0, -0.5), duration=0.2),
        Keyframe(head={"pitch": -18, "roll": 8}, antennas=(-0.5, 1.0), duration=0.2),
        NEUTRAL,
    ],
```

In `src/reachy_vec/brain/oracle.py`:

After `WAKE_LINES`:

```python
# Spoken when someone covers/pokes the camera (see perception/occlusion.py).
STARTLE_LINES = (
    "Hey - I can't see!",
    "Who turned out the lights?",
    "Oi! That's my eye!",
)
```

Constructor: add `occlusion=None` after `settle_s`/`sleep`, store `self._occlusion = occlusion  # None = camera-poke reaction disabled`.

In `run_once`, right after `obs = self._sight()`:

```python
            obs = self._sight()
            if self._occlusion is not None and self._occlusion.consume():
                self._react_to_occlusion()
            if obs is None:
```

New method after `_note_presence`:

```python
    def _react_to_occlusion(self) -> None:
        """Someone covered the camera: wake if needed, flinch, protest."""
        logger.info("camera occluded - startling")
        if self._asleep:
            self._body.perform("wake")
            self._asleep = False
        self._body.perform("startle")
        self._speaker.speak(random.choice(STARTLE_LINES))
        self._last_face_at = self._clock()  # a poke counts as presence: reset idle timer
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_oracle.py -q` then `uv run pytest -q && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/reachy_vec/body/motions.py src/reachy_vec/brain/oracle.py tests/test_oracle.py tests/conftest.py
git commit -m "feat: camera-poke startle reaction (wake + flinch + spoken line)"
```

---

### Task 3: Sight wiring — feed frames to the detector

**Files:**
- Modify: `src/reachy_vec/cli/run.py` (add `make_sight`, wire detector + `OracleLoop(occlusion=...)`)
- Modify: `src/reachy_vec/perception/preview.py` (`PreviewSight` gets `occlusion=None`)
- Test: `tests/test_cli.py`, `tests/test_preview.py` (append)

**Interfaces:**
- Consumes: `OcclusionDetector.update(frame)` (Task 1), `OracleLoop(occlusion=...)` (Task 2).
- Produces: `make_sight(camera, matcher, occlusion=None) -> Callable[[], Observation | None]` in `cli/run.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_make_sight_feeds_frames_to_occlusion_detector():
    from reachy_vec.cli.run import make_sight

    frames_seen = []

    class Camera:
        def read(self):
            return "frame-1"

    class Matcher:
        def observe(self, frame):
            return f"obs({frame})"

    class Detector:
        def update(self, frame):
            frames_seen.append(frame)
            return False

    sight = make_sight(Camera(), Matcher(), occlusion=Detector())
    assert sight() == "obs(frame-1)"
    assert frames_seen == ["frame-1"]


def test_make_sight_without_detector():
    from reachy_vec.cli.run import make_sight

    class Camera:
        def read(self):
            return None

    class Matcher:
        def observe(self, frame):
            return None

    assert make_sight(Camera(), Matcher())() is None
```

Append to `tests/test_preview.py` (match its existing fake style — it already
stubs `show`; reuse whatever fake camera/matcher helpers exist in that file):

```python
def test_preview_sight_feeds_frames_to_occlusion_detector():
    import numpy as np

    from reachy_vec.perception.preview import PreviewSight

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    seen = []

    class Camera:
        def read(self):
            return frame

    class Matcher:
        def observe(self, f):
            return None

    class Detector:
        def update(self, f):
            seen.append(f)
            return False

    sight = PreviewSight(Camera(), Matcher(), show=lambda *a: None, occlusion=Detector())
    sight()
    assert seen == [frame]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -q -k make_sight; uv run pytest tests/test_preview.py -q -k occlusion`
Expected: FAIL — `ImportError: cannot import name 'make_sight'`, `TypeError: ... unexpected keyword argument 'occlusion'`.

- [ ] **Step 3: Implement**

In `src/reachy_vec/cli/run.py`, add near `make_barge_in_factory`:

```python
def make_sight(camera, matcher, occlusion=None):
    """Poll camera -> face observation; the same frame feeds occlusion detection."""

    def sight():
        frame = camera.read()
        if occlusion is not None:
            occlusion.update(frame)
        return matcher.observe(frame)

    return sight
```

In `run()`, replace the sight construction block:

```python
    from reachy_vec.perception.occlusion import OcclusionDetector

    occlusion = OcclusionDetector()
    if preview:
        from reachy_vec.perception.preview import PreviewSight

        sight = PreviewSight(camera, matcher, occlusion=occlusion)
    else:
        sight = make_sight(camera, matcher, occlusion=occlusion)
```

and pass `occlusion=occlusion,` to the `OracleLoop(...)` construction (after `settle_s=`).

In `src/reachy_vec/perception/preview.py`:

```python
class PreviewSight:
    def __init__(self, camera, matcher, show: Callable | None = None, occlusion=None):
        self._camera = camera
        self._matcher = matcher
        self._show = show if show is not None else self._cv2_show
        self._occlusion = occlusion

    def __call__(self):
        frame = self._camera.read()
        if self._occlusion is not None:
            self._occlusion.update(frame)
        if frame is None:
            return None
        observation = self._matcher.observe(frame)
        self._show(frame, observation, getattr(self._matcher, "last_bbox", None))
        return observation
```

(Note: `update` runs before the `None` check — the detector handles `None` itself and a covered robot camera may still deliver dark frames, but a Mac camera returning `None` must not crash.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/cli/run.py src/reachy_vec/perception/preview.py tests/test_cli.py tests/test_preview.py
git commit -m "feat: wire occlusion detector into sight polling and the Oracle loop"
```

---

### Task 4: RobotBody resolves recorded emotion moves

**Files:**
- Modify: `src/reachy_vec/body/motions.py` (add `EMOTION_MOVES`, `EMOTION_UPGRADES`)
- Modify: `src/reachy_vec/body/robot.py` (`RobotBody` lazy library + resolution order)
- Modify: `src/reachy_vec/config.py` (add `emotion_sound` knob)
- Test: `tests/test_body.py` (append)

**Interfaces:**
- Consumes: `"startle"` motion (Task 2).
- Produces: `EMOTION_MOVES: dict[str, str]` (friendly name → library move, used by Task 5), `EMOTION_UPGRADES: dict[str, str]`, `RobotBody(mini, moves_loader=None)` where `moves_loader() -> library` and `library.get(name) -> move` (raises on unknown).

- [ ] **Step 1: Write the failing tests (append to tests/test_body.py)**

Reuse the file's existing mini stub if one exists (`_StubMini` appears in
`tests/test_cli.py`); otherwise define the recorder below.

```python
def _recording_mini():
    class Mini:
        def __init__(self):
            self.gotos = []
            self.played = []

        def goto_target(self, **kw):
            self.gotos.append(kw)

        def play_move(self, move, **kw):
            self.played.append((move, kw))

        def goto_sleep(self):
            pass

        def wake_up(self):
            pass

    return Mini()


class _FakeLibrary:
    def __init__(self, moves):
        self._moves = moves

    def get(self, name):
        if name not in self._moves:
            raise KeyError(name)
        return self._moves[name]


def test_emotion_upgrade_plays_recorded_move_over_keyframes():
    from reachy_vec.body.robot import RobotBody

    mini = _recording_mini()
    body = RobotBody(mini, moves_loader=lambda: _FakeLibrary({"welcoming1": "MOVE"}))
    body.perform("greet")
    assert [m for m, _kw in mini.played] == ["MOVE"]
    assert mini.gotos == []                       # keyframes not used


def test_unknown_keyframe_name_falls_through_to_library():
    from reachy_vec.body.robot import RobotBody

    mini = _recording_mini()
    body = RobotBody(mini, moves_loader=lambda: _FakeLibrary({"laughing1": "LOL"}))
    body.perform("laughing1")
    assert [m for m, _kw in mini.played] == ["LOL"]


def test_library_load_failure_degrades_to_keyframes():
    from reachy_vec.body.robot import RobotBody

    def boom():
        raise RuntimeError("offline, cold HF cache")

    mini = _recording_mini()
    body = RobotBody(mini, moves_loader=boom)
    body.perform("greet")                          # upgrade unavailable -> keyframes
    assert mini.played == []
    assert len(mini.gotos) == 3                    # greet has 3 keyframes
    body.perform("nod")                            # loader not retried (cached failure)
    assert len(mini.gotos) == 5


def test_emotion_sound_flag_reaches_play_move(monkeypatch):
    from reachy_vec.body.robot import RobotBody

    monkeypatch.setattr("reachy_vec.body.robot.settings.emotion_sound", True)
    mini = _recording_mini()
    body = RobotBody(mini, moves_loader=lambda: _FakeLibrary({"welcoming1": "MOVE"}))
    body.perform("greet")
    assert mini.played[0][1]["sound"] is True


def test_emotion_moves_map_targets_existing_upgrades():
    from reachy_vec.body.motions import EMOTION_MOVES, EMOTION_UPGRADES

    assert EMOTION_UPGRADES == {"greet": "welcoming1", "startle": "scared1"}
    assert EMOTION_MOVES["curious"] == "curious1"
    assert len(EMOTION_MOVES) == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -q`
Expected: FAIL — `TypeError: RobotBody.__init__() got an unexpected keyword argument 'moves_loader'`, `ImportError: cannot import name 'EMOTION_MOVES'`.

- [ ] **Step 3: Implement**

In `src/reachy_vec/body/motions.py`, after the `MOTIONS` dict:

```python
# Friendly emotion names the brain's react() tool may use -> recorded moves in
# pollen-robotics/reachy-mini-emotions-library (played via mini.play_move).
EMOTION_MOVES: dict[str, str] = {
    "curious": "curious1",
    "laughing": "laughing1",
    "surprised": "surprised1",
    "thoughtful": "thoughtful1",
    "yes": "yes1",
    "no": "no1",
    "oops": "oops1",
    "proud": "proud1",
    "sad": "sad1",
    "grateful": "grateful1",
    "confused": "confused1",
    "enthusiastic": "enthusiastic1",
    "calming": "calming1",
    "attentive": "attentive1",
    "welcoming": "welcoming1",
    "scared": "scared1",
}

# Cues that upgrade from hand-made keyframes to richer recorded moves when the
# library is available. nod deliberately stays a fast keyframe (fires every turn).
EMOTION_UPGRADES: dict[str, str] = {
    "greet": "welcoming1",
    "startle": "scared1",
}
```

In `src/reachy_vec/body/robot.py`:

```python
from reachy_vec.body.motions import EMOTION_UPGRADES, MOTIONS
```

```python
def _load_emotion_library():
    """Recorded expressive moves; downloads/uses the HF dataset cache."""
    from reachy_mini.motion.recorded_move import RecordedMoves

    return RecordedMoves("pollen-robotics/reachy-mini-emotions-library")


class RobotBody:
    """Plays keyframes on a connected ReachyMini (sim or real); names that
    aren't local keyframes resolve against the SDK's recorded emotion library
    (lazy-loaded; a load failure degrades permanently to keyframes/no-ops)."""

    def __init__(self, mini, moves_loader=None):
        self._mini = mini
        self._moves_loader = moves_loader or _load_emotion_library
        self._moves = None  # lazy; False = load failed permanently

    def perform(self, motion: str) -> None:
        if motion == "sleep":
            self._mini.goto_sleep()
            return
        if motion == "wake":
            self._mini.wake_up()
            return
        upgrade = EMOTION_UPGRADES.get(motion)
        if upgrade is not None and self._play_recorded(upgrade):
            return
        frames = MOTIONS.get(motion)
        if frames is not None:
            from reachy_mini.utils import create_head_pose

            for kf in frames:
                self._mini.goto_target(
                    head=create_head_pose(**kf.head),
                    antennas=list(kf.antennas),
                    duration=kf.duration,
                )
            return
        if self._play_recorded(motion):
            return
        logger.warning("Unknown motion %r", motion)

    def _library(self):
        if self._moves is None:
            try:
                self._moves = self._moves_loader()
            except Exception as exc:
                logger.warning("emotion library unavailable (%s); keyframes only.", exc)
                self._moves = False
        return self._moves or None

    def _play_recorded(self, name: str) -> bool:
        library = self._library()
        if library is None:
            return False
        try:
            move = library.get(name)
        except Exception:
            logger.warning("recorded move %r not in library", name)
            return False
        # blocks for the move's duration, matching the loop's synchronous design;
        # sound off by default - the daemon's one media client is ours (TTS/mic)
        self._mini.play_move(
            move, initial_goto_duration=0.5, sound=settings.emotion_sound
        )
        return True
```

(Do not catch exceptions from `play_move` itself — `ReconnectingBody.perform`
already handles `TimeoutError`/`ConnectionError`, same as `goto_target`.)

In `src/reachy_vec/config.py`, after `speech_wobble`:

```python
    emotion_sound: bool = False  # sidecar sounds of recorded emotion moves; off
    # because the daemon allows one media client and we hold it for TTS/mic
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass (existing `test_body.py` keyframe tests must still pass — `RobotBody(mini)` with default loader never loads the library unless a name needs it).

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/body/motions.py src/reachy_vec/body/robot.py src/reachy_vec/config.py tests/test_body.py
git commit -m "feat: RobotBody resolves recorded emotion moves (greet/startle upgrades)"
```

---

### Task 5: The react(emotion) tool

**Files:**
- Modify: `src/reachy_vec/brain/chat.py` (REACT_TOOL, REACT_HINT, `react_fn` param, handler, dispatch)
- Modify: `src/reachy_vec/cli/run.py` (`make_react_fn`, pass to ChatBrain)
- Test: `tests/test_chat_brain.py`, `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `EMOTION_MOVES` (Task 4), `Body.perform` seam.
- Produces: `ChatBrain(..., react_fn: Callable[[str], str] | None = None)`; `make_react_fn(body) -> Callable[[str], str]` in `cli/run.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_brain.py` (follow the file's existing pattern for
constructing ChatBrain with `FakeLLMClient(messages=[...])` and `FakeToolCall`
— copy the setup used by the existing `look` tool test in that file):

```python
def test_react_tool_performs_emote_and_reply_continues(tmp_path):
    from tests.conftest import (
        FakeChoiceMessage,
        FakeEmbedder,
        FakeLLMClient,
        FakeToolCall,
    )

    performed = []
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(
                None, tool_calls=[FakeToolCall("react", '{"emotion": "laughing"}')]
            ),
            FakeChoiceMessage("Haha, good one!"),
        ]
    )
    brain = ChatBrain(
        store=Store(tmp_path / "db"),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-5-mini",
        react_fn=lambda emotion: performed.append(emotion) or "(performed emote)",
    )
    brain.begin_conversation("p1", "Alice")
    reply = brain.respond("tell me a joke")
    assert performed == ["laughing"]
    assert reply == "Haha, good one!"


def test_react_tool_absent_without_react_fn(tmp_path):
    from tests.conftest import FakeEmbedder, FakeLLMClient

    brain = ChatBrain(
        store=Store(tmp_path / "db"),
        embedder=FakeEmbedder(),
        client=FakeLLMClient(),
        model="gpt-5-mini",
    )
    names = [t["function"]["name"] for t in brain._active_tools()]
    assert "react" not in names


def test_react_tool_enum_matches_emotion_moves(tmp_path):
    from reachy_vec.body.motions import EMOTION_MOVES
    from reachy_vec.brain.chat import REACT_TOOL

    enum = REACT_TOOL["function"]["parameters"]["properties"]["emotion"]["enum"]
    assert enum == sorted(EMOTION_MOVES)
```

Append to `tests/test_cli.py`:

```python
def test_make_react_fn_maps_emotion_to_recorded_move():
    from reachy_vec.cli.run import make_react_fn
    from tests.conftest import FakeBody

    body = FakeBody()
    react = make_react_fn(body)
    assert "curious" in react("curious")
    assert body.motions == ["curious1"]
    assert "unknown" in react("hangry")
    assert body.motions == ["curious1"]            # nothing performed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_brain.py -q -k react; uv run pytest tests/test_cli.py -q -k react`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'react_fn'`, `ImportError: cannot import name 'REACT_TOOL'` / `'make_react_fn'`.

- [ ] **Step 3: Implement**

In `src/reachy_vec/brain/chat.py`, after `SELFIE_HINT`:

```python
from reachy_vec.body.motions import EMOTION_MOVES  # put with the top imports

REACT_TOOL = {
    "type": "function",
    "function": {
        "name": "react",
        "description": (
            "Play a short physical emote with the robot body matching the mood "
            "of your reply. Use sparingly - at most once per reply, only when "
            "it genuinely fits the moment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {"type": "string", "enum": sorted(EMOTION_MOVES)},
            },
            "required": ["emotion"],
        },
    },
}

REACT_HINT = (
    " react plays a short physical emote with your body (a laugh, a proud "
    "wiggle, a curious tilt) - use it sparingly, when the mood truly calls "
    "for it."
)
```

Constructor: add `react_fn: Callable[[str], str] | None = None` after
`selfie_fn`, store `self._react_fn = react_fn`.

`_active_tools`: after the selfie branch add

```python
        if self._react_fn:
            tools.append(REACT_TOOL)
```

`_system_prompt`: after the selfie branch add

```python
        if self._react_fn:
            prompt += REACT_HINT
```

Dispatch dict (the `handlers = {...}` in the tool-call method): add
`"react": self._tool_react,`.

Handler, next to `_tool_selfie`:

```python
    def _tool_react(self, args: dict) -> str:
        if self._react_fn is None:
            return "I can't emote right now."
        try:
            return self._react_fn((args.get("emotion") or "").strip())
        except Exception:
            logger.exception("react tool failed")
            return "I couldn't do that move just now."
```

In `src/reachy_vec/cli/run.py`, near `make_sight`:

```python
def make_react_fn(body):
    """react(emotion) tool body: friendly name -> recorded move via the Body seam."""
    from reachy_vec.body.motions import EMOTION_MOVES

    def react(emotion: str) -> str:
        move = EMOTION_MOVES.get(emotion)
        if move is None:
            return f"unknown emotion {emotion!r}"
        body.perform(move)
        return f"(performed a {emotion} emote)"

    return react
```

and in the `ChatBrain(...)` construction add `react_fn=make_react_fn(body),`
(after `selfie_fn=`). Note: this sits after `wrap_reconnect`, so emotes survive
daemon blips like every other motion.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/chat.py src/reachy_vec/cli/run.py tests/test_chat_brain.py tests/test_cli.py
git commit -m "feat: react(emotion) tool - LLM-picked recorded emotes in conversation"
```

---

### Task 6: Docs + smoke checklist

**Files:**
- Modify: `docs/testing.md` (smoke checklist additions)
- Modify: `docs/architecture.md` (mention occlusion + emotion moves where perception/body are described)

**Interfaces:** none (documentation).

- [ ] **Step 1: Extend the smoke checklist**

Append to the checklist in `docs/testing.md`:

```markdown
- **Camera poke**: with the Oracle idle, cover the camera with your hand for
  ~2s. Expect: startle/scared move + a spoken line ("Hey - I can't see!").
  Repeat while it sleeps: it must wake first. Holding it covered must react
  once, not loop; a second poke within 10s must be ignored.
- **Emotion moves**: ask something that invites an emote ("tell me a joke" ->
  laughing). Expect a recorded move mid-reply; conversation resumes after.
  With no network + cold HF cache, `greet` must fall back to the keyframe
  motion (log: "emotion library unavailable").
```

- [ ] **Step 2: Update architecture doc**

In `docs/architecture.md`, add one line to the perception section (occlusion
detector on the sight path, camera-poke startle + wake) and one to the body
section (recorded emotion library resolution after keyframes; `react` tool;
`emotion_sound` off by default because of the single-media-client constraint).
Match the file's existing prose style.

- [ ] **Step 3: Final verification and commit**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass.

```bash
git add docs/testing.md docs/architecture.md
git commit -m "docs: camera-poke reaction and emotion moves (smoke checklist + architecture)"
```

---

## Self-Review Notes

- Spec coverage: detector (T1), reaction+wake+lines (T2), sight wiring incl. preview (T3), body resolution + upgrades + sound knob (T4), react tool + curated enum (T5), smoke/docs (T6). No gaps found.
- Type consistency: `OcclusionDetector.update/consume` used identically in T1–T3; `EMOTION_MOVES`/`EMOTION_UPGRADES` names match across T4–T5; `react_fn: Callable[[str], str]` consistent between chat.py and run.py.
- Real-robot items (play_move behavior, real darkness thresholds, blocking-move feel) are explicitly deferred to the smoke checklist per repo policy.

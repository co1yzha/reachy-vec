# Look-Around Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the robot a `look_around` capability that rotates its base to face left then right, captures the camera at each, recenters, and reports its surroundings in one or two spoken sentences.

**Architecture:** Reuse the existing injected-closure pattern (`make_look_fn`). Add base-rotation "hold" motions driven by a new optional `Keyframe.body_yaw`, threaded through `RobotBody.perform` to the SDK's `goto_target(body_yaw=…)`. A new `make_look_around_fn` closure orchestrates turn→capture→turn→capture→recenter and makes one multi-image vision call. A `look_around` tool is injected into `ChatBrain` by `run`, exactly like `look`/`selfie`.

**Tech Stack:** Python 3.12, uv, pytest, pydantic-settings, openai SDK, reachy_mini SDK (`goto_target`), cv2 (deferred import).

## Global Constraints

- Every heavy dependency sits behind a small Protocol with a test fake; the whole flow must run in pytest with no devices/models/network. (CLAUDE.md)
- Heavy imports (cv2, reachy_mini) are deferred to inside functions/methods. (CLAUDE.md)
- New config knobs go in `config.py` as `REACHY_VEC_`-prefixed pydantic-settings fields, not module constants. (CLAUDE.md)
- New behavior gets a test against the fakes first (TDD). (CLAUDE.md)
- Base `body_yaw` joint range is ±2.79 rad (≈ ±160°); the sweep angle must stay well inside it. Use ≈2.09 rad (120°).
- The `Body` protocol stays `perform(motion: str) -> None` only — no new protocol methods. (design spec)
- Run `uv run ruff check src tests` and `uv run pytest -q` before finishing; both must pass. No CI. (CLAUDE.md)
- `cli/run.py` wiring is not unit-tested; it is verified via the `docs/testing.md` smoke checklist on the real robot. (CLAUDE.md)

---

### Task 1: Base-rotation motions (`body_yaw` keyframes + perform pass-through)

Adds `Keyframe.body_yaw`, three directional hold-motions (`turn_left`, `turn_right`, `turn_front`), and threads `body_yaw` through `RobotBody.perform`. Existing motions (no `body_yaw`) keep today's behavior: `RobotBody.perform` omits the `body_yaw` kwarg, so the SDK default (`0.0`, recenter-to-front) still applies as before.

**Files:**
- Modify: `src/reachy_vec/body/motions.py`
- Modify: `src/reachy_vec/body/robot.py:30-48` (`RobotBody.perform`)
- Test: `tests/test_body.py`

**Interfaces:**
- Produces: `Keyframe(head: dict, antennas: tuple, duration: float, body_yaw: float | None = None)`.
- Produces: module constant `LOOK_AROUND_YAW_RAD: float` in `body/motions.py`.
- Produces: `MOTIONS` gains keys `"turn_left"`, `"turn_right"`, `"turn_front"`.
- Produces (unchanged): `RobotBody.perform(motion: str) -> None`, now passing `body_yaw=kf.body_yaw` to `goto_target` only when `kf.body_yaw is not None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_body.py`)

```python
def test_keyframe_body_yaw_defaults_none():
    kf = Keyframe(head={}, antennas=(0.0, 0.0), duration=0.4)
    assert kf.body_yaw is None


def test_turn_motions_defined_and_hold_a_body_yaw():
    from reachy_vec.body.motions import LOOK_AROUND_YAW_RAD

    left = MOTIONS["turn_left"][-1]
    right = MOTIONS["turn_right"][-1]
    front = MOTIONS["turn_front"][-1]
    assert left.body_yaw == LOOK_AROUND_YAW_RAD          # turns one way...
    assert right.body_yaw == -LOOK_AROUND_YAW_RAD        # ...and the other
    assert front.body_yaw == 0.0                         # recenters
    assert 0 < LOOK_AROUND_YAW_RAD < 2.79                # inside the joint limit


def test_robot_body_passes_body_yaw_for_turn_motions():
    mini = RecordingMini()
    RobotBody(mini).perform("turn_left")
    assert mini.yaws == [MOTIONS["turn_left"][-1].body_yaw]


def test_robot_body_omits_body_yaw_for_plain_motions():
    mini = RecordingMini()
    RobotBody(mini).perform("nod")
    assert mini.yaws == [None]  # no body_yaw kwarg passed -> SDK default applies
```

Also update the existing `EXPECTED` set and `RecordingMini` in `tests/test_body.py`:

```python
EXPECTED = {
    "greet", "nod", "listen", "idle", "acknowledge", "goodbye", "look", "pose",
    "wakeup", "sway", "startle", "turn_left", "turn_right", "turn_front",
}
```

```python
class RecordingMini:
    def __init__(self):
        self.calls = []
        self.modes: list[str] = []
        self.yaws: list[float | None] = []

    def goto_target(self, head=None, antennas=None, duration=0.5, body_yaw=None):
        self.calls.append((head is not None, tuple(antennas), duration))
        self.yaws.append(body_yaw)

    def goto_sleep(self):
        self.modes.append("sleep")

    def wake_up(self):
        self.modes.append("wake")
```

> Note: the current `EXPECTED` set in the file is missing `"startle"` (a motion that already exists in `MOTIONS`), so `test_all_motions_defined_and_well_formed` is currently failing/stale. Adding `"startle"` alongside the three new keys fixes it — verify with the run in Step 2.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -q`
Expected: FAIL — `AttributeError: ... 'body_yaw'` / `KeyError: 'turn_left'` / `RecordingMini` has no `yaws` (depending on order); the new tests and `test_all_motions_defined_and_well_formed` do not pass yet.

- [ ] **Step 3: Add `body_yaw` to `Keyframe`, the constant, and the turn motions**

In `src/reachy_vec/body/motions.py`, update the dataclass:

```python
@dataclass(frozen=True)
class Keyframe:
    head: dict[str, float]
    antennas: tuple[float, float]
    duration: float
    body_yaw: float | None = None  # base rotation in radians; None = keep current yaw
```

Add the constant just below `NEUTRAL`:

```python
LOOK_AROUND_YAW_RAD = 2.09  # ~120 deg each side; joint limit is +/-2.79 rad
```

Add three entries to the `MOTIONS` dict (single keyframes that rotate and hold — no return-to-neutral frame):

```python
    "turn_left": [
        Keyframe(head={}, antennas=(0.5, 0.5), duration=0.9,
                 body_yaw=LOOK_AROUND_YAW_RAD),
    ],
    "turn_right": [
        Keyframe(head={}, antennas=(0.5, 0.5), duration=0.9,
                 body_yaw=-LOOK_AROUND_YAW_RAD),
    ],
    "turn_front": [
        Keyframe(head={}, antennas=(0.0, 0.0), duration=0.9, body_yaw=0.0),
    ],
```

- [ ] **Step 4: Thread `body_yaw` through `RobotBody.perform`**

In `src/reachy_vec/body/robot.py`, replace the keyframe loop in `perform` (currently lines 43-48):

```python
        for kf in frames:
            kwargs = dict(
                head=create_head_pose(**kf.head),
                antennas=list(kf.antennas),
                duration=kf.duration,
            )
            if kf.body_yaw is not None:
                kwargs["body_yaw"] = kf.body_yaw
            self._mini.goto_target(**kwargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -q`
Expected: PASS (all body tests, including `test_all_motions_defined_and_well_formed`).

- [ ] **Step 6: Commit**

```bash
git add src/reachy_vec/body/motions.py src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: base-rotation turn motions via Keyframe.body_yaw"
```

---

### Task 2: `look_around` orchestrator closure

Adds `make_look_around_fn` to `perception/vision.py`: turn left → settle → capture, turn right → settle → capture, recenter, then one multi-image vision call describing the surroundings. Best-effort throughout, mirroring `make_look_fn`.

**Files:**
- Modify: `src/reachy_vec/perception/vision.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: `encode_frame_jpeg(frame, max_px) -> str` and `VISION_SYSTEM` (existing in the module); `Body.perform("turn_left"|"turn_right"|"turn_front")` from Task 1.
- Produces: `make_look_around_fn(camera, client, model: str, max_px: int, body=None, *, settle_s: float = 0.8, sleep=time.sleep) -> Callable[[], str]`. The returned `look_around()` takes no args and returns a spoken string.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_vision.py`)

```python
from reachy_vec.perception.vision import make_look_around_fn


def _image_parts(client):
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    return [p for p in content if p["type"] == "image_url"]


def test_look_around_turns_both_ways_then_recenters_and_captures_twice():
    body = FakeBody()
    client = FakeLLMClient(reply="a desk to my left and a window to my right")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), client,
        model="gpt-5-mini", max_px=1024, body=body, sleep=lambda s: None,
    )
    assert look_around() == "a desk to my left and a window to my right"
    assert body.motions == ["turn_left", "turn_right", "turn_front"]
    assert len(client.chat.completions.calls) == 1  # exactly one vision call


def test_look_around_sends_both_images_labelled_left_and_right():
    client = FakeLLMClient(reply="ok")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), client,
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    look_around()
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert len(_image_parts(client)) == 2
    text = " ".join(p["text"].lower() for p in content if p["type"] == "text")
    assert "left" in text and "right" in text


def test_look_around_uses_injected_sleep_between_turns():
    slept = []
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), FakeLLMClient(reply="ok"),
        model="gpt-5-mini", max_px=1024, body=FakeBody(),
        settle_s=0.5, sleep=slept.append,
    )
    look_around()
    assert slept == [0.5, 0.5]  # settled after each turn, no real time.sleep


def test_look_around_no_body_captures_once_forward():
    client = FakeLLMClient(reply="a tidy room")
    look_around = make_look_around_fn(
        FakeCamera([_frame()]), client,
        model="gpt-5-mini", max_px=1024, body=None, sleep=lambda s: None,
    )
    assert look_around() == "a tidy room"
    assert len(_image_parts(client)) == 1  # single forward image


def test_look_around_drops_a_missing_direction():
    client = FakeLLMClient(reply="just the left side")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), None]), client,      # right frame missing
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    assert look_around() == "just the left side"
    assert len(_image_parts(client)) == 1


def test_look_around_no_frames_at_all_is_friendly():
    look_around = make_look_around_fn(
        FakeCamera([None, None]), FakeLLMClient(reply="x"),
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    assert "can't see" in look_around().lower()


def test_look_around_recenters_even_when_vision_call_fails():
    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("api down")

    body = FakeBody()
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), BoomClient(),
        model="gpt-5-mini", max_px=1024, body=body, sleep=lambda s: None,
    )
    assert "trouble" in look_around().lower()
    assert body.motions[-1] == "turn_front"  # recentered despite the failure
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vision.py -q`
Expected: FAIL with `ImportError: cannot import name 'make_look_around_fn'`.

- [ ] **Step 3: Implement `make_look_around_fn`**

Add to the top of `src/reachy_vec/perception/vision.py` (with the other stdlib imports):

```python
import time
```

Add this constant next to `DEFAULT_LOOK_PROMPT`:

```python
LOOK_AROUND_PROMPT = (
    "You turned your head to look around the room. Using these views, describe "
    "your surroundings in one or two short spoken sentences, noting roughly "
    "where things are (to your left, to your right)."
)
```

Add the closure (after `make_look_fn`):

```python
def make_look_around_fn(
    camera, client, model: str, max_px: int, body=None, *,
    settle_s: float = 0.8, sleep=time.sleep,
) -> Callable[[], str]:
    """Return a look_around() closure: turn the base left then right, capture a
    frame at each, recenter, and describe the surroundings in one vision call.
    With no `body`, degrades to a single forward capture. Best-effort: a failed
    turn or a missing frame never aborts the report.
    """

    def _turn(motion: str) -> None:
        if body is None:
            return
        try:
            body.perform(motion)
        except Exception:
            logger.exception("look_around turn %r failed", motion)

    def look_around() -> str:
        if body is None:
            views = [("ahead", camera.read())]
        else:
            views = []
            for label, motion in (("left", "turn_left"), ("right", "turn_right")):
                _turn(motion)
                sleep(settle_s)
                views.append((label, camera.read()))
            _turn("turn_front")  # recenter before the (possibly slow) vision call

        present = [(label, frame) for label, frame in views if frame is not None]
        if not present:
            return "I can't see anything right now - no camera frame."

        content = [{"type": "text", "text": LOOK_AROUND_PROMPT}]
        for label, frame in present:
            data_url = encode_frame_jpeg(frame, max_px)
            content.append({"type": "text", "text": f"To your {label}:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": content},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("look_around vision call failed")
            return "I had trouble seeing just now."
        logger.info("look_around -> %r", answer[:80])
        return answer or "I looked around but I'm not sure what I'm seeing."

    return look_around
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vision.py -q`
Expected: PASS (all vision tests).

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/perception/vision.py tests/test_vision.py
git commit -m "feat: make_look_around_fn turn-and-observe orchestrator"
```

---

### Task 3: `look_around` tool in `ChatBrain`

Registers the tool so the brain can call it, gated on the closure being injected (like `look`/`selfie`).

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Consumes: `make_look_around_fn(...) -> Callable[[], str]` from Task 2 (the brain only sees a zero-arg callable returning a string).
- Produces: `ChatBrain(..., look_around_fn: Callable[[], str] | None = None)`; the `look_around` tool appears in `_active_tools()` and dispatches to `_tool_look_around` only when `look_around_fn` is set.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_chat_brain.py`)

First extend the `make_brain` helper so tests can inject the closure:

```python
def make_brain(tmp_path, client, opener=None, web_search_fetch=None,
               look_around_fn=None):
    return ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        opener=opener or (lambda url: None),
        web_search_fetch=web_search_fetch,
        look_around_fn=look_around_fn,
    )
```

Then the tests:

```python
def test_look_around_tool_offered_only_when_fn_present(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = make_brain(tmp_path, FakeLLMClient(), look_around_fn=lambda: "a room")
    assert not any(t["function"]["name"] == "look_around" for t in off._active_tools())
    assert any(t["function"]["name"] == "look_around" for t in on._active_tools())


def test_look_around_tool_dispatches_to_fn(tmp_path):
    calls = []

    def fake_look_around():
        calls.append(True)
        return "a whiteboard on my left, two desks on my right"

    tool_call = FakeToolCall("look_around", json.dumps({}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Here's what I saw."),
        ]
    )
    brain = make_brain(tmp_path, client, look_around_fn=fake_look_around)
    reply = brain.respond("have a look around", identity=YANG)
    assert calls == [True]
    assert reply == "Here's what I saw."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_brain.py -k look_around -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'look_around_fn'`.

- [ ] **Step 3: Implement the tool**

In `src/reachy_vec/brain/chat.py`, add the schema and hint after `SELFIE_HINT` (≈ line 211):

```python
LOOK_AROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "look_around",
        "description": (
            "Physically turn to look around the room and report your "
            "surroundings. Use when the user asks you to look/turn around, what "
            "is around or behind you, or to check or scan the room."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

LOOK_AROUND_HINT = (
    " look_around turns your body left and right to see the whole room - use it "
    "when asked to look around, turn around, or say what's around or behind you."
)
```

Add the constructor parameter (after `selfie_fn` in `__init__`, ≈ line 296):

```python
        look_around_fn: Callable[[], str] | None = None,
```

Store it (next to `self._selfie_fn = selfie_fn`, ≈ line 318):

```python
        self._look_around_fn = look_around_fn
```

Register in `_active_tools` (after the `selfie_fn` branch, ≈ line 438):

```python
        if self._look_around_fn:
            tools.append(LOOK_AROUND_TOOL)
```

Register in `_system_prompt` (after the `selfie_fn` branch, ≈ line 448):

```python
        if self._look_around_fn:
            prompt += LOOK_AROUND_HINT
```

Add to the `handlers` dict in `_execute_tool` (after `"selfie": self._tool_selfie,`, ≈ line 511):

```python
            "look_around": self._tool_look_around,
```

Add the handler method (after `_tool_selfie`, ≈ line 589):

```python
    def _tool_look_around(self, args: dict) -> str:
        if self._look_around_fn is None:
            return "I can't look around right now."
        try:
            return self._look_around_fn()
        except Exception:
            logger.exception("look_around tool failed")
            return "I had trouble looking around just now."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_brain.py -q`
Expected: PASS (all chat-brain tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: look_around tool in ChatBrain"
```

---

### Task 4: Config knob + `run` wiring

Adds the `look_around_settle_s` config knob and wires `make_look_around_fn` into the `run` command so the tool is live on the robot.

**Files:**
- Modify: `src/reachy_vec/config.py:36-38` (Vision block)
- Modify: `src/reachy_vec/cli/run.py:154` (import), `:241-247` (build closure), `:255-264` (pass to `ChatBrain`)
- Test: `tests/test_config.py` (create if absent) or append to an existing config test

**Interfaces:**
- Consumes: `make_look_around_fn(...)` (Task 2), `ChatBrain(..., look_around_fn=...)` (Task 3), `settings.look_around_settle_s`.
- Produces: `Settings.look_around_settle_s: float = 0.8`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py` (or append if it exists):

```python
from reachy_vec.config import Settings


def test_look_around_settle_default():
    assert Settings().look_around_settle_s == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'look_around_settle_s'`.

- [ ] **Step 3: Add the config knob**

In `src/reachy_vec/config.py`, in the Vision block (after `vision_image_max_px`, ≈ line 38):

```python
    look_around_settle_s: float = 0.8  # pause after a base turn before capturing,
    # so the WebRTC frame catches up to the new heading (first frame lags ~2s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Wire into `run`**

In `src/reachy_vec/cli/run.py`, add `make_look_around_fn` to the vision import (line 154):

```python
    from reachy_vec.perception.vision import (
        make_look_around_fn,
        make_look_fn,
        make_selfie_fn,
    )
```

After the `look_fn = make_look_fn(...)` block (≈ line 241-247), add:

```python
    look_around_fn = make_look_around_fn(
        camera,
        client,
        model=settings.vision_model or settings.llm_model,
        max_px=settings.vision_image_max_px,
        body=body,
        settle_s=settings.look_around_settle_s,
    )
```

Then pass it into the `ChatBrain(...)` call (≈ line 255-264), after `selfie_fn=selfie_fn,`:

```python
        look_around_fn=look_around_fn,
```

- [ ] **Step 6: Run the full suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/reachy_vec/config.py src/reachy_vec/cli/run.py tests/test_config.py
git commit -m "feat: wire look_around into run + settle-time config knob"
```

- [ ] **Step 8: Manual smoke test (on the real robot / sim)**

This wiring is not unit-tested (per Global Constraints). Follow `docs/testing.md`; minimum check with the daemon up:
- `uv run reachy-vec run --preview` (or `--source robot`), then say "have a look around."
- Expect: base rotates left, pauses, right, pauses, recenters; robot speaks a one/two-sentence description mentioning left/right. If it rotates but the description is stale, raise `REACHY_VEC_LOOK_AROUND_SETTLE_S` (WebRTC first-frame lag).

---

## Notes for the implementer

- **Deferred imports:** `time` is stdlib and cheap — importing it at module top in `vision.py` is fine (the deferred-import rule targets torch/cv2/insightface/etc.). `cv2` stays deferred inside `encode_frame_jpeg` (already so).
- **Why `body_yaw` is not a config knob:** `MOTIONS` is a static module-level dict built at import; reaching into runtime settings there would be an import-time side effect. Retune the sweep via `LOOK_AROUND_YAW_RAD` in `body/motions.py`.
- **Ordering matters in tests:** `FakeCamera` serves frames in list order and returns `None` once exhausted; pass exactly as many frames as the test's capture count.
- **No new `Body` protocol method:** turning is expressed entirely through named motions + `perform(str)`, so `NullBody`, `ReconnectingBody`, and all fakes need no interface changes.

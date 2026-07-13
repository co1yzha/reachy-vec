# Camera Tools (look + selfie) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two camera tools to `ChatBrain` — `look(question?)` (answer questions about what the camera sees via an OpenAI vision sub-call) and `selfie()` (photograph the person in front of the robot, save it, and open it) — each with an expressive body gesture.

**Architecture:** Both tools are built as injected closures (`look_fn`, `selfie_fn`) constructed in `cli/run.py` where the camera, body, speaker, and OpenAI client are in scope, and passed into `ChatBrain`. All OpenCV/vision/file logic lives in a new `perception/vision.py`, keeping `ChatBrain` text-only and device-free. This mirrors the existing `web_search_fetch` injection pattern. When a closure is absent (text-only `chat` command, tests), its tool is simply not offered.

**Tech Stack:** Python 3.12, OpenAI SDK (chat completions, vision), OpenCV (`cv2`), pydantic-settings, pytest.

## Global Constraints

- Python 3.12+, uv-managed venv; run `uv run pytest -q` and `uv run ruff check src tests` before every commit (both must pass — no CI).
- Heavy imports (`cv2`) are deferred to inside functions, per the repo convention (keeps `import reachy_vec` and the test suite fast).
- Every heavy dependency sits behind a small function/closure with a test fake; new behavior gets a test against fakes first (TDD).
- New config knobs go in `config.py` (pydantic-settings, `REACHY_VEC_` env prefix), not as module constants.
- The brain model is `gpt-5-mini` (vision-capable); the design does NOT depend on the brain model being vision-capable — the vision call is a separate sub-call.
- Optional tools are advertised via a conditional hint string appended to the system prompt only when the tool is active (the established `WEB_SEARCH_HINT` pattern) — never via `PERSONALITY` (so the robot never claims a capability it lacks).
- `data/` is git-ignored and privacy-sensitive; never commit it.

---

### Task 1: Config knobs

**Files:**
- Modify: `src/reachy_vec/config.py`
- Test: `tests/test_config_camera.py` (create)

**Interfaces:**
- Produces: `settings.vision_model: str | None`, `settings.vision_image_max_px: int`, `settings.photos_dir: Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_camera.py`:

```python
from pathlib import Path

from reachy_vec.config import Settings


def test_vision_defaults():
    s = Settings()
    assert s.vision_model is None            # None => reuse llm_model
    assert s.vision_image_max_px == 1024
    assert s.photos_dir == Path("data") / "photos"


def test_vision_model_override_via_env(monkeypatch):
    monkeypatch.setenv("REACHY_VEC_VISION_MODEL", "gpt-4o")
    monkeypatch.setenv("REACHY_VEC_VISION_IMAGE_MAX_PX", "512")
    s = Settings()
    assert s.vision_model == "gpt-4o"
    assert s.vision_image_max_px == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_camera.py -v`
Expected: FAIL (`AttributeError` / no `vision_model`).

- [ ] **Step 3: Add the config fields**

In `src/reachy_vec/config.py`, under the `# Models` block (after the `tts_model` lines, before `voice_sample`), add:

```python
    # Vision (look() tool); None => reuse llm_model
    vision_model: str | None = None
    vision_image_max_px: int = 1024  # downscale long edge before encoding, to cut vision tokens
```

Then in the `# Storage` block, after the `faces_dir` property, add:

```python
    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_camera.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/config.py tests/test_config_camera.py
git commit -m "feat: add vision_model, vision_image_max_px, photos_dir config knobs"
```

---

### Task 2: `perception/vision.py` — frame encoding + `make_look_fn`

**Files:**
- Create: `src/reachy_vec/perception/vision.py`
- Test: `tests/test_vision.py` (create)

**Interfaces:**
- Consumes: a `camera` with `.read()` (returns a BGR `ndarray` or `None`), an OpenAI-shaped `client` with `client.chat.completions.create(...)`, an optional `body` with `.perform(motion: str)`.
- Produces:
  - `encode_frame_jpeg(frame, max_px: int) -> str` — returns a `data:image/jpeg;base64,...` URL.
  - `make_look_fn(camera, client, model: str, max_px: int, body=None) -> Callable[[str], str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vision.py`:

```python
import base64

import numpy as np

from reachy_vec.perception.vision import encode_frame_jpeg, make_look_fn
from tests.conftest import FakeBody, FakeCamera, FakeLLMClient


def _frame(h=10, w=20):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_encode_frame_jpeg_returns_data_url():
    url = encode_frame_jpeg(_frame(), max_px=1024)
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_encode_frame_jpeg_downscales_long_edge():
    import cv2

    url = encode_frame_jpeg(_frame(h=100, w=400), max_px=50)
    raw = base64.b64decode(url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 50


def test_look_fn_performs_gesture_then_answers():
    body = FakeBody()
    look = make_look_fn(
        FakeCamera([_frame()]), FakeLLMClient(reply="a tidy desk"),
        model="gpt-5-mini", max_px=1024, body=body,
    )
    assert look("what do you see?") == "a tidy desk"
    assert body.motions == ["look"]


def test_look_fn_sends_image_and_question():
    client = FakeLLMClient(reply="ok")
    look = make_look_fn(FakeCamera([_frame()]), client, model="gpt-5-mini", max_px=1024)
    look("count the people")
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    assert any(p["type"] == "text" and "count the people" in p["text"] for p in content)


def test_look_fn_empty_question_uses_default_prompt():
    client = FakeLLMClient(reply="ok")
    look = make_look_fn(FakeCamera([_frame()]), client, model="gpt-5-mini", max_px=1024)
    look("")
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert any(p["type"] == "text" and "Describe what you see" in p["text"] for p in content)


def test_look_fn_no_frame_is_friendly():
    look = make_look_fn(FakeCamera([None]), FakeLLMClient(reply="x"),
                        model="gpt-5-mini", max_px=1024)
    assert "can't see" in look("hi").lower()


def test_look_fn_gesture_failure_does_not_block():
    class BoomBody:
        def perform(self, motion):
            raise RuntimeError("wifi drop")

    look = make_look_fn(FakeCamera([_frame()]), FakeLLMClient(reply="a wall"),
                        model="gpt-5-mini", max_px=1024, body=BoomBody())
    assert look("what's there?") == "a wall"


def test_look_fn_vision_error_is_friendly():
    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("api down")

    look = make_look_fn(FakeCamera([_frame()]), BoomClient(),
                        model="gpt-5-mini", max_px=1024)
    assert "trouble" in look("hi").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vision.py -v`
Expected: FAIL (`ModuleNotFoundError: reachy_vec.perception.vision`).

- [ ] **Step 3: Write the implementation**

Create `src/reachy_vec/perception/vision.py`:

```python
"""Camera vision tools: encode a frame and answer questions about it (look),
built as injected closures so ChatBrain stays text-only and device-free.
Heavy imports (cv2) are deferred per repo convention.
"""

import base64
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

VISION_SYSTEM = (
    "You are the robot's eyes. Answer in one or two short spoken sentences. "
    "No markdown, no lists."
)
DEFAULT_LOOK_PROMPT = "Describe what you see."


def encode_frame_jpeg(frame, max_px: int) -> str:
    """BGR ndarray -> long-edge-downscaled JPEG -> base64 data URL."""
    import cv2

    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge > max_px:
        scale = max_px / long_edge
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG encode failed")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_look_fn(camera, client, model: str, max_px: int, body=None) -> Callable[[str], str]:
    """Return a look(question) closure over the camera + OpenAI client.
    If `body` is given, perform the 'look' gesture before capturing (best-effort).
    """

    def look(question: str) -> str:
        prompt = (question or "").strip() or DEFAULT_LOOK_PROMPT
        if body is not None:
            try:
                body.perform("look")
            except Exception:
                logger.exception("look gesture failed; capturing anyway")
        frame = camera.read()
        if frame is None:
            return "I can't see anything right now - no camera frame."
        try:
            data_url = encode_frame_jpeg(frame, max_px)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("look vision call failed")
            return "I had trouble seeing just now."
        logger.info("look(%r) -> %r", prompt, answer[:80])
        return answer or "I looked but I'm not sure what I'm seeing."

    return look
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vision.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/perception/vision.py tests/test_vision.py
git commit -m "feat: perception/vision.py with encode_frame_jpeg and make_look_fn"
```

---

### Task 3: `ChatBrain` — wire the `look()` tool

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py` (add tests)

**Interfaces:**
- Consumes: a `look_fn: Callable[[str], str]` (from Task 2).
- Produces: `ChatBrain(..., look_fn=...)`; a `look` tool in `_active_tools()`; `_tool_look(args)` handler.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_chat_brain.py`:

```python
def test_look_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=lambda q: "a desk",
    )
    assert not any(t["function"]["name"] == "look" for t in off._active_tools())
    assert any(t["function"]["name"] == "look" for t in on._active_tools())
    assert "camera" in on._system_prompt().lower()


def test_look_tool_passes_question_and_returns_answer(tmp_path):
    seen = []
    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=lambda q: seen.append(q) or "two monitors",
    )
    assert brain._tool_look({"question": "what's on my desk?"}) == "two monitors"
    assert seen == ["what's on my desk?"]


def test_look_tool_failure_is_friendly(tmp_path):
    def boom(q):
        raise RuntimeError("cam died")

    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=boom,
    )
    assert "trouble" in brain._tool_look({"question": "x"}).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_brain.py -k look -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'look_fn'`).

- [ ] **Step 3: Write the implementation**

In `src/reachy_vec/brain/chat.py`:

(a) After the `WEB_SEARCH_HINT = (...)` block, add the tool + hint:

```python
LOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "look",
        "description": (
            "Look through the robot's camera and answer about what's physically "
            "in view right now (surroundings, objects, how many people, or to "
            "read visible text)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "what to look for or answer about the scene; omit to describe the view",
                },
            },
        },
    },
}

LOOK_HINT = (
    " look lets you actually see through your camera - use it when the user asks "
    "about your physical surroundings, to read something in view, or to count "
    "people; describe only what you can see."
)
```

(b) In `ChatBrain.__init__`, add the parameter (after `web_search_fetch=None`):

```python
        web_search_fetch=None,
        look_fn: Callable[[str], str] | None = None,
```

and store it (near `self._web_search_fetch = web_search_fetch`):

```python
        self._look_fn = look_fn
```

(c) Replace `_active_tools` and `_system_prompt` with list-building versions:

```python
    def _active_tools(self) -> list:
        tools = list(TOOLS)
        if self._web_search_fetch:
            tools.append(WEB_SEARCH_TOOL)
        if self._look_fn:
            tools.append(LOOK_TOOL)
        return tools

    def _system_prompt(self) -> str:
        prompt = PERSONALITY
        if self._web_search_fetch:
            prompt += WEB_SEARCH_HINT
        if self._look_fn:
            prompt += LOOK_HINT
        return prompt
```

(d) In `_execute_tool`, add to the `handlers` dict:

```python
            "look": self._tool_look,
```

(e) Add the handler (near `_tool_web_search`):

```python
    def _tool_look(self, args: dict) -> str:
        if self._look_fn is None:
            return "I can't see right now."
        question = args.get("question", "").strip()
        try:
            return self._look_fn(question)
        except Exception:
            logger.exception("look tool failed")
            return "I had trouble seeing just now."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat_brain.py -v`
Expected: PASS (all existing + 3 new).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: wire look() tool into ChatBrain"
```

---

### Task 4: Body motions — add `"look"` and `"pose"` keyframes

**Files:**
- Modify: `src/reachy_vec/body/motions.py`
- Test: `tests/test_body.py` (add a test)

**Interfaces:**
- Produces: `MOTIONS["look"]` and `MOTIONS["pose"]` (each a `list[Keyframe]`), invoked via the existing `Body.perform(name)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_body.py`:

```python
def test_look_and_pose_motions_exist_and_are_valid():
    from reachy_vec.body.motions import MOTIONS, Keyframe

    for name in ("look", "pose"):
        frames = MOTIONS[name]
        assert frames and all(isinstance(kf, Keyframe) for kf in frames)
        assert all(kf.duration > 0 for kf in frames)
        assert frames[-1].head == {} and frames[-1].antennas == (0.0, 0.0)  # ends neutral
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_body.py::test_look_and_pose_motions_exist_and_are_valid -v`
Expected: FAIL (`KeyError: 'look'`).

- [ ] **Step 3: Add the motions**

In `src/reachy_vec/body/motions.py`, add two entries to the `MOTIONS` dict (before the closing `}`):

```python
    "look": [
        Keyframe(head={"pitch": -6, "yaw": 8}, antennas=(0.4, 0.4), duration=0.35),
        Keyframe(head={"pitch": -6, "yaw": -8}, antennas=(0.4, 0.4), duration=0.35),
        NEUTRAL,
    ],
    "pose": [
        Keyframe(head={"pitch": -10}, antennas=(0.7, 0.7), duration=0.4),
        Keyframe(head={"pitch": -10, "roll": 4}, antennas=(0.7, 0.7), duration=0.4),
        NEUTRAL,
    ],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_body.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/body/motions.py tests/test_body.py
git commit -m "feat: add 'look' and 'pose' expressive motions"
```

---

### Task 5: `perception/vision.py` — `make_selfie_fn`

**Files:**
- Modify: `src/reachy_vec/perception/vision.py`
- Test: `tests/test_vision.py` (add tests)

**Interfaces:**
- Consumes: `camera.read()`, a `photos_dir: Path`, optional `body.perform`, optional `speak(text)`, optional `opener(path)`.
- Produces: `make_selfie_fn(camera, photos_dir, *, body=None, speak=None, opener=default_opener) -> Callable[[], str]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vision.py`:

```python
from reachy_vec.perception.vision import make_selfie_fn


def test_selfie_saves_file_and_opens_it(tmp_path):
    opened = []
    selfie = make_selfie_fn(
        FakeCamera([_frame()]), tmp_path / "photos",
        body=FakeBody(), speak=lambda t: None, opener=opened.append,
    )
    result = selfie()
    saved = list((tmp_path / "photos").glob("*.jpg"))
    assert len(saved) == 1
    assert opened == [str(saved[0])]
    assert "photo" in result.lower()


def test_selfie_ceremony_order(tmp_path):
    events = []

    class RecCamera:
        def read(self):
            events.append("capture")
            return _frame()

    class RecBody:
        def perform(self, motion):
            events.append(f"motion:{motion}")

    selfie = make_selfie_fn(
        RecCamera(), tmp_path / "photos",
        body=RecBody(), speak=lambda t: events.append("speak"),
        opener=lambda p: events.append("open"),
    )
    selfie()
    assert events == ["speak", "motion:pose", "capture", "open"]


def test_selfie_no_frame_writes_nothing(tmp_path):
    selfie = make_selfie_fn(FakeCamera([None]), tmp_path / "photos",
                            body=FakeBody(), speak=lambda t: None, opener=lambda p: None)
    assert "couldn't take" in selfie().lower()
    assert not (tmp_path / "photos").exists() or not list((tmp_path / "photos").glob("*.jpg"))


def test_selfie_best_effort_failures_still_save(tmp_path):
    opened = []

    def boom(*a):
        raise RuntimeError("hardware")

    selfie = make_selfie_fn(
        FakeCamera([_frame()]), tmp_path / "photos",
        body=type("B", (), {"perform": boom})(), speak=boom, opener=opened.append,
    )
    result = selfie()
    assert list((tmp_path / "photos").glob("*.jpg"))  # saved despite speak+pose failing
    assert "photo" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vision.py -k selfie -v`
Expected: FAIL (`ImportError: cannot import name 'make_selfie_fn'`).

- [ ] **Step 3: Write the implementation**

In `src/reachy_vec/perception/vision.py`, add the import at the top (with the other imports):

```python
from datetime import datetime
```

and, at the bottom, the factory:

```python
def make_selfie_fn(
    camera, photos_dir, *, body=None, speak=None, opener=None
) -> Callable[[], str]:
    """Return a selfie() closure: say 'Smile!', pose, capture, save, and open.
    speak/body/opener are all best-effort; a failure in any never blocks the save.
    """
    from reachy_vec.brain.chat import default_opener

    open_file = opener if opener is not None else default_opener

    def selfie() -> str:
        if speak is not None:
            try:
                speak("Smile!")
            except Exception:
                logger.exception("selfie 'Smile!' failed; continuing")
        if body is not None:
            try:
                body.perform("pose")
            except Exception:
                logger.exception("selfie pose failed; continuing")
        frame = camera.read()
        if frame is None:
            return "I couldn't take the photo - no camera frame."
        try:
            import cv2

            photos_dir.mkdir(parents=True, exist_ok=True)
            path = photos_dir / f"{datetime.now():%Y-%m-%d-%H%M%S}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError("imwrite returned False")
        except Exception:
            logger.exception("selfie save failed")
            return "I couldn't save the photo just now."
        try:
            open_file(str(path))
        except Exception:
            logger.exception("selfie open failed; photo is still saved")
        logger.info("selfie -> %s", path)
        return "took a photo and popped it up"

    return selfie
```

Note: `opener=None` (resolved to `default_opener` inside) rather than a default in the signature, to avoid importing `default_opener` at module load (deferred import keeps `perception` free of a `brain` dependency at import time).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vision.py -v`
Expected: PASS (all look + selfie tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/perception/vision.py tests/test_vision.py
git commit -m "feat: add make_selfie_fn (say cheese, pose, capture, save, show)"
```

---

### Task 6: `ChatBrain` — wire the `selfie()` tool

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py` (add tests)

**Interfaces:**
- Consumes: a `selfie_fn: Callable[[], str]` (from Task 5).
- Produces: `ChatBrain(..., selfie_fn=...)`; a `selfie` tool in `_active_tools()`; `_tool_selfie(args)` handler.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_chat_brain.py`:

```python
def test_selfie_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=lambda: "took a photo and popped it up",
    )
    assert not any(t["function"]["name"] == "selfie" for t in off._active_tools())
    assert any(t["function"]["name"] == "selfie" for t in on._active_tools())


def test_selfie_tool_invokes_closure(tmp_path):
    calls = []
    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=lambda: calls.append(1) or "took a photo and popped it up",
    )
    assert brain._tool_selfie({}) == "took a photo and popped it up"
    assert calls == [1]


def test_selfie_tool_failure_is_friendly(tmp_path):
    def boom():
        raise RuntimeError("cam died")

    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=boom,
    )
    assert "couldn't take" in brain._tool_selfie({}).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_brain.py -k selfie -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'selfie_fn'`).

- [ ] **Step 3: Write the implementation**

In `src/reachy_vec/brain/chat.py`:

(a) After the `LOOK_HINT` block (Task 3), add:

```python
SELFIE_TOOL = {
    "type": "function",
    "function": {
        "name": "selfie",
        "description": (
            "Take a photo of the person(s) in front of you and show it to them. "
            "Use when asked for a photo, selfie, or picture."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

SELFIE_HINT = (
    " selfie snaps a photo through your camera and pops it up on screen - use it "
    "when the user asks you to take a photo, selfie, or picture of them."
)
```

(b) In `ChatBrain.__init__`, add the parameter (after `look_fn=...`):

```python
        look_fn: Callable[[str], str] | None = None,
        selfie_fn: Callable[[], str] | None = None,
```

and store it (after `self._look_fn = look_fn`):

```python
        self._selfie_fn = selfie_fn
```

(c) Extend `_active_tools` and `_system_prompt`:

```python
    def _active_tools(self) -> list:
        tools = list(TOOLS)
        if self._web_search_fetch:
            tools.append(WEB_SEARCH_TOOL)
        if self._look_fn:
            tools.append(LOOK_TOOL)
        if self._selfie_fn:
            tools.append(SELFIE_TOOL)
        return tools

    def _system_prompt(self) -> str:
        prompt = PERSONALITY
        if self._web_search_fetch:
            prompt += WEB_SEARCH_HINT
        if self._look_fn:
            prompt += LOOK_HINT
        if self._selfie_fn:
            prompt += SELFIE_HINT
        return prompt
```

(d) In `_execute_tool`, add to the `handlers` dict:

```python
            "selfie": self._tool_selfie,
```

(e) Add the handler (after `_tool_look`):

```python
    def _tool_selfie(self, args: dict) -> str:
        if self._selfie_fn is None:
            return "I can't take a photo right now."
        try:
            return self._selfie_fn()
        except Exception:
            logger.exception("selfie tool failed")
            return "I couldn't take the photo just now."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat_brain.py -v`
Expected: PASS (all existing + 3 new).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: wire selfie() tool into ChatBrain"
```

---

### Task 7: Wire both closures in `cli/run.py` + docs

**Files:**
- Modify: `src/reachy_vec/cli/run.py:150-186`
- Modify: `docs/architecture.md` (add a one-line privacy note about `data/photos/`)

**Interfaces:**
- Consumes: `make_look_fn`, `make_selfie_fn` (Tasks 2, 5), `default_opener` (`brain/chat.py`), `settings.vision_model`, `settings.vision_image_max_px`, `settings.photos_dir` (Task 1).
- Produces: a `ChatBrain` constructed with `look_fn` and `selfie_fn` in the `run` command.

- [ ] **Step 1: Add imports**

In the import block inside `run()` (around `run.py:81-87`), add:

```python
    from reachy_vec.brain.chat import ChatBrain, default_opener
    from reachy_vec.perception.vision import make_look_fn, make_selfie_fn
```

(Replace the existing `from reachy_vec.brain.chat import ChatBrain` line with the first line above.)

- [ ] **Step 2: Reorder `wrap_reconnect` above `ChatBrain` and build the closures**

Replace the current block (`run.py:157-169`), which is:

```python
    brain = ChatBrain(
        store=store,
        embedder=embedder,
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
    )
    body = wrap_reconnect(
        body,
        connect_body=lambda: make_robot(with_media=False)[0],
        announce=speaker.speak,
    )
```

with (note `wrap_reconnect` now comes first, so `look_fn`/`selfie_fn` close over the resilient body):

```python
    body = wrap_reconnect(
        body,
        connect_body=lambda: make_robot(with_media=False)[0],
        announce=speaker.speak,
    )
    look_fn = make_look_fn(
        camera,
        client,
        model=settings.vision_model or settings.llm_model,
        max_px=settings.vision_image_max_px,
        body=body,
    )
    selfie_fn = make_selfie_fn(
        camera,
        settings.photos_dir,
        body=body,
        speak=speaker.speak,
        opener=default_opener,
    )
    brain = ChatBrain(
        store=store,
        embedder=embedder,
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
        look_fn=look_fn,
        selfie_fn=selfie_fn,
    )
```

- [ ] **Step 3: Add the docs privacy note**

In `docs/architecture.md`, find the data-layer / privacy section that mentions `data/reachy.log` and add one line:

```markdown
- `selfie()` writes photos to `data/photos/` — pictures of people; git-ignored, never committed (same privacy stance as `data/reachy.log`).
```

- [ ] **Step 4: Verify — lint, import, full suite**

Run:

```bash
uv run ruff check src tests
uv run python -c "import reachy_vec.cli.run; import reachy_vec.perception.vision"
uv run pytest -q
```

Expected: ruff clean; import OK (no error); full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/cli/run.py docs/architecture.md
git commit -m "feat: wire look() and selfie() into the run command"
```

---

## Self-Review

**Spec coverage:**
- `look()` delegate-to-vision-sub-call → Task 2 (`make_look_fn`) + Task 3 (tool wiring). ✓
- `vision_model` reuse of `llm_model`, `vision_image_max_px` → Task 1 + Task 7 wiring. ✓
- `"look"` motion, best-effort, before capture → Task 4 + Task 2 (`body.perform("look")`). ✓
- `selfie()` save + show, Smile! + pose, plain file → Task 5 (`make_selfie_fn`) + Task 6 (tool wiring). ✓
- `"pose"` motion → Task 4. ✓
- `photos_dir` config → Task 1; privacy note → Task 7. ✓
- `run`-only availability, absent in `chat`/tests → closures only built in `run.py` (Task 7); tools gated on presence (Tasks 3, 6). ✓
- Error handling (no frame, vision error, motion/speak/opener best-effort) → Tasks 2, 5 impl + tests. ✓
- `wrap_reconnect` reorder → Task 7 Step 2. ✓
- Testing against fakes, no network/devices → every task uses `FakeCamera`/`FakeBody`/`FakeLLMClient` or real numpy+cv2 (local). ✓

**Deviation from spec (intentional, noted):** optional-tool guidance lives in conditional `LOOK_HINT`/`SELFIE_HINT` appended to the system prompt only when the tool is active, NOT in `PERSONALITY`. This follows the established `WEB_SEARCH_HINT` pattern and prevents the robot from advertising a capability it lacks (e.g. in the text-only `chat` command). Recorded in Global Constraints.

**Placeholder scan:** none — every code step contains complete code.

**Type consistency:** `make_look_fn(camera, client, model, max_px, body=None)` and `make_selfie_fn(camera, photos_dir, *, body=None, speak=None, opener=None)` are used identically in Task 7. `look_fn: Callable[[str], str]` and `selfie_fn: Callable[[], str]` match between Tasks 3/6 and Task 7. Tool names `"look"`/`"selfie"` consistent across handler dict, tool defs, and tests. ✓

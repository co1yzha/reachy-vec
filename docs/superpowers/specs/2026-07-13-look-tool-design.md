# `look()` tool — design

**Date:** 2026-07-13
**Status:** Approved, ready for implementation planning

## Summary

Add a `look()` tool to `ChatBrain` so the robot can answer questions about what
its camera currently sees: "what's on my desk?", "how many people are here?",
"read this whiteboard". The brain calls `look(question?)`; a handler grabs the
current camera frame, sends it to an OpenAI vision model in an isolated call, and
returns a short spoken-ready text answer that flows back through the normal
tool-calling loop.

This is the highest-value, lowest-effort use of the Reachy camera for a team
assistant — far more useful than 3D reconstruction — and it reuses infrastructure
already in place (OpenAI client, the camera, the text-in/text-out tool pattern).

## Hardware / context

Reachy Mini has **one wide-angle RGB camera** (Raspberry Pi Camera v3 / IMX708,
120° FOV, autofocus) — no depth, no stereo. `Camera.read()` returns a single BGR
`ndarray` (`perception/camera.py`). The brain model is `gpt-5-mini`, which is
vision-capable, but the design deliberately does **not** depend on that (see
below).

## Approach

**Chosen: delegate to a vision sub-call.** The `look()` handler makes its own
OpenAI vision call (frame + question) and returns text — exactly like
`get_weather` and `web_search` delegate to an external source and return a
string. This is the pattern already used throughout `brain/chat.py`.

**Rejected alternative: inject the image into the brain's own message history**
(so `gpt-5-mini` sees it directly). Rejected because:

- The image would land in `_history` and be re-sent on every subsequent turn
  until trimmed — significant, wasteful token cost.
- It requires reshaping tool-result messages into multimodal `image_url` content
  and special-casing history handling.
- It locks the brain model to being vision-capable forever.

Delegation keeps `ChatBrain` text-only and trivially testable, sends the image
exactly once, and lets the vision model be swapped independently.

## Components

### 1. New module: `perception/vision.py`

Keeps OpenCV / image-encoding logic out of `ChatBrain`.

```python
def encode_frame_jpeg(frame, max_px: int) -> str:
    """BGR ndarray -> long-edge-downscaled JPEG -> base64 data URL."""

def make_look_fn(camera, client, model: str, max_px: int) -> Callable[[str], str]:
    """Return a look(question) closure over the camera + OpenAI client."""
```

`make_look_fn` returns a `look(question: str) -> str` closure that:

1. `frame = camera.read()`. If `None`, return
   `"I can't see anything right now — no camera frame."`
2. Downscale the long edge to `max_px`, JPEG-encode, base64-encode into a
   `data:image/jpeg;base64,...` URL. OpenCV import is deferred (module-level
   convention). `cv2.imencode` treats the input as BGR and produces a correct
   JPEG, so no manual BGR→RGB conversion is needed.
3. One **non-streaming** `client.chat.completions.create` call with:
   - a terse system prompt: *"You are the robot's eyes. Answer in one or two
     short spoken sentences. No markdown, no lists."*
   - a user message containing the image (`image_url` content part) and the
     question text (default `"Describe what you see."` when the question is
     empty).
4. Return the answer text. Any exception is logged and a friendly fallback
   string is returned (e.g. `"I had trouble seeing just now."`).

### 2. `ChatBrain` changes (`brain/chat.py`)

- `__init__` gains `look_fn: Callable[[str], str] | None = None`, stored as
  `self._look_fn`. Mirrors `web_search_fetch`.
- Add a `LOOK_TOOL` definition: function name `look`, one optional parameter
  `question: string` ("what to look for or answer about the scene; omit to
  describe what's in view").
- Add a `LOOK_HINT` string appended to the system prompt when look is enabled —
  tells the robot it can actually see through its camera and to use `look()` when
  asked about its physical surroundings, to read something, or to count people.
- `_active_tools()` appends `LOOK_TOOL` when `self._look_fn` is set.
- `_system_prompt()` appends `LOOK_HINT` when `self._look_fn` is set.
- The `PERSONALITY` tools sentence gains a short clause describing `look`.
- `_execute_tool`'s handler dict gains `"look": self._tool_look`.
- `_tool_look(args)` reads `question = args.get("question", "").strip()`, calls
  `self._look_fn(question)` inside a try/except that logs and returns a friendly
  string on failure.

### 3. Config (`config.py`)

- `vision_model: str | None = None` (env `REACHY_VEC_VISION_MODEL`). `None` means
  **reuse `llm_model`** (gpt-5-mini). Optional override only — nothing extra to
  configure by default.
- `vision_image_max_px: int = 1024` (env `REACHY_VEC_VISION_IMAGE_MAX_PX`).
  Downscale the long edge of the 12MP / 120° frame before encoding, to cut vision
  token cost.

### 4. Wiring (`cli/run.py`)

After `client` and `camera` are constructed and before/at `ChatBrain`
construction (`run.py:157`):

```python
from reachy_vec.perception.vision import make_look_fn

look_fn = make_look_fn(
    camera,
    client,
    model=settings.vision_model or settings.llm_model,
    max_px=settings.vision_image_max_px,
)
brain = ChatBrain(..., look_fn=look_fn)
```

The tool is available **only in `run`** (where a camera exists). In the text-only
`chat` command and in tests, `look_fn` is absent, so the tool is simply not
offered and never called.

## Data flow

```
brain streams a turn
  → emits tool_call look(question)
  → _tool_look → look_fn(question)
      → camera.read() → BGR frame
      → downscale + JPEG + base64 data URL
      → OpenAI vision call (frame + question)
      → text answer
  → answer appended to history as tool result
  → brain re-completes and speaks the answer
```

## Error handling

- No camera frame (`read()` returns `None`) → friendly string, no exception.
- OpenAI / network error in the vision call → caught, logged, friendly string.
- Empty `question` → default prompt `"Describe what you see."`.
- `look_fn` absent → tool not offered (cannot be called).

## Testing (against fakes, no devices/network)

- Inject a fake `look_fn` into `ChatBrain`:
  - `look` is offered in `_active_tools()` iff `look_fn` is present.
  - `LOOK_HINT` appears in `_system_prompt()` iff `look_fn` is present.
  - `_tool_look` passes the question through to `look_fn` and returns its result.
  - A `look_fn` that raises → `_tool_look` returns the friendly fallback string.
- Unit-test `encode_frame_jpeg` on a small synthetic numpy array: verifies
  downscaling of the long edge to `max_px` and the `data:image/jpeg;base64,`
  shape.
- No new dependencies (`opencv-python` is already required).

## Notes / decisions

- **Logging:** each `look()` call logs the question and a short form of the
  answer to `data/reachy.log` — privacy-relevant (it describes the room),
  consistent with existing logging of everything heard and said.
- **Latency:** `look()` adds one extra round-trip mid-turn (brain → tool → vision
  call → brain re-completion), the same shape as `web_search`. Accepted.
- **Scope: read-only.** `look()` never writes to the store. If the brain wants to
  remember what it saw, it calls `save_note` separately. No auto-remember.
- **No spoken filler** (e.g. "let me have a look…") in v1 — deferred.
- **`vision_image_max_px` default = 1024** — confirmed acceptable; can be raised
  via env var for OCR-heavy scenes.

## Out of scope

- 3D reconstruction / depth (separate future spike; needs a controlled head sweep
  + a feed-forward multi-view model).
- On-robot camera streaming over WiFi — a pre-existing gap. `look()` works today
  with the Mac webcam (`--source mac`) and will work with the robot camera once
  that path is wired.

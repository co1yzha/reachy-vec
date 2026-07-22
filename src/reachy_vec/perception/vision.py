"""Camera vision tools: encode a frame and answer questions about it (look),
built as injected closures so ChatBrain stays text-only and device-free.
Heavy imports (cv2) are deferred per repo convention.
"""

import base64
import logging
import time
from collections.abc import Callable
from datetime import datetime

logger = logging.getLogger(__name__)

VISION_SYSTEM = (
    "You are the robot's eyes. Answer in one or two short spoken sentences. "
    "No markdown, no lists."
)
DEFAULT_LOOK_PROMPT = "Describe what you see."
LOOK_AROUND_PROMPT = (
    "You turned your head to look around the room. Using these views, describe "
    "your surroundings in one or two short spoken sentences, noting roughly "
    "where things are (to your left, to your right)."
)


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
            path = photos_dir / f"{datetime.now():%Y-%m-%d-%H%M%S-%f}.jpg"
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

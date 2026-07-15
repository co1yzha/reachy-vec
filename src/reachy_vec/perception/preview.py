"""Live preview of what Reachy sees: webcam frame + face box + match label.

Wraps the (camera, matcher) pair as a sight() callable for the Oracle loop,
rendering each polled frame to a window. Note: the window only refreshes
when sight() is polled - it freezes while the robot is listening/speaking.
"""

from collections.abc import Callable

GREEN, ORANGE, GRAY = (80, 220, 80), (0, 160, 255), (160, 160, 160)


class PreviewSight:
    def __init__(self, camera, matcher, show: Callable | None = None):
        self._camera = camera
        self._matcher = matcher
        self._show = show if show is not None else self._cv2_show

    def __call__(self):
        frame = self._camera.read()
        if frame is None:
            return None
        observation = self._matcher.observe(frame)
        self._show(frame, observation, getattr(self._matcher, "last_bbox", None))
        return observation

    @staticmethod
    def _annotate(frame, observation, bbox):
        """Face box + label, drawn on a writable copy when needed. Robot
        WebRTC frames arrive read-only/non-contiguous - cv2 can't draw on
        those in place - and bbox coords may be numpy floats."""
        import cv2
        import numpy as np

        if bbox is None:
            return frame
        frame = np.ascontiguousarray(frame)
        if not frame.flags.writeable:
            frame = frame.copy()
        if observation is None:
            color, label = GRAY, "borderline"
        elif observation.person_id is None:
            color, label = ORANGE, f"unknown ({observation.score:.2f})"
        else:
            color, label = GREEN, f"{observation.name} ({observation.score:.2f})"
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        return frame

    @staticmethod
    def _cv2_show(frame, observation, bbox) -> None:
        import cv2

        frame = PreviewSight._annotate(frame, observation, bbox)
        cv2.imshow("Reachy sees", frame)
        cv2.waitKey(1)

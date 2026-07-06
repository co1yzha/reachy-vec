"""Frame sources. Phase 1 uses the Mac webcam; the robot camera slots in later."""

from typing import Protocol


class Camera(Protocol):
    def read(self): ...  # returns an ndarray frame or None


class WebcamCamera:
    def __init__(self, index: int = 0):
        import cv2

        self._cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

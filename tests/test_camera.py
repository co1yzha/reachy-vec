import numpy as np

from reachy_vec.perception.camera import RobotCamera
from tests.conftest import FakeMedia


def test_robot_camera_reads_frames_from_media():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cam = RobotCamera(FakeMedia(frames=[frame]))
    assert cam.read() is frame
    assert cam.read() is None  # exhausted


def test_robot_camera_passes_through_none():
    cam = RobotCamera(FakeMedia(frames=[]))
    assert cam.read() is None

from reachy_vec.perception.face import Observation
from reachy_vec.perception.preview import PreviewSight
from tests.conftest import FakeCamera, FakeFaceMatcher

ALICE = Observation(person_id="p1", name="Alice", score=0.9)


def test_preview_sight_returns_observation_and_renders():
    shown = []
    matcher = FakeFaceMatcher(observations=[ALICE])
    matcher.last_bbox = (10, 20, 110, 140)
    sight = PreviewSight(
        FakeCamera(frames=["frame"]),
        matcher,
        show=lambda frame, obs, bbox: shown.append((frame, obs, bbox)),
    )
    assert sight() == ALICE
    assert shown == [("frame", ALICE, (10, 20, 110, 140))]


def test_preview_sight_no_frame_skips_render():
    shown = []
    sight = PreviewSight(
        FakeCamera(frames=[]),
        FakeFaceMatcher(observations=[]),
        show=lambda *a: shown.append(a),
    )
    assert sight() is None
    assert shown == []


def test_annotate_draws_on_readonly_float_bbox_frame():
    """Robot WebRTC frames arrive read-only, and bboxes can be numpy floats;
    annotation must copy the frame and coerce coords, not crash cv2."""
    import numpy as np

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame.setflags(write=False)
    bbox = (np.float32(5.0), np.float32(6.0), np.float32(40.0), np.float32(42.0))
    out = PreviewSight._annotate(frame, ALICE, bbox)
    assert out.flags.writeable
    assert out.any()  # the green box was actually drawn


def test_annotate_without_bbox_passes_frame_through():
    import numpy as np

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame.setflags(write=False)
    out = PreviewSight._annotate(frame, None, None)
    assert not out.any()

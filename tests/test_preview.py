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

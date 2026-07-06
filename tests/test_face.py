from reachy_vec.perception.face import Observation, enroll_person
from reachy_vec.store.db import Store

from tests.conftest import FakeCamera, FakeFaceMatcher


def test_enroll_person_stores_frames_and_is_matchable(tmp_path):
    store = Store(tmp_path / "db")
    vec = [1.0] + [0.0] * 511
    camera = FakeCamera(frames=["f"] * 5)
    matcher = FakeFaceMatcher(observations=[], embedding=vec)
    prompts: list[str] = []

    person_id = enroll_person("Alice", camera, matcher, store, prompts.append)

    assert person_id is not None
    assert store.people_count() == 1
    matched = store.match_face(vec)
    assert matched is not None and matched[1] == "Alice"
    assert len(prompts) == 5  # one guidance prompt per capture


def test_enroll_person_fails_gracefully_without_face(tmp_path):
    store = Store(tmp_path / "db")
    camera = FakeCamera(frames=["f"] * 5)
    matcher = FakeFaceMatcher(observations=[], embedding=None)  # never sees a face
    assert enroll_person("Alice", camera, matcher, store, lambda _: None) is None
    assert store.people_count() == 0


def test_observation_unknown_vs_known():
    unknown = Observation(person_id=None, name=None, score=0.2)
    known = Observation(person_id="p1", name="Alice", score=0.9)
    assert unknown.person_id is None and known.person_id == "p1"

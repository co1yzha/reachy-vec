import pytest

from reachy_vec.perception.face import Observation
from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity, fuse

ALICE_FACE = Observation(person_id="p1", name="Alice", score=0.9)
BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.1)
UNKNOWN_FACE = Observation(person_id=None, name=None, score=0.1)


@pytest.mark.parametrize(
    ("face", "voice", "expected"),
    [
        # voice knows -> voice wins, even against a confident face
        (ALICE_FACE, BOB_VOICE, TurnIdentity("p2", "Bob")),
        (None, BOB_VOICE, TurnIdentity("p2", "Bob")),
        # confident unknown voice -> anonymous, face cannot override
        (ALICE_FACE, UNKNOWN_VOICE, ANONYMOUS),
        (None, UNKNOWN_VOICE, ANONYMOUS),
        # voice can't tell -> face decides
        (ALICE_FACE, None, TurnIdentity("p1", "Alice")),
        (UNKNOWN_FACE, None, ANONYMOUS),
        (None, None, ANONYMOUS),
    ],
)
def test_fusion_truth_table(face, voice, expected):
    assert fuse(face, voice) == expected


def test_observation_face_count_defaults_to_one():
    assert ALICE_FACE.face_count == 1
    assert Observation(person_id="p1", name="Alice", score=0.9, face_count=2).face_count == 2

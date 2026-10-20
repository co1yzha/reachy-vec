import pytest

from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity, from_voice
from reachy_vec.perception.voice import Observation

BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.1)


@pytest.mark.parametrize(
    ("voice", "expected"),
    [
        (BOB_VOICE, TurnIdentity("p2", "Bob")),   # known voice -> that person
        (UNKNOWN_VOICE, ANONYMOUS),               # stranger -> anonymous
        (None, ANONYMOUS),                        # can't tell -> anonymous, never guess
    ],
)
def test_identity_follows_voice_only(voice, expected):
    assert from_voice(voice) == expected

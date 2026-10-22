import pytest

from reachy_vec.body.emotion import split_emojis
from reachy_vec.body.motions import MOTIONS


@pytest.mark.parametrize(("text", "clean", "emotions"), [
    ("Great news! 😄🎉", "Great news!", ["happy", "excited"]),
    ("That's hilarious 😂😂", "That's hilarious", ["laugh"]),          # deduped
    ("Oh no 😢 that's sad.", "Oh no that's sad.", ["sad"]),
    ("Hmm 🤔", "Hmm", ["thinking"]),
    ("Love it ❤️", "Love it", ["love"]),
    ("Plain text.", "Plain text.", []),
    ("Weird 🦖 dino", "Weird dino", []),                                # unknown: just removed
    ("Launch! 🚀👍", "Launch!", ["excited", "happy"]),
])
def test_split_emojis(text, clean, emotions):
    assert split_emojis(text) == (clean, emotions)


@pytest.mark.parametrize("emotion", ["happy", "excited", "laugh", "love", "sad", "surprised",
                                     "thinking"])
def test_every_emotion_has_a_motion_that_returns_to_neutral(emotion):
    frames = MOTIONS[emotion]
    assert frames and frames[-1].head == {} and frames[-1].antennas == (0.0, 0.0)
    assert sum(f.duration for f in frames) <= 2.0          # brief: it plays while talking

"""Post-hoc vocabulary snapping: replaces the coercive STT "Vocabulary: ..."
prompt (which made gpt-4o-transcribe hallucinate demo titles verbatim on echo
and noise) with fuzzy matching of the transcript against known titles."""

import numpy as np

from reachy_vec.audio.listen import Utterance
from reachy_vec.audio.vocab import VocabSnappingTranscriber, snap_to_vocab
from tests.conftest import FakeTranscriber

TITLES = [
    "CareBrain Note Taker Interface",
    "Bon-Tag Realtime GPS Dashboard",
    "Advanced Energy Management System for Liverpool Science Park IC2",
]


def test_snaps_close_mishearing_of_a_whole_title():
    heard = "carebrain notetaker interface"
    assert snap_to_vocab(heard, TITLES) == "CareBrain Note Taker Interface"


def test_snaps_title_embedded_in_a_longer_utterance():
    heard = "tell me about carebrain note taker interface please"
    assert (
        snap_to_vocab(heard, TITLES)
        == "tell me about CareBrain Note Taker Interface please"
    )


def test_leaves_unrelated_text_alone():
    heard = "what is the weather like today"
    assert snap_to_vocab(heard, TITLES) == heard


def test_no_titles_is_a_no_op():
    assert snap_to_vocab("anything at all", []) == "anything at all"


def test_empty_text_is_a_no_op():
    assert snap_to_vocab("", TITLES) == ""


def test_short_unrelated_utterance_is_not_snapped():
    # a lone stray syllable (echo residue) must not become a demo title
    assert snap_to_vocab("g", TITLES) == "g"


def test_transcriber_wrapper_snaps_text_and_keeps_audio():
    audio = np.zeros(16000, dtype=np.float32)
    inner = FakeTranscriber([Utterance(text="care brain note taker interface", audio=audio)])
    wrapped = VocabSnappingTranscriber(inner, TITLES)
    utt = wrapped.listen_once(5)
    assert utt.text == "CareBrain Note Taker Interface"
    assert utt.audio is audio


def test_transcriber_wrapper_passes_silence_through():
    wrapped = VocabSnappingTranscriber(FakeTranscriber([]), TITLES)
    assert wrapped.listen_once(5) is None

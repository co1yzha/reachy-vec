from reachy_vec.audio.listen import collect_utterance
from tests.conftest import FakeTranscriber

SPEECH, SILENCE = "s", "."


def run_collect(pattern: str, max_silence: int = 2):
    chunks = list(pattern)
    return collect_utterance(
        iter(chunks), is_speech=lambda c: c == SPEECH, max_silence_chunks=max_silence
    )


def test_collects_speech_and_stops_after_trailing_silence():
    assert run_collect("..sss..x") == ["s", "s", "s", ".", "."]


def test_returns_none_when_no_speech_at_all():
    assert run_collect("......") is None


def test_short_pause_inside_utterance_is_kept():
    assert run_collect("ss.ss..", max_silence=2) == ["s", "s", ".", "s", "s", ".", "."]


def test_fake_transcriber_scripts_then_silence():
    t = FakeTranscriber(["hello"])
    assert t.listen_once(5) == "hello"
    assert t.listen_once(5) is None

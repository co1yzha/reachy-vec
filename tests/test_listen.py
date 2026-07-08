import numpy as np

from reachy_vec.audio.listen import MicTranscriber, Utterance, collect_utterance
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
    assert t.listen_once(5).text == "hello"
    assert t.listen_once(5) is None


def test_utterance_carries_text_and_audio():
    audio = np.zeros(16000, dtype=np.float32)
    utt = Utterance(text="hello", audio=audio)
    assert utt.text == "hello"
    assert utt.audio is audio


def test_mic_transcriber_returns_utterance(monkeypatch):
    t = MicTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: audio)
    monkeypatch.setattr(t, "_load", lambda: None)

    class FakeSeg:
        text = " hi there "

    t._whisper = type("W", (), {"transcribe": lambda self, a, **kw: ([FakeSeg()], None)})()
    utt = t.listen_once(5)
    assert utt.text == "hi there"
    assert utt.audio is audio


def test_mic_transcriber_silence_returns_none(monkeypatch):
    t = MicTranscriber()
    monkeypatch.setattr(t, "_capture", lambda timeout_s: None)
    monkeypatch.setattr(t, "_load", lambda: None)
    assert t.listen_once(5) is None

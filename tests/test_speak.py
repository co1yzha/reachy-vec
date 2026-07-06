import pytest

from reachy_vec.audio.speak import SaySpeaker, make_speaker


def test_say_speaker_invokes_say():
    calls = []
    speaker = SaySpeaker(run=lambda cmd, **kw: calls.append(cmd))
    speaker.speak("hello team")
    assert calls == [["say", "hello team"]]


def test_say_speaker_skips_empty_text():
    calls = []
    SaySpeaker(run=lambda cmd, **kw: calls.append(cmd)).speak("   ")
    assert calls == []


def test_make_speaker_say_backend(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "say")
    assert isinstance(make_speaker(), SaySpeaker)


def test_make_speaker_unimplemented_backend_raises(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "fish-speech")
    with pytest.raises(NotImplementedError, match="fish-speech"):
        make_speaker()

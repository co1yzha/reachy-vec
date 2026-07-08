import logging
from pathlib import Path

import pytest

from reachy_vec.audio.speak import QwenTTSSpeaker, SaySpeaker, make_speaker


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


def test_qwen_speaker_synthesizes_and_plays():
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (f"AUDIO<{text}>", 24000),
        play=lambda audio, sr: played.append((audio, sr)),
    )
    speaker.speak("hello team")
    assert played == [("AUDIO<hello team>", 24000)]


def test_qwen_speaker_skips_empty_text():
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: ("AUDIO", 24000),
        play=lambda audio, sr: played.append((audio, sr)),
    )
    speaker.speak("   ")
    assert played == []


def test_qwen_speaker_logs_and_skips_on_synthesis_error(caplog):
    def boom(text):
        raise RuntimeError("mlx exploded")

    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=boom,
        play=lambda audio, sr: played.append((audio, sr)),
    )
    with caplog.at_level(logging.ERROR):
        speaker.speak("hello")  # must not raise
    assert played == []
    assert "TTS synthesis failed" in caplog.text

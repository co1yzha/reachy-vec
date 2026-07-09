import logging
from pathlib import Path

import numpy as np
import pytest

from reachy_vec.audio.speak import (
    QwenTTSSpeaker,
    RobotAudioSink,
    SaySpeaker,
    make_speaker,
)
from tests.conftest import FakeMedia


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self._done = False

    def wait(self):
        self._done = True

    def poll(self):
        return 0 if self._done else None

    def terminate(self):
        self.terminated = True


def test_say_speaker_invokes_say():
    cmds = []

    def popen(cmd):
        cmds.append(cmd)
        return _FakeProc()

    SaySpeaker(popen=popen).speak("hello team")
    assert cmds == [["say", "hello team"]]


def test_say_speaker_skips_empty_text():
    cmds = []
    SaySpeaker(popen=lambda cmd: cmds.append(cmd) or _FakeProc()).speak("   ")
    assert cmds == []


def test_say_speaker_stop_terminates_running_process():
    proc = _FakeProc()
    speaker = SaySpeaker(popen=lambda cmd: proc)
    speaker.speak("a long sentence")  # _FakeProc.wait() marks it done immediately
    proc._done = False                # pretend it's still playing
    speaker.stop()
    assert proc.terminated is True


def test_say_speaker_stop_is_safe_when_idle():
    SaySpeaker(popen=lambda cmd: _FakeProc()).stop()  # must not raise


def test_qwen_speaker_stop_calls_injected_stop():
    stops = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: ("AUDIO", 24000),
        play=lambda audio, sr: None,
        stop=lambda: stops.append(True),
    )
    speaker.stop()
    assert stops == [True]


def test_make_speaker_say_backend(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "say")
    assert isinstance(make_speaker(), SaySpeaker)


def test_make_speaker_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "kokoro")
    with pytest.raises(NotImplementedError, match="kokoro"):
        make_speaker()


def test_make_speaker_qwen_backend(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    assert isinstance(make_speaker(), QwenTTSSpeaker)


def test_make_speaker_qwen_requires_voice_sample(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", None)
    with pytest.raises(ValueError, match="REACHY_VEC_VOICE_SAMPLE"):
        make_speaker()


def test_make_speaker_qwen_rejects_missing_sample_file(monkeypatch, tmp_path):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr(
        "reachy_vec.audio.speak.settings.voice_sample", tmp_path / "nope.wav"
    )
    with pytest.raises(ValueError, match="REACHY_VEC_VOICE_SAMPLE"):
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


def test_robot_sink_pushes_resampled_audio_to_media():
    media = FakeMedia(out_rate=16000)  # no resample when rates match
    sink = RobotAudioSink(media)
    audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    sink(audio, sample_rate=16000)
    assert len(media.pushed) == 1
    np.testing.assert_allclose(media.pushed[0], audio)


def test_robot_sink_resamples_to_output_rate():
    media = FakeMedia(out_rate=48000)
    RobotAudioSink(media)(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert abs(len(media.pushed[0]) - 48000) <= 2  # upsampled ~3x


def test_make_speaker_qwen_uses_robot_sink_when_media_given(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    speaker = make_speaker(media=FakeMedia())
    assert isinstance(speaker._play, RobotAudioSink)


def test_make_speaker_say_with_media_warns_and_stays_local(monkeypatch, caplog):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "say")
    with caplog.at_level(logging.WARNING):
        speaker = make_speaker(media=FakeMedia())
    assert isinstance(speaker, SaySpeaker)
    assert "say" in caplog.text.lower()

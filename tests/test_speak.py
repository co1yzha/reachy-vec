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
    sink = RobotAudioSink(media, sleep=lambda s: None)
    audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    sink(audio, sample_rate=16000)
    assert len(media.pushed) == 1
    np.testing.assert_allclose(media.pushed[0], audio)


def test_robot_sink_resamples_to_output_rate():
    media = FakeMedia(out_rate=48000)
    sink = RobotAudioSink(media, sleep=lambda s: None)
    sink(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert abs(len(media.pushed[0]) - 48000) <= 2  # upsampled ~3x


def test_robot_sink_blocks_for_the_playback_duration():
    """push_audio_sample returns before the speaker finishes; without the
    wait, speak() returns mid-utterance and the mic hears the robot talk."""
    media = FakeMedia(out_rate=16000)
    naps = []
    sink = RobotAudioSink(media, sleep=naps.append)
    sink(np.zeros(8000, dtype=np.float32), sample_rate=16000)  # 0.5s of audio
    assert naps == [0.5]


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


def test_qwen_speaker_boosts_quiet_audio_to_target_rms():
    quiet = np.full(1000, 0.01, dtype=np.float32)
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (quiet, 24000),
        play=lambda audio, sr: played.append(audio),
        target_rms=0.1,
    )
    speaker.speak("hello")
    rms = float(np.sqrt(np.mean(np.square(played[0]))))
    assert rms == pytest.approx(0.1, rel=1e-3)


def test_qwen_speaker_boost_never_clips():
    """A quiet-RMS clip with a loud transient must be gain-limited by its
    peak, not clipped - clipping is audible distortion on playback."""
    audio = np.full(1000, 0.01, dtype=np.float32)
    audio[0] = 0.5  # transient
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (audio, 24000),
        play=lambda a, sr: played.append(a),
        target_rms=0.5,
    )
    speaker.speak("hello")
    assert float(np.max(np.abs(played[0]))) <= 1.0
    assert played[0][0] == pytest.approx(0.99)  # peak-limited, not clipped


def test_qwen_speaker_leaves_loud_audio_alone():
    loud = np.full(1000, 0.3, dtype=np.float32)
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (loud, 24000),
        play=lambda a, sr: played.append(a),
        target_rms=0.1,
    )
    speaker.speak("hello")
    np.testing.assert_array_equal(played[0], loud)


def test_qwen_speaker_zero_target_rms_disables_boost():
    quiet = np.full(1000, 0.01, dtype=np.float32)
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (quiet, 24000),
        play=lambda a, sr: played.append(a),
        target_rms=0.0,
    )
    speaker.speak("hello")
    np.testing.assert_array_equal(played[0], quiet)


def test_afplay_player_writes_wav_and_plays_it():
    from reachy_vec.audio.speak import AfplayPlayer

    cmds = []

    def popen(cmd):
        cmds.append(cmd)
        return _FakeProc()

    player = AfplayPlayer(popen=popen)
    player(np.zeros(2400, dtype=np.float32), 24000)
    assert len(cmds) == 1 and cmds[0][0] == "afplay"
    wav_path = Path(cmds[0][1])
    assert not wav_path.exists()  # temp file cleaned up after playback


def test_afplay_player_stop_terminates_playback():
    from reachy_vec.audio.speak import AfplayPlayer

    proc = _FakeProc()
    player = AfplayPlayer(popen=lambda cmd: proc)
    player(np.zeros(100, dtype=np.float32), 24000)
    proc._done = False  # pretend it's still playing
    player.stop()
    assert proc.terminated is True


def test_afplay_player_stop_is_safe_when_idle():
    from reachy_vec.audio.speak import AfplayPlayer

    AfplayPlayer(popen=lambda cmd: _FakeProc()).stop()  # must not raise


def test_make_speaker_qwen_defaults_to_afplay_on_mac(monkeypatch, tmp_path):
    """Mac playback must run out-of-process: sounddevice's Python callback
    starves under app load (GIL) and crackles."""
    from reachy_vec.audio.speak import AfplayPlayer

    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    speaker = make_speaker()
    assert isinstance(speaker._play, AfplayPlayer)


def test_qwen_speaker_default_stop_stops_the_afplay_player(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    speaker = make_speaker()
    proc = _FakeProc()
    speaker._play._popen = lambda cmd: proc
    speaker._play(np.zeros(100, dtype=np.float32), 24000)
    proc._done = False
    speaker.stop()
    assert proc.terminated is True

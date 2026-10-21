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


def test_robot_sink_starts_playback_before_pushing_once():
    """The gstreamer output appsrc drops every buffer (silently, only a
    stderr warning) until start_playing() runs. Without it the robot makes no
    sound while the brain happily logs replies. Start once, before the push."""
    media = FakeMedia(out_rate=16000)
    sink = RobotAudioSink(media, sleep=lambda s: None)
    sink(np.zeros(1600, dtype=np.float32), sample_rate=16000)
    sink(np.zeros(1600, dtype=np.float32), sample_rate=16000)
    assert media.play_started == 1  # started once, not per clip
    assert media.pushed_after_start == [True, True]  # every push after start


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


def _band_energy(audio, sample_rate, lo, hi):
    spec = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def test_presence_shelf_lifts_highs_relative_to_lows():
    """A positive presence gain must boost high-frequency energy (clarity)
    while leaving low-frequency energy essentially untouched."""
    from reachy_vec.audio.speak import _presence_shelf

    sr = 24000
    t = np.arange(sr) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)).astype(
        np.float32
    )
    out = _presence_shelf(audio, sr, gain_db=6.0, corner_hz=3000.0)

    low_in = _band_energy(audio, sr, 100, 400)
    low_out = _band_energy(out, sr, 100, 400)
    high_in = _band_energy(audio, sr, 4000, 6000)
    high_out = _band_energy(out, sr, 4000, 6000)

    assert high_out / high_in == pytest.approx(2.0, rel=0.15)  # ~+6 dB
    assert low_out / low_in == pytest.approx(1.0, rel=0.1)  # lows unchanged


def test_presence_shelf_zero_gain_is_a_no_op():
    from reachy_vec.audio.speak import _presence_shelf

    audio = np.linspace(-0.4, 0.4, 500, dtype=np.float32)
    out = _presence_shelf(audio, 24000, gain_db=0.0, corner_hz=3000.0)
    np.testing.assert_array_equal(out, audio)


def test_soft_limit_tames_peaks_without_clipping():
    from reachy_vec.audio.speak import _soft_limit

    audio = np.array([0.2, -0.5, 3.0, -8.0, 0.1], dtype=np.float32)
    out = _soft_limit(audio, ceiling=0.99)
    assert float(np.max(np.abs(out))) <= 0.99
    # a sample already well under the ceiling is left near-transparent
    assert out[0] == pytest.approx(0.2, abs=0.02)


def test_soft_limit_transparent_below_ceiling():
    from reachy_vec.audio.speak import _soft_limit

    audio = np.array([0.2, -0.5, 0.9, -0.3], dtype=np.float32)
    out = _soft_limit(audio, ceiling=0.99)
    np.testing.assert_array_equal(out, audio)  # nothing exceeds the ceiling


def test_qwen_speaker_boosts_body_by_reducing_crest_factor():
    """The loudness stage boosts the sustained body toward the target even when
    a transient would peak-limit a single-gain approach - the soft limiter
    tames the transient instead of capping the whole clip by it."""
    audio = np.full(2000, 0.02, dtype=np.float32)
    audio[0] = 0.6  # loud transient
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (audio, 24000),
        play=lambda a, sr: played.append(a),
        target_rms=0.2,
    )
    speaker.speak("hello")
    body = played[0][1:]  # exclude the transient
    body_rms = float(np.sqrt(np.mean(np.square(body))))
    assert float(np.max(np.abs(played[0]))) <= 0.99  # never clips
    # old single-gain peak-limit would leave the body near 0.02 * (0.99/0.6);
    # crest reduction gets it far louder than that.
    assert body_rms > 0.1


def test_qwen_speaker_applies_presence_when_configured():
    sr = 24000
    t = np.arange(sr) / sr
    tone = (0.2 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)).astype(
        np.float32
    )
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (tone, sr),
        play=lambda a, sr_: played.append(a),
        target_rms=0.0,  # isolate the EQ from the loudness stage
        presence_db=6.0,
        presence_hz=3000.0,
    )
    speaker.speak("hello")
    high_ratio = _band_energy(played[0], sr, 4000, 6000) / _band_energy(tone, sr, 4000, 6000)
    assert high_ratio > 1.5


def test_make_speaker_qwen_wires_presence_from_settings(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_presence_db", 5.0)
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_presence_hz", 3200.0)
    speaker = make_speaker()
    assert speaker._presence_db == 5.0
    assert speaker._presence_hz == 3200.0


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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"),
        ("some/other-model", "some/other-model"),
    ],
)
def test_make_speaker_qwen_model_falls_back_to_default(
    monkeypatch, tmp_path, configured, expected
):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_model", configured)
    speaker = make_speaker()
    assert speaker._model_id == expected


# -- pre-generated filler lines ------------------------------------------------


def test_qwen_precache_synthesizes_once_then_plays_from_cache():
    generated, played = [], []

    def generate(text):
        generated.append(text)
        return f"AUDIO<{text}>", 24000

    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"), generate=generate,
        play=lambda audio, sr: played.append(audio),
    )
    speaker.precache(["Mm.", "Right."])
    assert generated == ["Mm.", "Right."]          # synthesized up front
    speaker.speak("Mm.")
    speaker.speak("Mm.")
    assert generated == ["Mm.", "Right."]          # no re-synthesis
    assert played == ["AUDIO<Mm.>", "AUDIO<Mm.>"]
    speaker.speak("Something new.")
    assert generated[-1] == "Something new."       # uncached text still synthesizes


def test_qwen_precache_failure_is_non_fatal(caplog):
    def generate(text):
        raise RuntimeError("model down")

    speaker = QwenTTSSpeaker(sample_path=Path("me.wav"), generate=generate,
                             play=lambda a, s: None)
    with caplog.at_level(logging.WARNING):
        speaker.precache(["Mm."])                  # must not raise
    assert "Mm." in caplog.text


def test_say_precache_is_a_noop():
    SaySpeaker(popen=lambda *a, **k: None).precache(["Mm."])


def test_qwen_synthesize_and_play_clip_split_speak():
    played = []
    speaker = QwenTTSSpeaker(sample_path=Path("me.wav"),
                             generate=lambda text: (f"AUDIO<{text}>", 24000),
                             play=lambda audio, sr: played.append(audio))
    clip = speaker.synthesize("Hello.")
    assert played == []                       # synthesis alone plays nothing
    speaker.play_clip(clip)
    assert played == ["AUDIO<Hello.>"]


def test_qwen_synthesize_uses_precache():
    generated = []
    speaker = QwenTTSSpeaker(sample_path=Path("me.wav"),
                             generate=lambda t: generated.append(t) or ("A", 24000),
                             play=lambda a, s: None)
    speaker.precache(["Mm."])
    speaker.synthesize("Mm.")
    assert generated == ["Mm."]               # from cache the second time

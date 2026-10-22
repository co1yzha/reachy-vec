import numpy as np

from reachy_vec.audio.listen import (
    BargeInMonitor,
    MicSource,
    MicTranscriber,
    Utterance,
    _AudioCapture,
    collect_utterance,
)
from tests.conftest import FakeTranscriber

SPEECH, SILENCE = "s", "."


class ScriptedSource:
    """AudioSource fake: yields pre-baked frames, then stops (silence -> None)."""

    def __init__(self, frames):
        self._frames = frames

    def frames(self, chunk_samples):
        yield from self._frames


def test_capture_uses_injected_source():
    # two "speech" frames then two "silence" frames; VAD keyed on frame[0]
    speech = np.ones(512, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)
    cap = _AudioCapture(source=ScriptedSource([speech, speech, silence, silence]))
    cap._load_vad = lambda: None  # skip the real model
    cap._vad = lambda frame, rate: type("P", (), {"item": lambda self: float(frame[0])})()
    out = cap._capture(timeout_s=5)
    assert out is not None
    assert len(out) == 512 * 4  # 2 speech + 2 trailing silence (max_silence not hit)


def test_capture_keeps_a_sentence_started_late_in_the_window():
    """timeout_s bounds the wait for speech to START; once talking begins the
    utterance must not be truncated by the remaining window."""
    speech = np.ones(512, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)
    frames = [silence] * 5 + [speech] * 20  # speech starts just before timeout
    cap = _AudioCapture(source=ScriptedSource(frames))
    cap._load_vad = lambda: None
    cap._vad = lambda frame, rate: type("P", (), {"item": lambda self: float(frame[0])})()
    out = cap._capture(timeout_s=0.2)  # ~6 chunks of wait budget
    assert out is not None
    assert len(out) == 512 * 20  # every speech chunk kept


def test_default_source_is_mic():
    assert isinstance(_AudioCapture()._source, MicSource)


def _inline(fn):
    fn()  # run the watch synchronously; returns None (no thread)


def test_barge_in_fires_after_sustained_speech():
    speech = np.ones(512, dtype=np.float32)
    src = ScriptedSource([speech] * 5)
    fired = []
    mon = BargeInMonitor(
        src, min_speech_s=0.096, is_speech=lambda f: bool(f[0]), spawn=_inline
    )  # 0.096 / 0.032 = 3 chunks
    mon.start(on_fire=lambda: fired.append(True))
    assert mon.fired is True
    assert fired == [True]


def test_barge_in_ignores_brief_speech():
    speech, silence = np.ones(512, dtype=np.float32), np.zeros(512, dtype=np.float32)
    src = ScriptedSource([speech, silence, speech, silence])  # never 3 in a row
    mon = BargeInMonitor(
        src, min_speech_s=0.096, is_speech=lambda f: bool(f[0]), spawn=_inline
    )
    mon.start(on_fire=lambda: None)
    assert mon.fired is False


def test_barge_in_self_voice_is_vetoed():
    """The robot's own voice through the mic must not interrupt its reply:
    the injected is_self_voice check vetoes the run and watching continues."""
    speech = np.ones(512, dtype=np.float32)
    src = ScriptedSource([speech] * 6)
    checked = []

    def is_self(audio):
        checked.append(len(audio))
        return True

    mon = BargeInMonitor(
        src,
        min_speech_s=0.096,  # 3 chunks
        is_speech=lambda f: bool(f[0]),
        spawn=_inline,
        is_self_voice=is_self,
    )
    mon.start(on_fire=lambda: None)
    assert mon.fired is False
    assert checked == [512 * 3, 512 * 3]  # consulted with each run's audio, then reset


def test_barge_in_fires_when_voice_is_not_self():
    speech = np.ones(512, dtype=np.float32)
    src = ScriptedSource([speech] * 5)
    mon = BargeInMonitor(
        src,
        min_speech_s=0.096,
        is_speech=lambda f: bool(f[0]),
        spawn=_inline,
        is_self_voice=lambda audio: False,
    )
    mon.start(on_fire=lambda: None)
    assert mon.fired is True


def test_barge_in_survives_a_broken_source():
    class Boom:
        def frames(self, chunk_samples):
            raise RuntimeError("mic gone")
            yield  # pragma: no cover

    mon = BargeInMonitor(Boom(), is_speech=lambda f: True, spawn=_inline)
    mon.start(on_fire=lambda: None)  # must not raise
    assert mon.broken is True
    assert mon.fired is False


def run_collect(pattern: str, max_silence: int = 2, **kw):
    chunks = list(pattern)
    return collect_utterance(
        iter(chunks), is_speech=lambda c: c == SPEECH, max_silence_chunks=max_silence, **kw
    )


def test_gives_up_waiting_for_speech_after_max_wait():
    assert run_collect("........", max_wait_chunks=4) is None


def test_wait_budget_is_not_charged_against_the_utterance():
    """A question started late in the listen window must not be truncated:
    the timeout applies to waiting for speech, not to the whole capture."""
    assert run_collect("......ssssssss", max_wait_chunks=8) == list("ssssssss")


def test_utterance_length_is_capped_independently():
    assert run_collect("ssssssss", max_utterance_chunks=5) == list("sssss")


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


def test_gate_quiet_discards_near_silence():
    """VAD can trigger on room noise; near-silent segments make STT
    hallucinate its vocabulary prompt (it 'heard' demo names nobody said)."""
    from reachy_vec.audio.listen import gate_quiet

    noise = (np.random.default_rng(0).standard_normal(16000) * 0.001).astype(np.float32)
    assert gate_quiet(noise, min_rms=0.005) is None


def test_gate_quiet_passes_real_speech_levels():
    from reachy_vec.audio.listen import gate_quiet

    speech = (np.sin(np.linspace(0, 400 * np.pi, 16000)) * 0.05).astype(np.float32)
    out = gate_quiet(speech, min_rms=0.005)
    assert out is speech


def test_gate_quiet_passes_none_through():
    from reachy_vec.audio.listen import gate_quiet

    assert gate_quiet(None, min_rms=0.005) is None


def test_normalize_rms_boosts_quiet_capture_to_target():
    from reachy_vec.audio.listen import normalize_rms

    quiet = np.full(16000, 0.01, dtype=np.float32)  # RMS 0.01
    out = normalize_rms(quiet, target_rms=0.05)
    assert abs(float(np.sqrt(np.mean(np.square(out)))) - 0.05) < 1e-4


def test_normalize_rms_never_attenuates_loud_audio():
    from reachy_vec.audio.listen import normalize_rms

    loud = np.full(16000, 0.2, dtype=np.float32)
    assert normalize_rms(loud, target_rms=0.05) is loud


def test_normalize_rms_caps_the_gain():
    from reachy_vec.audio.listen import normalize_rms

    whisper_quiet = np.full(16000, 0.001, dtype=np.float32)  # would need 50x
    out = normalize_rms(whisper_quiet, target_rms=0.05, max_gain=10.0)
    assert abs(float(np.sqrt(np.mean(np.square(out)))) - 0.01) < 1e-4  # 10x only


def test_normalize_rms_passthrough_when_disabled_or_empty():
    from reachy_vec.audio.listen import normalize_rms

    audio = np.full(16000, 0.01, dtype=np.float32)
    assert normalize_rms(audio, target_rms=0.0) is audio
    assert normalize_rms(None, target_rms=0.05) is None


def test_mic_transcriber_normalizes_stt_audio_but_keeps_raw_capture(monkeypatch):
    """STT gets the level-boosted audio (far-field mic is quiet); the
    Utterance keeps the raw capture for speaker-ID."""
    t = MicTranscriber()
    quiet = np.full(16000, 0.01, dtype=np.float32)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: quiet)
    monkeypatch.setattr(t, "_load", lambda: None)
    monkeypatch.setattr("reachy_vec.audio.listen.settings.stt_target_rms", 0.05)
    received = {}

    class FakeSeg:
        text = "hi"

    class W:
        def transcribe(self, a, **kw):
            received["audio"] = a
            return [FakeSeg()], None

    t._whisper = W()
    utt = t.listen_once(5)
    assert utt.audio is quiet  # raw capture preserved
    got_rms = float(np.sqrt(np.mean(np.square(received["audio"]))))
    assert abs(got_rms - 0.05) < 1e-4


def _whisper_recording(seen):
    class FakeSeg:
        text = " Reachy "

    def transcribe(self, a, **kw):
        seen.update(kw)
        return [FakeSeg()], None

    return type("W", (), {"transcribe": transcribe})()


def test_mic_transcriber_passes_wake_hotwords(monkeypatch):
    """'Reachy' was heard as "I'm reaching." - hotwords bias whisper to it."""
    seen = {}
    t = MicTranscriber(hotwords="Reachy")
    monkeypatch.setattr(t, "_capture", lambda timeout_s: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(t, "_load", lambda: None)
    t._whisper = _whisper_recording(seen)
    t.listen_once(5)
    assert seen["hotwords"] == "Reachy"


def test_mic_transcriber_no_hotwords_when_blank(monkeypatch):
    seen = {}
    t = MicTranscriber(hotwords="")
    monkeypatch.setattr(t, "_capture", lambda timeout_s: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(t, "_load", lambda: None)
    t._whisper = _whisper_recording(seen)
    t.listen_once(5)
    assert seen.get("hotwords") is None


def test_default_hotwords_setting_is_reachy():
    from reachy_vec.config import Settings

    assert Settings.model_fields["stt_hotwords"].default == "Reachy"


# -- Parakeet backend -----------------------------------------------------------


def test_parakeet_transcriber_returns_utterance_and_keeps_raw_audio(monkeypatch):
    from reachy_vec.audio.listen import ParakeetTranscriber

    seen = []
    t = ParakeetTranscriber(transcribe=lambda audio: seen.append(audio) or " Reachy, hi. ")
    audio = np.full(16000, 0.001, dtype=np.float32)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: audio)
    monkeypatch.setattr(t, "_load", lambda: None)
    utt = t.listen_once(5)
    assert utt.text == "Reachy, hi." and utt.audio is audio
    assert seen[0] is not audio                 # STT gets the loudness-normalized copy


def test_parakeet_transcriber_silence_and_empty_text(monkeypatch):
    from reachy_vec.audio.listen import ParakeetTranscriber

    t = ParakeetTranscriber(transcribe=lambda audio: "  ")
    monkeypatch.setattr(t, "_load", lambda: None)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: None)
    assert t.listen_once(5) is None             # nothing captured
    monkeypatch.setattr(t, "_capture", lambda timeout_s: np.zeros(16000, dtype=np.float32))
    assert t.listen_once(5).text == ""          # captured, but no words: not silence


def test_parakeet_default_model_setting():
    from reachy_vec.config import Settings

    assert Settings.model_fields["parakeet_model"].default == "mlx-community/parakeet-tdt-0.6b-v2"


def test_stt_tap_saves_each_utterance_with_its_transcript(monkeypatch, tmp_path):
    import soundfile as sf

    from reachy_vec.audio.listen import ParakeetTranscriber

    monkeypatch.setenv("REACHY_VEC_STT_TAP", str(tmp_path))
    t = ParakeetTranscriber(transcribe=lambda audio: "")
    monkeypatch.setattr(t, "_load", lambda: None)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: np.full(8000, 0.01, dtype=np.float32))
    t.listen_once(5)
    files = sorted(tmp_path.glob("stt_tap_*.wav"))
    assert len(files) == 1
    audio, rate = sf.read(files[0])
    assert rate == 16000 and len(audio) == 8000
    assert (tmp_path / "stt_tap.log").read_text().strip().endswith("''")


def test_barge_in_logs_when_it_fires(caplog):
    import logging

    speech = np.ones(512, dtype=np.float32)
    mon = BargeInMonitor(ScriptedSource([speech] * 5), min_speech_s=0.096,
                         is_speech=lambda f: bool(f[0]), spawn=_inline)
    with caplog.at_level(logging.INFO, logger="reachy_vec.audio.listen"):
        mon.start(on_fire=lambda: None)
    assert "barge-in fired" in caplog.text


def test_capture_survives_a_vanished_microphone(monkeypatch, caplog):
    """A USB mic dropping off raised PortAudioError out of the Oracle loop."""
    import logging

    class GoneSource:
        def frames(self, chunk_samples):
            raise OSError("Error querying device -1")   # stands in for PortAudioError
            yield

    naps = []
    monkeypatch.setattr("reachy_vec.audio.listen.time.sleep", naps.append)
    cap = _AudioCapture(source=GoneSource())
    cap._load_vad = lambda: None
    cap._vad = lambda frame, rate: None
    with caplog.at_level(logging.WARNING, logger="reachy_vec.audio.listen"):
        assert cap._capture(timeout_s=5) is None
    assert "microphone unavailable" in caplog.text and naps == [1.0]


def test_parakeet_returns_empty_utterance_for_wordless_sound(monkeypatch):
    """Sound but no words is not silence: callers keep listening."""
    from reachy_vec.audio.listen import ParakeetTranscriber

    t = ParakeetTranscriber(transcribe=lambda audio: "   ")
    monkeypatch.setattr(t, "_load", lambda: None)
    audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: audio)
    utt = t.listen_once(5)
    assert utt is not None and utt.text == "" and utt.audio is audio


def test_barge_in_with_confirm_fires_only_when_confirmed():
    speech = np.ones(512, dtype=np.float32)
    checks = []

    def confirm(audio):
        checks.append(len(audio))
        return len(checks) >= 2           # first window rejected, second accepted

    fired = []
    mon = BargeInMonitor(ScriptedSource([speech] * 12), min_speech_s=0.096,
                         is_speech=lambda f: bool(f[0]), spawn=_inline, confirm=confirm)
    mon.start(on_fire=lambda: fired.append(True))
    assert fired == [True] and len(checks) == 2


def test_barge_in_with_confirm_never_fires_on_unconfirmed_speech():
    """Reachy's own voice keeps the VAD busy for the whole reply - without the
    interrupt phrase it must never fire."""
    speech = np.ones(512, dtype=np.float32)
    checks = []
    mon = BargeInMonitor(ScriptedSource([speech] * 200), min_speech_s=0.096,
                         is_speech=lambda f: bool(f[0]), spawn=_inline,
                         confirm=lambda audio: checks.append(len(audio)) or False)
    mon.start(on_fire=lambda: None)
    assert mon.fired is False and len(checks) > 3
    assert max(checks) <= int(16000 * 3.0)      # rolling window stays bounded


def test_parakeet_transcribe_audio_for_barge_in():
    from reachy_vec.audio.listen import ParakeetTranscriber

    t = ParakeetTranscriber(transcribe=lambda audio: " Reachy, stop. ")
    t._load = lambda: None
    assert t.transcribe_audio(np.zeros(8000, dtype=np.float32)) == "Reachy, stop."


def test_note_mic_logs_only_when_the_mic_changes(caplog, monkeypatch):
    import logging

    from reachy_vec.audio import listen

    monkeypatch.setattr(listen, "_current_mic", None)
    with caplog.at_level(logging.INFO, logger="reachy_vec.audio.listen"):
        listen.note_mic("Mac input 'Wireless Microphone RX'")
        listen.note_mic("Mac input 'Wireless Microphone RX'")   # barge-in reopens it
        listen.note_mic("Mac input 'Mac mini Microphone'")
    mics = [r.getMessage() for r in caplog.records if r.getMessage().startswith("mic:")]
    assert mics == ["mic: Mac input 'Wireless Microphone RX'",
                    "mic: Mac input 'Mac mini Microphone'"]

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


def test_barge_in_survives_a_broken_source():
    class Boom:
        def frames(self, chunk_samples):
            raise RuntimeError("mic gone")
            yield  # pragma: no cover

    mon = BargeInMonitor(Boom(), is_speech=lambda f: True, spawn=_inline)
    mon.start(on_fire=lambda: None)  # must not raise
    assert mon.broken is True
    assert mon.fired is False


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

import io

from reachy_vec.audio.listen import MicTranscriber, OpenAITranscriber, make_transcriber


def test_make_transcriber_local(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.listen.settings.stt_backend", "local")
    t = make_transcriber(initial_prompt="Vocabulary: VEC")
    assert isinstance(t, MicTranscriber)


def test_make_transcriber_openai(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.listen.settings.stt_backend", "openai")
    t = make_transcriber(client=object(), initial_prompt=None)
    assert isinstance(t, OpenAITranscriber)


class FakeTranscriptions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class R:
            text = "hello from openai"

        return R()


class FakeAudioClient:
    def __init__(self):
        self.audio = type("A", (), {"transcriptions": FakeTranscriptions()})()


def test_openai_transcriber_sends_wav_and_prompt(monkeypatch):
    client = FakeAudioClient()
    t = OpenAITranscriber(client, initial_prompt="Vocabulary: foodmapping")
    import numpy as np

    monkeypatch.setattr(t, "_capture", lambda timeout_s: np.zeros(16000, dtype=np.float32))
    assert t.listen_once(5).text == "hello from openai"
    kwargs = client.audio.transcriptions.last_kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    assert kwargs["prompt"] == "Vocabulary: foodmapping"
    assert isinstance(kwargs["file"], io.BytesIO)


def test_openai_transcriber_silence_returns_none(monkeypatch):
    t = OpenAITranscriber(FakeAudioClient())
    monkeypatch.setattr(t, "_capture", lambda timeout_s: None)
    assert t.listen_once(5) is None

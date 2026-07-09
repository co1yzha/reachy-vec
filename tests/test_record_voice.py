import wave

import numpy as np
from typer.testing import CliRunner

from reachy_vec.audio.record import record_sample, write_wav
from reachy_vec.cli import app

runner = CliRunner()


def test_write_wav_round_trip(tmp_path):
    audio = np.linspace(-1.0, 1.0, 2400, dtype=np.float32)
    out = tmp_path / "sample.wav"

    write_wav(out, audio, sample_rate=24000)

    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 24000
        assert w.getsampwidth() == 2  # 16-bit PCM
        assert w.getnframes() == 2400


def test_record_sample_uses_injected_recorder():
    fake = np.ones(240, dtype=np.float32)
    captured = {}

    def fake_record(frames, sample_rate):
        captured["frames"] = frames
        captured["sample_rate"] = sample_rate
        return fake

    out = record_sample(0.01, 24000, record=fake_record)

    assert out is fake
    assert captured == {"frames": 240, "sample_rate": 24000}


def test_record_voice_command_writes_wav_and_prints_env(tmp_path, monkeypatch):
    out = tmp_path / "voice_sample.wav"
    monkeypatch.setattr(
        "reachy_vec.cli.record_voice.record_sample",
        lambda duration_s, sample_rate: np.ones(
            int(sample_rate * duration_s), dtype=np.float32
        ),
    )

    result = runner.invoke(app, ["record-voice", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "REACHY_VEC_TTS_BACKEND=qwen-tts" in result.output
    assert f"REACHY_VEC_VOICE_SAMPLE={out}" in result.output
    assert "REACHY_VEC_VOICE_SAMPLE_TEXT=" in result.output


def test_record_voice_command_fails_on_empty_capture(tmp_path, monkeypatch):
    out = tmp_path / "voice_sample.wav"
    monkeypatch.setattr(
        "reachy_vec.cli.record_voice.record_sample",
        lambda duration_s, sample_rate: None,
    )

    result = runner.invoke(app, ["record-voice", "--out", str(out)])

    assert result.exit_code != 0
    assert not out.exists()

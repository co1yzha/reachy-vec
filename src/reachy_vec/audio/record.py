"""Fixed-duration mic recording for the voice-clone reference sample.

Distinct from `listen.py` (VAD-segmented utterances + STT): here we want a
clean, continuous block of speech saved to a WAV file. `write_wav` is pure
and unit-tested; `record_sample` owns the mic (injectable for tests).
"""

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000  # matches the qwen-tts reference-sample docs


def _record_blocking(frames: int, sample_rate: int) -> np.ndarray | None:
    import sounddevice as sd

    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].copy()


def record_sample(
    duration_s: float, sample_rate: int = SAMPLE_RATE, record=None
) -> np.ndarray | None:
    """Record `duration_s` of mono audio; None if capture yields nothing."""
    record = record or _record_blocking
    frames = int(sample_rate * duration_s)
    return record(frames, sample_rate)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono 16-bit PCM WAV; creates the parent directory if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

"""Mic capture -> VAD-segmented utterance -> text (faster-whisper).

collect_utterance is pure and unit-tested; MicTranscriber does the device
and model work and is covered by the manual smoke test only.
"""

import logging
from typing import Callable, Iterator, Protocol

import numpy as np

from reachy_vec.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_S = 0.032  # silero-vad native frame for 16 kHz (512 samples)


class Transcriber(Protocol):
    def listen_once(self, timeout_s: float) -> str | None: ...


def collect_utterance(chunks: Iterator, is_speech: Callable, max_silence_chunks: int):
    """Accumulate chunks from first speech until max_silence_chunks of quiet.

    Returns the collected chunk list (speech plus trailing/inner silence),
    or None if the iterator ends before any speech occurs.
    """
    collected: list = []
    silence_run = 0
    for chunk in chunks:
        if is_speech(chunk):
            collected.append(chunk)
            silence_run = 0
        elif collected:
            collected.append(chunk)
            silence_run += 1
            if silence_run >= max_silence_chunks:
                break
    return collected or None


class MicTranscriber:
    """Blocks on the default input device; lazy-loads VAD + whisper models."""

    def __init__(self, model_size: str | None = None, sample_rate: int = SAMPLE_RATE):
        self._model_size = model_size or settings.stt_model
        self._sample_rate = sample_rate
        self._vad = None
        self._whisper = None

    def _load(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(self._model_size, compute_type="int8")

    def listen_once(self, timeout_s: float) -> str | None:
        import sounddevice as sd
        import torch

        self._load()
        chunk_samples = int(self._sample_rate * CHUNK_S)
        max_chunks = int(timeout_s / CHUNK_S)
        max_silence = int(0.8 / CHUNK_S)  # 0.8 s of quiet ends the utterance

        def frames() -> Iterator[np.ndarray]:
            with sd.InputStream(
                samplerate=self._sample_rate, channels=1, dtype="float32"
            ) as stream:
                for _ in range(max_chunks):
                    data, _overflow = stream.read(chunk_samples)
                    yield data[:, 0].copy()

        def is_speech(chunk: np.ndarray) -> bool:
            prob = self._vad(torch.from_numpy(chunk), self._sample_rate).item()
            return prob > 0.5

        collected = collect_utterance(frames(), is_speech, max_silence)
        if collected is None:
            return None
        audio = np.concatenate(collected)
        segments, _info = self._whisper.transcribe(audio, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("heard: %r", text)
        return text or None

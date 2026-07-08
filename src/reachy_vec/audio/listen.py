"""Mic capture -> VAD-segmented utterance -> text.

collect_utterance is pure and unit-tested. _AudioCapture owns the mic + VAD;
MicTranscriber transcribes locally (faster-whisper), OpenAITranscriber via
the OpenAI API (gpt-4o-transcribe). Select with settings.stt_backend.
"""

import io
import logging
import wave
from collections.abc import Callable, Iterator
from typing import Protocol

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


class _AudioCapture:
    """Shared mic + VAD front-end; lazy-loads the VAD model."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._vad = None

    def _load_vad(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()

    def _capture(self, timeout_s: float) -> np.ndarray | None:
        """Record one VAD-segmented utterance; None if silence until timeout."""
        import sounddevice as sd
        import torch

        self._load_vad()
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
        return np.concatenate(collected) if collected else None


class MicTranscriber(_AudioCapture):
    """Local STT: faster-whisper, lazily loaded."""

    def __init__(
        self,
        model_size: str | None = None,
        sample_rate: int = SAMPLE_RATE,
        initial_prompt: str | None = None,
    ):
        super().__init__(sample_rate)
        self._model_size = model_size or settings.stt_model
        self._initial_prompt = initial_prompt
        self._whisper = None

    def _load(self):
        self._load_vad()
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(self._model_size, compute_type="int8")

    def warm_up(self) -> None:
        """Load models and run a throwaway transcription so the first real
        utterance doesn't pay the cold-start cost mid-conversation."""
        self._load()
        silence = np.zeros(self._sample_rate, dtype=np.float32)
        list(self._whisper.transcribe(silence, language="en")[0])

    def listen_once(self, timeout_s: float) -> str | None:
        self._load()
        audio = self._capture(timeout_s)
        if audio is None:
            return None
        segments, _info = self._whisper.transcribe(
            audio, language="en", initial_prompt=self._initial_prompt
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("heard: %r", text)
        return text or None


class OpenAITranscriber(_AudioCapture):
    """Cloud STT: gpt-4o-transcribe. Best accuracy; ~1s network latency."""

    def __init__(self, client, initial_prompt: str | None = None):
        super().__init__()
        self._client = client
        self._initial_prompt = initial_prompt

    def listen_once(self, timeout_s: float) -> str | None:
        audio = self._capture(timeout_s)
        if audio is None:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sample_rate)
            w.writeframes((audio * 32767).astype(np.int16).tobytes())
        buf.seek(0)
        buf.name = "speech.wav"  # the SDK needs a filename hint
        try:
            kwargs = {"model": "gpt-4o-transcribe", "file": buf}
            if self._initial_prompt:
                kwargs["prompt"] = self._initial_prompt
            result = self._client.audio.transcriptions.create(**kwargs)
        except Exception:
            logger.exception("OpenAI transcription failed")
            return None
        text = result.text.strip()
        logger.info("heard (openai): %r", text)
        return text or None


def make_transcriber(client=None, initial_prompt: str | None = None) -> Transcriber:
    if settings.stt_backend == "openai":
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        return OpenAITranscriber(client, initial_prompt=initial_prompt)
    return MicTranscriber(initial_prompt=initial_prompt)

"""Mic capture -> VAD-segmented utterance -> text.

collect_utterance is pure and unit-tested. _AudioCapture owns the mic + VAD;
MicTranscriber transcribes locally (faster-whisper), OpenAITranscriber via
the OpenAI API (gpt-4o-transcribe). Select with settings.stt_backend.
"""

import io
import logging
import threading
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from reachy_vec.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_S = 0.032  # silero-vad native frame for 16 kHz (512 samples)


@dataclass(frozen=True)
class Utterance:
    """One transcribed utterance plus the raw mono 16 kHz float32 capture
    (kept in memory so speaker ID can embed the same audio; never persisted)."""

    text: str
    audio: np.ndarray | None = field(default=None, repr=False)


class Transcriber(Protocol):
    def listen_once(self, timeout_s: float) -> Utterance | None: ...


def collect_utterance(
    chunks: Iterator,
    is_speech: Callable,
    max_silence_chunks: int,
    max_wait_chunks: int | None = None,
    max_utterance_chunks: int | None = None,
):
    """Accumulate chunks from first speech until max_silence_chunks of quiet.

    max_wait_chunks bounds the wait for speech to *start* only — a sentence
    begun late in the listen window is never truncated by the wait budget.
    max_utterance_chunks caps the utterance itself once speech has started.
    Returns the collected chunk list (speech plus trailing/inner silence), or
    None if no speech occurs before the iterator ends or the wait budget runs out.
    """
    collected: list = []
    silence_run = 0
    waited = 0
    for chunk in chunks:
        if not collected:
            if is_speech(chunk):
                collected.append(chunk)
            else:
                waited += 1
                if max_wait_chunks is not None and waited >= max_wait_chunks:
                    return None
            continue
        collected.append(chunk)
        if is_speech(chunk):
            silence_run = 0
        else:
            silence_run += 1
            if silence_run >= max_silence_chunks:
                break
        if max_utterance_chunks is not None and len(collected) >= max_utterance_chunks:
            break
    return collected or None


class AudioSource(Protocol):
    def frames(self, chunk_samples: int) -> Iterator[np.ndarray]:
        """Yield mono float32 frames of length chunk_samples at 16 kHz."""
        ...


class MicSource:
    """Default AudioSource: the Mac's default mic via sounddevice, 16 kHz mono."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._sample_rate = sample_rate

    def frames(self, chunk_samples: int) -> Iterator[np.ndarray]:
        import sounddevice as sd

        with sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype="float32"
        ) as stream:
            while True:
                data, _overflow = stream.read(chunk_samples)
                yield data[:, 0].copy()


class BargeInMonitor:
    """Watch the mic while Reachy speaks; fire after sustained speech.

    Shares the AudioSource seam and silero-VAD. The ~min_speech_s hysteresis
    rejects brief blips, but the robot's own reply IS sustained speech, so an
    optional is_self_voice(audio) check can veto a run (speaker-embedding
    match against the TTS voice); a vetoed run resets and watching continues.
    """

    def __init__(
        self,
        source,
        min_speech_s=0.7,
        sample_rate=SAMPLE_RATE,
        is_speech=None,
        spawn=None,
        is_self_voice=None,
    ):
        self._source = source
        self._min_chunks = max(1, int(min_speech_s / CHUNK_S))
        self._sample_rate = sample_rate
        self._is_speech = is_speech
        self._is_self_voice = is_self_voice
        self._spawn = spawn or _daemon_spawn
        self.fired = False
        self.broken = False
        self._stop = False
        self._on_fire = None
        self._vad = None

    def start(self, on_fire) -> None:
        self.fired = False
        self._stop = False
        self._on_fire = on_fire
        self._spawn(self._watch)

    def stop(self) -> None:
        self._stop = True

    def _watch(self) -> None:
        try:
            is_speech = self._is_speech or self._silero_is_speech()
            chunk_samples = int(self._sample_rate * CHUNK_S)
            run: list[np.ndarray] = []
            frame_iter = self._source.frames(chunk_samples)
            try:
                for frame in frame_iter:
                    if self._stop:
                        return
                    if is_speech(frame):
                        run.append(frame)
                        if len(run) >= self._min_chunks:
                            if self._is_self_voice and self._is_self_voice(
                                np.concatenate(run)
                            ):
                                logger.info("barge-in vetoed: robot's own voice")
                                run = []
                                continue
                            logger.info("barge-in fired: %.1fs of speech over the reply",
                                        len(run) * CHUNK_S)
                            self.fired = True
                            if self._on_fire:
                                self._on_fire()
                            return
                    else:
                        run = []
            finally:
                frame_iter.close()
        except Exception:
            logger.exception("barge-in monitor crashed; disabling for the session")
            self.broken = True

    def _silero_is_speech(self):
        import torch
        from silero_vad import load_silero_vad

        if self._vad is None:
            self._vad = load_silero_vad()

        def is_speech(frame):
            return self._vad(torch.from_numpy(frame), self._sample_rate).item() > 0.5

        return is_speech


def _daemon_spawn(fn):
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def gate_quiet(audio: "np.ndarray | None", min_rms: float) -> "np.ndarray | None":
    """Discard a captured utterance whose level says it isn't speech.

    VAD can trigger on room noise, and near-silent segments make STT
    hallucinate its vocabulary prompt (it once 'heard' demo names nobody
    said). Below min_rms the segment is treated as nothing heard.
    """
    if audio is None:
        return None
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < min_rms:
        logger.info("utterance below RMS gate (%.4f < %.4f) - discarded", rms, min_rms)
        return None
    return audio


def normalize_rms(
    audio: "np.ndarray | None", target_rms: float, max_gain: float = 10.0
) -> "np.ndarray | None":
    """Boost a quiet capture to target_rms before STT (never attenuates).

    The far-field robot mic runs quiet and low-level audio measurably hurts
    transcription accuracy. Gain is capped so silence-adjacent captures don't
    become amplified noise. target_rms <= 0 disables.
    """
    if audio is None or target_rms <= 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 0 or rms >= target_rms:
        return audio
    gain = min(target_rms / rms, max_gain)
    return np.clip(audio * gain, -1.0, 1.0)


class _AudioCapture:
    """Shared VAD front-end over a pluggable AudioSource; lazy-loads the VAD."""

    def __init__(
        self, source: "AudioSource | None" = None, sample_rate: int = SAMPLE_RATE
    ):
        self._sample_rate = sample_rate
        self._source = source or MicSource(sample_rate)
        self._vad = None

    def _load_vad(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()

    def _capture(self, timeout_s: float) -> np.ndarray | None:
        """Record one VAD-segmented utterance; None if silence until timeout.

        timeout_s bounds the wait for speech to start; the utterance itself
        is bounded separately (settings.utterance_max_s), so a sentence begun
        late in the window is never truncated.
        """
        import torch

        self._load_vad()
        chunk_samples = int(self._sample_rate * CHUNK_S)
        max_silence = int(settings.utterance_max_silence_s / CHUNK_S)

        frame_iter = self._source.frames(chunk_samples)

        def is_speech(chunk: np.ndarray) -> bool:
            prob = self._vad(torch.from_numpy(chunk), self._sample_rate).item()
            return prob > 0.5

        try:
            collected = collect_utterance(
                frame_iter,
                is_speech,
                max_silence,
                max_wait_chunks=int(timeout_s / CHUNK_S),
                max_utterance_chunks=int(settings.utterance_max_s / CHUNK_S),
            )
        finally:
            frame_iter.close()
        utterance = np.concatenate(collected) if collected else None
        return gate_quiet(utterance, settings.min_utterance_rms)


class MicTranscriber(_AudioCapture):
    """Local STT: faster-whisper, lazily loaded."""

    def __init__(
        self,
        model_size: str | None = None,
        sample_rate: int = SAMPLE_RATE,
        initial_prompt: str | None = None,
        source: "AudioSource | None" = None,
        hotwords: str | None = None,
    ):
        super().__init__(source=source, sample_rate=sample_rate)
        self._model_size = model_size or settings.stt_model
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords if hotwords is not None else settings.stt_hotwords
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

    def listen_once(self, timeout_s: float) -> Utterance | None:
        self._load()
        audio = self._capture(timeout_s)
        if audio is None:
            return None
        segments, _info = self._whisper.transcribe(
            normalize_rms(audio, settings.stt_target_rms),
            language="en",
            initial_prompt=self._initial_prompt,
            hotwords=self._hotwords or None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("heard: %r", text)
        return Utterance(text=text, audio=audio) if text else None


_stt_tap_counter = 0


def _stt_tap(audio: np.ndarray, sample_rate: int, text: str) -> None:
    """REACHY_VEC_STT_TAP=<dir>: save each captured utterance (raw, before the
    STT loudness boost) plus its transcript, to see what the recogniser got.
    Debug aid; off unless the env var is set."""
    import os

    tap_dir = os.environ.get("REACHY_VEC_STT_TAP")
    if not tap_dir:
        return
    global _stt_tap_counter
    import soundfile as sf

    _stt_tap_counter += 1
    os.makedirs(tap_dir, exist_ok=True)
    name = f"stt_tap_{_stt_tap_counter:03d}.wav"
    sf.write(os.path.join(tap_dir, name), audio, sample_rate)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    with open(os.path.join(tap_dir, "stt_tap.log"), "a") as log:
        log.write(f"{name} {len(audio) / sample_rate:.1f}s rms={rms:.4f} {text!r}\n")


class ParakeetTranscriber(_AudioCapture):
    """Local STT: NVIDIA Parakeet TDT via parakeet-mlx (Apple GPU), lazily loaded.

    More accurate than whisper base.en and far less prone to inventing text
    from noise; it has no hotwords option, so the wake-word list carries the
    "Reachy" mishearings. `transcribe` (16 kHz float32 -> text) is injectable.
    """

    def __init__(
        self,
        model_id: str | None = None,
        sample_rate: int = SAMPLE_RATE,
        source: "AudioSource | None" = None,
        transcribe=None,
    ):
        super().__init__(source=source, sample_rate=sample_rate)
        self._model_id = model_id or settings.parakeet_model
        self._transcribe = transcribe

    def _load(self):
        self._load_vad()
        if self._transcribe is None:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            model = from_pretrained(self._model_id)
            config = model.preprocessor_config
            if config.sample_rate != self._sample_rate:
                raise ValueError(
                    f"parakeet expects {config.sample_rate} Hz audio, capture is "
                    f"{self._sample_rate} Hz"
                )

            def transcribe(audio: np.ndarray) -> str:
                mel = get_logmel(mx.array(audio, dtype=mx.float32), config)  # as load_audio
                return model.generate(mel)[0].text

            self._transcribe = transcribe

    def warm_up(self) -> None:
        self._load()
        self._transcribe(np.zeros(self._sample_rate, dtype=np.float32))

    def listen_once(self, timeout_s: float) -> Utterance | None:
        self._load()
        audio = self._capture(timeout_s)
        if audio is None:
            return None
        text = self._transcribe(normalize_rms(audio, settings.stt_target_rms)).strip()
        _stt_tap(audio, self._sample_rate, text)
        logger.info("heard (parakeet): %r", text)
        return Utterance(text=text, audio=audio) if text else None


class OpenAITranscriber(_AudioCapture):
    """Cloud STT: gpt-4o-transcribe. Best accuracy; ~1s network latency."""

    def __init__(
        self, client, initial_prompt: str | None = None, source: "AudioSource | None" = None
    ):
        super().__init__(source=source)
        self._client = client
        self._initial_prompt = initial_prompt

    def listen_once(self, timeout_s: float) -> Utterance | None:
        audio = self._capture(timeout_s)
        if audio is None:
            return None
        stt_audio = normalize_rms(audio, settings.stt_target_rms)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sample_rate)
            w.writeframes((stt_audio * 32767).astype(np.int16).tobytes())
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
        return Utterance(text=text, audio=audio) if text else None


def make_transcriber(
    client=None, initial_prompt: str | None = None, source: "AudioSource | None" = None
) -> Transcriber:
    if settings.stt_backend == "openai":
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        return OpenAITranscriber(client, initial_prompt=initial_prompt, source=source)
    if settings.stt_backend == "parakeet":
        return ParakeetTranscriber(source=source)
    return MicTranscriber(initial_prompt=initial_prompt, source=source)

"""Text-to-speech behind a pluggable Speaker protocol.

Backends (settings.tts_backend): "say" (macOS built-in, no cloning) and
"qwen-tts" (Qwen3-TTS via mlx-audio, clones settings.voice_sample).
"""

import logging
import subprocess
from pathlib import Path
from typing import Protocol

from reachy_vec.config import settings

logger = logging.getLogger(__name__)

QWEN_TTS_DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def precache(self, texts: list[str]) -> None: ...


class SaySpeaker:
    """macOS `say` via Popen so a reply can be interrupted mid-sentence."""

    def __init__(self, popen=subprocess.Popen):
        self._popen = popen
        self._proc = None

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._proc = self._popen(["say", text])
        self._proc.wait()

    def precache(self, texts: list[str]) -> None:
        pass  # `say` starts instantly; nothing to pre-generate

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


class AfplayPlayer:
    """Blocking Mac playback via an `afplay` subprocess.

    sounddevice services the output device from a Python callback, which
    needs the GIL; under app load (face detection, VAD, MLX) the callback
    arrives late and playback crackles. A subprocess is immune. stop()
    terminates mid-clip for barge-in, like SaySpeaker.
    """

    def __init__(self, popen=subprocess.Popen):
        self._popen = popen
        self._proc = None

    def __call__(self, audio, sample_rate: int) -> None:
        import os
        import tempfile
        import wave

        import numpy as np

        data = (np.clip(np.asarray(audio), -1, 1) * 32767).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f, wave.open(f, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(data.tobytes())
            self._proc = self._popen(["afplay", path])
            self._proc.wait()
        finally:
            os.unlink(path)

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


def _presence_shelf(audio, sample_rate: int, gain_db: float, corner_hz: float):
    """Lift high frequencies with an RBJ high-shelf biquad, for clarity.

    The clone's brightness tracks the (often dull) reference sample; a few dB
    of presence above ~3 kHz makes speech more intelligible. gain_db == 0 is a
    no-op (returns the input untouched, so callers stay byte-exact when off).
    """
    import numpy as np

    if gain_db == 0:
        return audio
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    from scipy.signal import lfilter

    # RBJ Audio EQ Cookbook high-shelf, shelf slope S = 1.
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * corner_hz / sample_rate
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((a_gain + 1 / a_gain) * (1 / 1.0 - 1) + 2)
    two_sqrt_a_alpha = 2 * np.sqrt(a_gain) * alpha

    b0 = a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0)
    b2 = a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_gain + 1) - (a_gain - 1) * cos_w0 + two_sqrt_a_alpha
    a1 = 2 * ((a_gain - 1) - (a_gain + 1) * cos_w0)
    a2 = (a_gain + 1) - (a_gain - 1) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, audio).astype(np.float32)


def _soft_limit(audio, ceiling: float = 0.99):
    """tanh soft-clip peaks that exceed the ceiling, transparent below it.

    A hard clip (np.clip) is audible distortion; tanh rolls the loudest peaks
    off smoothly. Engaged only when something exceeds the ceiling, so ordinary
    content is left byte-exact.
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or float(np.max(np.abs(audio))) <= ceiling:
        return audio
    return (ceiling * np.tanh(audio / ceiling)).astype(np.float32)


def _process_loudness(
    audio, sample_rate: int, target_rms: float, presence_db: float, presence_hz: float
):
    """Make TTS output clear and loud: presence shelf, then boost toward
    target_rms, then soft-limit peaks.

    Qwen's clone output tracks the (often quiet, dull) reference sample. Boost
    is applied before limiting so the sustained body can reach the target while
    the limiter tames transients - louder than a single gain capped by the peak.
    target_rms <= 0 disables the loudness stage; never attenuates.
    """
    import numpy as np

    audio = _presence_shelf(audio, sample_rate, presence_db, presence_hz)
    if target_rms <= 0:
        return audio
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms > 0 and rms < target_rms:
        audio = audio * (target_rms / rms)
    return _soft_limit(audio)


class RobotAudioSink:
    """Play TTS audio through the robot's speaker via mini.media.push_audio_sample.

    push_audio_sample queues and returns immediately, so this blocks for the
    clip's duration — otherwise speak() returns while the robot is still
    talking and the mic opens on its own voice.

    The gstreamer output appsrc silently drops every pushed buffer (a stderr
    warning, no exception) until start_playing() runs — the robot makes no
    sound while the brain logs replies. So start the pipeline once, lazily,
    before the first push.
    """

    def __init__(self, media, sleep=None):
        import time

        self._media = media
        self._sleep = sleep or time.sleep
        self._playing = False

    def __call__(self, audio, sample_rate: int) -> None:
        import numpy as np

        from reachy_vec.audio.resample import to_mono_16k

        if not self._playing:
            self._media.start_playing()
            self._playing = True
        out_rate = self._media.get_output_audio_samplerate()
        data = to_mono_16k(np.asarray(audio, dtype=np.float32), sample_rate, out_rate)
        self._media.push_audio_sample(data)
        self._sleep(len(data) / out_rate)


_tap_counter = 0


def _debug_tap(audio, sample_rate: int) -> None:
    """REACHY_VEC_TTS_TAP=<dir>: dump each clip as played, to tell generation
    noise from playback noise. Debug aid; off unless the env var is set."""
    import os

    tap_dir = os.environ.get("REACHY_VEC_TTS_TAP")
    if not tap_dir:
        return
    global _tap_counter
    import wave

    import numpy as np

    _tap_counter += 1
    path = os.path.join(tap_dir, f"tts_tap_{_tap_counter:03d}.wav")
    data = (np.clip(np.asarray(audio), -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())
    logger.info("TTS tap: %s", path)


class QwenTTSSpeaker:
    """Qwen3-TTS via mlx-audio — clones the voice in `sample_path`.

    Model + clone conditioning load lazily on first speak() and stay cached.
    `generate` / `play` are injectable for tests (no model, no audio device).
    """

    def __init__(
        self,
        sample_path: Path,
        sample_text: str | None = None,
        model_id: str = QWEN_TTS_DEFAULT_MODEL,
        generate=None,
        play=None,
        stop=None,
        target_rms: float = 0.0,
        presence_db: float = 0.0,
        presence_hz: float = 3200.0,
    ):
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_id = model_id
        self._target_rms = target_rms
        self._presence_db = presence_db
        self._presence_hz = presence_hz
        self._generate = generate
        if play is None:
            player = AfplayPlayer()
            play, stop = player, stop or player.stop
        self._play = play
        self._stop = stop
        self._cache: dict[str, tuple] = {}  # text -> processed (audio, sample_rate)

    def _synthesize(self, text: str):
        audio, sample_rate = self._ensure_generate()(text)
        audio = _process_loudness(
            audio,
            sample_rate,
            self._target_rms,
            self._presence_db,
            self._presence_hz,
        )
        return audio, sample_rate

    def precache(self, texts: list[str]) -> None:
        """Pre-generate short fixed lines (acknowledgement fillers) so they
        play instantly; a line that fails to synthesize is skipped."""
        for text in texts:
            text = text.strip()
            if not text or text in self._cache:
                continue
            try:
                self._cache[text] = self._synthesize(text)
            except Exception:
                logger.warning("could not pre-generate %r; it will synthesize live", text)

    def _ensure_generate(self):
        if self._generate is None:
            import numpy as np
            from mlx_audio.tts.utils import load_model

            model = load_model(self._model_id)

            def generate(text: str):
                kwargs = {"text": text, "ref_audio": str(self._sample_path)}
                if self._sample_text:
                    kwargs["ref_text"] = self._sample_text
                results = list(model.generate(**kwargs))
                audio = np.concatenate([np.asarray(r.audio) for r in results])
                return audio, results[0].sample_rate

            self._generate = generate
        return self._generate

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            audio, sample_rate = self._cache.get(text) or self._synthesize(text)
            _debug_tap(audio, sample_rate)
            self._play(audio, sample_rate)
        except Exception:
            logger.exception("TTS synthesis failed; skipping sentence")

    def stop(self) -> None:
        if self._stop is not None:
            self._stop()


def make_speaker(media=None) -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        if media is not None:
            logger.warning(
                "tts_backend='say' has no on-robot output yet; playing through the "
                "Mac speaker. Use 'qwen-tts' for robot audio."
            )
        return SaySpeaker()
    if backend == "qwen-tts":
        sample = settings.voice_sample
        if sample is None or not Path(sample).is_file():
            raise ValueError(
                "tts_backend=qwen-tts needs REACHY_VEC_VOICE_SAMPLE pointing to "
                "a short WAV of the voice to clone (see .env.example)"
            )
        return QwenTTSSpeaker(
            sample_path=Path(sample),
            sample_text=settings.voice_sample_text,
            model_id=settings.tts_model or QWEN_TTS_DEFAULT_MODEL,
            play=RobotAudioSink(media) if media is not None else None,
            target_rms=settings.tts_target_rms,
            presence_db=settings.tts_presence_db,
            presence_hz=settings.tts_presence_hz,
        )
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired - use 'say' or 'qwen-tts'"
    )

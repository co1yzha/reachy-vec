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


def _normalize_loudness(audio, target_rms: float):
    """Boost quiet TTS output toward target_rms, gain-limited by the peak.

    Qwen's output level tracks the (often quietly recorded) reference sample.
    Unlike listen.normalize_rms this never clips - hard clipping is audible
    distortion on playback. target_rms <= 0 disables; never attenuates.
    """
    import numpy as np

    if target_rms <= 0:
        return audio
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 0 or rms >= target_rms:
        return audio
    peak = float(np.max(np.abs(audio)))
    gain = min(target_rms / rms, 0.99 / peak)
    return audio * gain


class RobotAudioSink:
    """Play TTS audio through the robot's speaker via mini.media.push_audio_sample.

    push_audio_sample queues and returns immediately, so this blocks for the
    clip's duration — otherwise speak() returns while the robot is still
    talking and the mic opens on its own voice.
    """

    def __init__(self, media, sleep=None):
        import time

        self._media = media
        self._sleep = sleep or time.sleep

    def __call__(self, audio, sample_rate: int) -> None:
        import numpy as np

        from reachy_vec.audio.resample import to_mono_16k

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
    ):
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_id = model_id
        self._target_rms = target_rms
        self._generate = generate
        if play is None:
            player = AfplayPlayer()
            play, stop = player, stop or player.stop
        self._play = play
        self._stop = stop

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
            audio, sample_rate = self._ensure_generate()(text)
            audio = _normalize_loudness(audio, self._target_rms)
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
            model_id=settings.tts_model,
            play=RobotAudioSink(media) if media is not None else None,
            target_rms=settings.tts_target_rms,
        )
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired - use 'say' or 'qwen-tts'"
    )

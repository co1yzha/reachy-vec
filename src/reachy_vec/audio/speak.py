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


class SaySpeaker:
    """macOS `say` — dev/debug backend, blocks until speech finishes."""

    def __init__(self, run=subprocess.run):
        self._run = run

    def speak(self, text: str) -> None:
        text = text.strip()
        if text:
            self._run(["say", text], check=False)


def _play_blocking(audio, sample_rate: int) -> None:
    import numpy as np
    import sounddevice as sd

    sd.play(np.asarray(audio, dtype=np.float32), sample_rate)
    sd.wait()


class RobotAudioSink:
    """Play TTS audio through the robot's speaker via mini.media.push_audio_sample."""

    def __init__(self, media):
        self._media = media

    def __call__(self, audio, sample_rate: int) -> None:
        import numpy as np

        from reachy_vec.audio.resample import to_mono_16k

        out_rate = self._media.get_output_audio_samplerate()
        data = to_mono_16k(np.asarray(audio, dtype=np.float32), sample_rate, out_rate)
        self._media.push_audio_sample(data)


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
    ):
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_id = model_id
        self._generate = generate
        self._play = play or _play_blocking

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
            self._play(audio, sample_rate)
        except Exception:
            logger.exception("TTS synthesis failed; skipping sentence")


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
        )
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired - use 'say' or 'qwen-tts'"
    )

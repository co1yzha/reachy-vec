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


def make_speaker() -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        return SaySpeaker()
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired yet - set REACHY_VEC_TTS_BACKEND=say"
    )

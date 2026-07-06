"""Text-to-speech behind a pluggable Speaker protocol.

Backends (settings.tts_backend): "say" (macOS built-in, no cloning) now;
"fish-speech" (voice clone, primary per spec) and "openvoice" (fallback)
land in a later plan and use settings.voice_sample as the clone reference.
"""

import subprocess
from typing import Protocol

from reachy_vec.config import settings


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


def make_speaker() -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        return SaySpeaker()
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired yet - set REACHY_VEC_TTS_BACKEND=say"
    )

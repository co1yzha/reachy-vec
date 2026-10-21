"""Speak a streamed reply without gaps: synthesize ahead while playing.

Cloned-voice TTS takes about as long to generate a sentence as to play it, so
speaking sentence by sentence leaves the robot silent while each one renders.
SpeechPipeline queues sentences in order: one worker synthesizes, another
plays, so sentence N+1 renders while N plays - and say() returns at once, so
the LLM keeps streaming meanwhile. Speakers without a synthesize/play_clip
split (macOS `say`, test fakes) are spoken whole on the play worker.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SpeechPipeline:
    """One reply's speech. say() queues; wait() blocks until all is played;
    stop() (barge-in) drops what's queued and cuts the current line."""

    def __init__(self, speaker):
        self._speaker = speaker
        self._two_stage = hasattr(speaker, "synthesize") and hasattr(speaker, "play_clip")
        self._synth = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-synth")
        self._play = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-play")
        self._last: Future | None = None
        self._stopped = threading.Event()

    def say(self, text: str) -> None:
        if self._stopped.is_set() or not text.strip():
            return
        if self._two_stage:
            clip = self._synth.submit(self._synthesize, text)
            self._last = self._play.submit(self._play_clip, clip)
        else:
            self._last = self._play.submit(self._speak, text)

    def wait(self) -> None:
        if self._last is not None:
            self._last.result()
        self._synth.shutdown(wait=False)
        self._play.shutdown(wait=False)

    def stop(self) -> None:
        self._stopped.set()
        self._speaker.stop()

    def _synthesize(self, text: str):
        if self._stopped.is_set():
            return None
        try:
            return self._speaker.synthesize(text)
        except Exception:
            logger.exception("TTS synthesis failed; skipping %r", text)
            return None

    def _play_clip(self, clip: Future) -> None:
        audio = clip.result()
        if audio is None or self._stopped.is_set():
            return
        try:
            self._speaker.play_clip(audio)
        except Exception:
            logger.exception("playback failed")

    def _speak(self, text: str) -> None:
        if self._stopped.is_set():
            return
        try:
            self._speaker.speak(text)
        except Exception:
            logger.exception("speech failed")

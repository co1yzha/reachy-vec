"""Gentle sway loop while the robot is speaking (Mac-speaker fallback).

When replies play on the robot's own speaker, the SDK's audio-synced
wobble covers this (see make_robot). This module fakes life for the
Mac-speaker path: a background thread loops the 'sway' keyframes until
stopped. Motion is decoration - any body failure just ends the sway.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SpeakingSway:
    """Loops the 'sway' motion on a body between start() and stop()."""

    def __init__(self, body):
        self._body = body
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._body.perform("sway")
            except Exception as exc:
                logger.debug("sway ended on body error: %s", exc)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class SwayingSpeaker:
    """Speaker decorator: sway while each sentence plays; halt on barge-in."""

    def __init__(self, speaker, sway: SpeakingSway):
        self._speaker = speaker
        self._sway = sway

    def speak(self, text: str) -> None:
        self._sway.start()
        try:
            self._speaker.speak(text)
        finally:
            self._sway.stop()

    def stop(self) -> None:
        self._speaker.stop()
        self._sway.stop()

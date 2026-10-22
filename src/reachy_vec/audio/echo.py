"""Echo guard: recognise Reachy's own reply when the mic hears it back.

Off-robot mics (a USB or wireless mic beside the robot) get no echo
cancellation, so keyword barge-in can transcribe the reply itself - "...I'll
stay quiet..." - and stop mid-sentence. EchoGuard wraps the Speaker, keeps the
last few seconds of spoken text, and is_echo() says whether a transcript is
mostly a stretch of it.
"""

import re
import threading
import time
from difflib import SequenceMatcher

MIN_ECHO_WORDS = 3     # shorter transcripts ("stop", "Reachy, wait") are never echoes
ECHO_MATCH_RATIO = 0.6  # fraction of heard words found, in order, in what was said


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def is_echo(heard: str, said: list[str]) -> bool:
    """True if `heard` is mostly an in-order stretch of the `said` sentences."""
    heard_words = _words(heard)
    if len(heard_words) < MIN_ECHO_WORDS:
        return False
    said_words = _words(" ".join(said))
    if not said_words:
        return False
    matcher = SequenceMatcher(None, heard_words, said_words, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(heard_words) >= ECHO_MATCH_RATIO


class EchoGuard:
    """Speaker decorator that remembers what was said in the last window_s."""

    def __init__(self, speaker, *, window_s: float = 30.0, clock=time.monotonic):
        self._speaker = speaker
        self._window_s = window_s
        self._clock = clock
        self._said: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        if hasattr(speaker, "synthesize") and hasattr(speaker, "play_clip"):
            # keep the two-stage split visible to SpeechPipeline
            self.synthesize = self._synthesize
            self.play_clip = speaker.play_clip

    def _remember(self, text: str) -> None:
        with self._lock:
            self._said.append((self._clock(), text))

    def _synthesize(self, text: str):
        self._remember(text)
        return self._speaker.synthesize(text)

    def speak(self, text: str) -> None:
        self._remember(text)
        self._speaker.speak(text)

    def stop(self) -> None:
        self._speaker.stop()

    def precache(self, texts: list[str]) -> None:
        self._speaker.precache(texts)

    def is_echo(self, heard: str) -> bool:
        cutoff = self._clock() - self._window_s
        with self._lock:
            self._said = [(t, s) for t, s in self._said if t >= cutoff]
            said = [s for _, s in self._said]
        return is_echo(heard, said)

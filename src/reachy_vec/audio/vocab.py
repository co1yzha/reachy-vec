"""Post-hoc vocabulary snapping for transcripts.

Biasing STT with a "Vocabulary: <titles>" prompt made gpt-4o-transcribe emit
demo titles verbatim on echo/noise (it hallucinates its prompt). Instead the
transcript is corrected after the fact: a word window that fuzzily matches a
known title is replaced by the canonical title; everything else is untouched.
"""

import logging
import re
from difflib import SequenceMatcher

from reachy_vec.audio.listen import Utterance

logger = logging.getLogger(__name__)

SNAP_CUTOFF = 0.85  # SequenceMatcher ratio needed to snap a window to a title


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def snap_to_vocab(text: str, titles: list[str], cutoff: float = SNAP_CUTOFF) -> str:
    """Replace the best fuzzy-matching word window with its canonical title.

    Only near-matches snap (ratio >= cutoff); unrelated text passes through,
    so stray echo syllables can never *become* a title.
    """
    words = text.split()
    if not words or not titles:
        return text
    best: tuple[float, int, int, str] | None = None
    for title in titles:
        title_norm = _norm(title)
        if not title_norm:
            continue
        title_words = len(title_norm.split())
        for size in range(max(1, title_words - 2), title_words + 3):
            for start in range(0, len(words) - size + 1):
                window = _norm(" ".join(words[start : start + size]))
                ratio = SequenceMatcher(None, window, title_norm).ratio()
                if ratio >= cutoff and (best is None or ratio > best[0]):
                    best = (ratio, start, start + size, title)
    if best is None:
        return text
    ratio, start, end, title = best
    snapped = " ".join(words[:start] + [title] + words[end:])
    if snapped != text:
        logger.info("snapped %r -> %r (%.2f)", text, snapped, ratio)
    return snapped


class VocabSnappingTranscriber:
    """Transcriber wrapper: same protocol, transcripts snapped to the vocab."""

    def __init__(self, inner, titles: list[str], cutoff: float = SNAP_CUTOFF):
        self._inner = inner
        self._titles = titles
        self._cutoff = cutoff

    def listen_once(self, timeout_s: float) -> Utterance | None:
        utterance = self._inner.listen_once(timeout_s)
        if utterance is None:
            return None
        text = snap_to_vocab(utterance.text, self._titles, self._cutoff)
        return Utterance(text=text, audio=utterance.audio)

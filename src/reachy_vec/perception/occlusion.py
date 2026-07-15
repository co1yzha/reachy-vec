"""Camera occlusion detection: covering/poking the lens makes the frame go
near-black. Pure numpy on frames sight() already reads - no models.

Fires once per cover (bright -> sustained dark transition), stays quiet while
held covered, re-arms only after brightness recovers, and honors a cooldown.
"""

import logging
import time

import numpy as np

from reachy_vec.config import settings

logger = logging.getLogger(__name__)


class OcclusionDetector:
    def __init__(
        self,
        dark_floor: float | None = None,
        drop_ratio: float | None = None,
        polls: int | None = None,
        cooldown_s: float | None = None,
        clock=time.time,
    ):
        self._dark_floor = dark_floor if dark_floor is not None else settings.occlusion_dark_floor
        self._drop_ratio = drop_ratio if drop_ratio is not None else settings.occlusion_drop_ratio
        self._polls = polls if polls is not None else settings.occlusion_polls
        self._cooldown_s = cooldown_s if cooldown_s is not None else settings.occlusion_cooldown_s
        self._clock = clock
        self._baseline: float | None = None  # EMA of unoccluded brightness
        self._dark_run = 0
        self._armed = True
        self._last_fired_at: float | None = None
        self._fired = False  # latched until consume()

    def update(self, frame) -> bool:
        """Feed one frame; True exactly once per bright->dark transition."""
        if frame is None:
            return False  # camera hiccup: neither dark nor a recovery
        brightness = float(np.asarray(frame).mean())
        dark = (
            self._baseline is not None
            and brightness < self._dark_floor
            and brightness < self._baseline * self._drop_ratio
        )
        if not dark:
            if brightness >= self._dark_floor:  # only bright frames teach the baseline
                self._baseline = (
                    brightness
                    if self._baseline is None
                    else 0.9 * self._baseline + 0.1 * brightness
                )
                self._armed = True
            self._dark_run = 0
            return False
        self._dark_run += 1
        if not self._armed or self._dark_run < self._polls:
            return False
        now = self._clock()
        if self._last_fired_at is not None and now - self._last_fired_at < self._cooldown_s:
            return False
        self._armed = False
        self._last_fired_at = now
        self._fired = True
        logger.info("camera occluded (brightness %.1f, baseline %.1f)", brightness, self._baseline)
        return True

    def consume(self) -> bool:
        """Read-and-clear the fired latch (polled by the Oracle loop)."""
        fired, self._fired = self._fired, False
        return fired

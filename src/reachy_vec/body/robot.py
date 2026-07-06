"""Body implementations: real robot/sim via the SDK, or a logging no-op."""

import logging
from typing import Protocol

from reachy_vec.body.motions import MOTIONS

logger = logging.getLogger(__name__)


class Body(Protocol):
    def perform(self, motion: str) -> None: ...


class NullBody:
    """Used when no daemon is reachable; motions become logged no-ops."""

    def perform(self, motion: str) -> None:
        logger.debug("NullBody: skipping motion %r", motion)


class RobotBody:
    """Plays keyframes on a connected ReachyMini (sim or real)."""

    def __init__(self, mini):
        self._mini = mini

    def perform(self, motion: str) -> None:
        frames = MOTIONS.get(motion)
        if frames is None:
            logger.warning("Unknown motion %r", motion)
            return
        from reachy_mini.utils import create_head_pose

        for kf in frames:
            self._mini.goto_target(
                head=create_head_pose(**kf.head),
                antennas=list(kf.antennas),
                duration=kf.duration,
            )


def make_body() -> Body:
    """Connect to the daemon if possible; otherwise degrade to NullBody."""
    try:
        from reachy_mini import ReachyMini

        mini = ReachyMini(media_backend="no_media")
        return RobotBody(mini)
    except Exception as exc:  # daemon down, robot absent, etc.
        logger.warning("No robot/daemon available (%s); running body-less.", exc)
        return NullBody()

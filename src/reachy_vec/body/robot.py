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
        if motion == "sleep":
            self._mini.goto_sleep()
            return
        if motion == "wake":
            self._mini.wake_up()
            return
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


def make_robot(with_media: bool = False, connect=None) -> tuple[Body, object | None]:
    """Connect to the daemon; optionally acquire camera+mic+speaker media.

    Returns (body, media). `media` is mini.media when with_media and the
    connection succeed, else None. Any failure degrades to (NullBody(), None).
    Registers an atexit cleanup: ReachyMini keeps non-daemon threads alive,
    which would otherwise hang interpreter shutdown. `connect` is injectable
    for tests.
    """
    try:
        import atexit

        if connect is None:
            from reachy_mini import ReachyMini

            def connect(**kw):
                return ReachyMini(**kw)

        backend = "default" if with_media else "no_media"
        mini = connect(media_backend=backend)
        if with_media:
            mini.acquire_media()

            def _cleanup():
                try:
                    mini.release_media()
                finally:
                    mini.client.disconnect()

            atexit.register(_cleanup)
            return RobotBody(mini), mini.media
        atexit.register(mini.client.disconnect)
        return RobotBody(mini), None
    except Exception as exc:  # daemon down, robot absent, etc.
        logger.warning("No robot/daemon available (%s); running body-less.", exc)
        return NullBody(), None


def make_body() -> Body:
    """Body only (no media); back-compat wrapper over make_robot."""
    return make_robot(with_media=False)[0]

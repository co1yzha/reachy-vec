"""Body implementations: real robot/sim via the SDK, or a logging no-op."""

import logging
from collections.abc import Callable
from typing import Protocol

from reachy_vec.body.motions import MOTIONS
from reachy_vec.config import settings

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


class ReconnectingBody:
    """Wraps a Body; rebuilds its connection after a transient drop, and
    degrades to a silent no-op (announcing once) after max_attempts failures.

    Media is NOT re-acquired here (camera/mic soft-degrade independently);
    this only keeps motions alive across a daemon/WiFi blip.
    """

    def __init__(
        self,
        connect_body: "Callable[[], Body]",
        max_attempts: int = 3,
        announce: "Callable[[str], None] | None" = None,
    ):
        self._connect_body = connect_body
        self._max_attempts = max_attempts
        self._announce = announce or (lambda _msg: None)
        self._inner: Body | None = None
        self._failures = 0
        self._dead = False

    def perform(self, motion: str) -> None:
        if self._dead:
            return
        try:
            if self._inner is None:
                self._inner = self._connect_body()
            self._inner.perform(motion)
            self._failures = 0
        except (ConnectionError, TimeoutError) as exc:
            self._inner = None
            self._failures += 1
            logger.warning(
                "Body command %r failed (%s); reconnect attempt %d/%d.",
                motion,
                exc,
                self._failures,
                self._max_attempts,
            )
            if self._failures >= self._max_attempts:
                self._dead = True
                self._announce(
                    "I've lost connection to my body, but I can still hear you."
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
        kwargs = {"media_backend": backend}
        if settings.robot_host:
            kwargs.update(
                host=settings.robot_host,
                port=settings.robot_port,
                connection_mode="network",
            )
        mini = connect(**kwargs)
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

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

    def fix(self) -> str:
        return "No robot is connected, so I'm running without a body right now."


class RobotBody:
    """Plays keyframes on a connected ReachyMini (sim or real)."""

    def __init__(self, mini, motor_status: "Callable[[], str | None] | None" = None):
        self._mini = mini
        # daemon's motor mode ("enabled"/"disabled"/...) or None if unknown
        self._motor_status = motor_status or (lambda: None)

    def enable_motors(self) -> None:
        """Motors come up disabled after a daemon (re)start; with them off every
        motion "succeeds" but nothing moves, and the head stays down."""
        self._mini.enable_motors()

    def perform(self, motion: str) -> None:
        if motion == "sleep":
            self._mini.goto_sleep()
            return
        if motion == "wake":
            self.enable_motors()
            self._mini.wake_up()
            return
        frames = MOTIONS.get(motion)
        if frames is None:
            logger.warning("Unknown motion %r", motion)
            return
        from reachy_mini.utils import create_head_pose

        for kf in frames:
            kwargs = dict(
                head=create_head_pose(**kf.head),
                antennas=list(kf.antennas),
                duration=kf.duration,
            )
            if kf.body_yaw is not None:
                kwargs["body_yaw"] = kf.body_yaw
            self._mini.goto_target(**kwargs)


    def fix(self) -> str:
        """Self-repair on request ("get your head up"): motors on, head up,
        then report what changed. Never restarts the daemon (that would cut
        the live mic/speaker)."""
        before = self._motor_status()
        try:
            self.enable_motors()
            self._mini.wake_up()
        except Exception as exc:
            logger.warning("body fix failed: %s", exc)
            return ("I tried, but the robot isn't responding - restart its daemon or "
                    "power-cycle it.")
        after = self._motor_status()
        if after not in (None, "enabled"):
            return (f"My motors are still off ({after}) - restart the robot daemon or "
                    "power-cycle me.")
        if before == "disabled":
            return "My motors were off - I've switched them on and lifted my head."
        if before == "enabled":
            return "My motors were already on - I've lifted my head again."
        return "I've switched my motors on and lifted my head."


def _daemon_motor_status() -> str | None:
    """Motor mode from the daemon REST API (network mode), or None."""
    import json
    import urllib.request

    host = settings.robot_host or "localhost"
    try:
        url = f"http://{host}:{settings.robot_port}/api/motors/status"
        with urllib.request.urlopen(url, timeout=3) as response:
            return json.loads(response.read()).get("mode")
    except Exception:
        return None


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
        initial: "Body | None" = None,
    ):
        self._connect_body = connect_body
        self._max_attempts = max_attempts
        self._announce = announce or (lambda _msg: None)
        # Seed with the already-connected body: a fresh no_media connect at
        # startup would RELEASE the daemon's media and kill the robot
        # camera/mic stream. Only re-dial after an actual failure.
        self._inner: Body | None = initial
        self._failures = 0
        self._dead = False

    def fix(self) -> str:
        if self._dead or self._inner is None:
            try:
                self.perform("idle")          # re-dial through the normal path
            except Exception:
                pass
        inner = self._inner
        if inner is None or not hasattr(inner, "fix"):
            return "I can't reach my body right now - check the robot is on and its daemon."
        return inner.fix()

    def perform(self, motion: str) -> None:
        if self._dead:
            return
        try:
            if self._inner is None:
                inner = self._connect_body()
                if isinstance(inner, NullBody):
                    # make_robot degrades to NullBody instead of raising; a
                    # silent no-op body must not pass for a live reconnect.
                    raise ConnectionError("daemon still unreachable")
                self._inner = inner
            self._inner.perform(motion)
            self._failures = 0
        except TimeoutError as exc:
            # Daemon busy (e.g. its own choreography overran the SDK's
            # duration+1s wait) - not gone. Re-dialing would release the
            # robot's media stream, so just skip this motion.
            logger.warning("Body command %r timed out (%s); skipped.", motion, exc)
        except ConnectionError as exc:
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


def _rest_acquire_media() -> None:
    """Ask the daemon to start its media pipeline before the SDK connects.

    On daemon 1.8.0 the WebRTC signalling server only starts after
    /api/media/acquire, and the SDK media client needs it DURING
    construction - so the REST call must come first.
    """
    import time
    import urllib.request

    host = settings.robot_host or "localhost"
    req = urllib.request.Request(
        f"http://{host}:{settings.robot_port}/api/media/acquire", method="POST"
    )
    urllib.request.urlopen(req, timeout=5).close()
    time.sleep(2.0)  # let the signalling server come up


def _connected_body(mini) -> "RobotBody":
    body = RobotBody(mini, motor_status=_daemon_motor_status)
    try:
        body.enable_motors()  # off after any daemon restart: the head wouldn't lift
    except Exception as exc:
        logger.warning("could not enable motors (%s); continuing.", exc)
    return body


def make_robot(
    with_media: bool = False, connect=None, pre_acquire=None
) -> tuple[Body, object | None]:
    """Connect to the daemon; optionally acquire camera+mic+speaker media.

    Returns (body, media). `media` is mini.media when with_media and the
    connection succeed, else None. Any failure degrades to (NullBody(), None).
    Registers an atexit cleanup: ReachyMini keeps non-daemon threads alive,
    which would otherwise hang interpreter shutdown. `connect` and
    `pre_acquire` are injectable for tests.
    """
    try:
        import atexit

        if connect is None:
            from reachy_mini import ReachyMini

            def connect(**kw):
                return ReachyMini(**kw)

        if with_media:
            try:
                (pre_acquire or _rest_acquire_media)()
            except Exception as exc:
                logger.warning("media pre-acquire failed (%s); continuing.", exc)

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
            if settings.speech_wobble:
                try:
                    mini.enable_wobbling()
                except Exception as exc:
                    logger.warning(
                        "could not enable speech wobble (%s); continuing.", exc
                    )

            def _cleanup():
                try:
                    mini.release_media()
                finally:
                    mini.client.disconnect()

            atexit.register(_cleanup)
            return _connected_body(mini), mini.media
        atexit.register(mini.client.disconnect)
        return _connected_body(mini), None
    except Exception as exc:  # daemon down, robot absent, etc.
        logger.warning("No robot/daemon available (%s); running body-less.", exc)
        return NullBody(), None


def make_body() -> Body:
    """Body only (no media); back-compat wrapper over make_robot."""
    return make_robot(with_media=False)[0]

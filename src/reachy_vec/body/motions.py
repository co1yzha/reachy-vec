"""Expressive motion primitives as pure keyframe data.

Head kwargs feed reachy_mini.utils.create_head_pose (degrees, mm=False);
antennas are (left, right) in radians; duration in seconds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Keyframe:
    head: dict[str, float]
    antennas: tuple[float, float]
    duration: float


NEUTRAL = Keyframe(head={}, antennas=(0.0, 0.0), duration=0.4)

MOTIONS: dict[str, list[Keyframe]] = {
    "greet": [
        Keyframe(head={"pitch": -15, "yaw": 10}, antennas=(0.6, -0.6), duration=0.4),
        Keyframe(head={"pitch": -15, "yaw": -10}, antennas=(-0.6, 0.6), duration=0.4),
        NEUTRAL,
    ],
    "nod": [
        Keyframe(head={"pitch": 15}, antennas=(0.0, 0.0), duration=0.3),
        NEUTRAL,
    ],
    "listen": [
        Keyframe(head={"pitch": -8, "roll": 6}, antennas=(0.3, 0.3), duration=0.5),
    ],
    "idle": [
        Keyframe(head={"yaw": 6}, antennas=(0.1, 0.1), duration=1.2),
        Keyframe(head={"yaw": -6}, antennas=(0.1, 0.1), duration=1.2),
        NEUTRAL,
    ],
    "acknowledge": [
        Keyframe(head={"yaw": 12}, antennas=(0.5, 0.5), duration=0.3),
        NEUTRAL,
    ],
    "goodbye": [
        Keyframe(head={"pitch": 18}, antennas=(-0.8, -0.8), duration=0.6),
        NEUTRAL,
    ],
    "look": [
        Keyframe(head={"pitch": -6, "yaw": 8}, antennas=(0.4, 0.4), duration=0.35),
        Keyframe(head={"pitch": -6, "yaw": -8}, antennas=(0.4, 0.4), duration=0.35),
        NEUTRAL,
    ],
    "pose": [
        Keyframe(head={"pitch": -10}, antennas=(0.7, 0.7), duration=0.4),
        Keyframe(head={"pitch": -10, "roll": 4}, antennas=(0.7, 0.7), duration=0.4),
        NEUTRAL,
    ],
    "wakeup": [
        # stretch up, antennas high
        Keyframe(head={"pitch": -20}, antennas=(0.9, 0.9), duration=0.8),
        # look left, then right — "who's there?"
        Keyframe(head={"pitch": -10, "yaw": 25}, antennas=(0.6, -0.6), duration=0.6),
        Keyframe(head={"pitch": -10, "yaw": -25}, antennas=(-0.6, 0.6), duration=0.6),
        # quick antenna wiggle
        Keyframe(head={"pitch": -5}, antennas=(0.8, -0.8), duration=0.3),
        Keyframe(head={"pitch": -5}, antennas=(-0.8, 0.8), duration=0.3),
        NEUTRAL,
    ],
}

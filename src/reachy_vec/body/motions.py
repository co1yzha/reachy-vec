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
}

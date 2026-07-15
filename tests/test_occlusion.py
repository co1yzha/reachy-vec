"""Brightness-drop occlusion detector: covering the camera makes the frame
go near-black. Fires once per cover, re-arms on recovery, honors cooldown."""

import numpy as np

from reachy_vec.perception.occlusion import OcclusionDetector

BRIGHT = np.full((48, 64, 3), 120, dtype=np.uint8)
DARK = np.full((48, 64, 3), 5, dtype=np.uint8)


def make_detector(now, **kw):
    kw.setdefault("dark_floor", 28.0)
    kw.setdefault("drop_ratio", 0.3)
    kw.setdefault("polls", 2)
    kw.setdefault("cooldown_s", 10.0)
    return OcclusionDetector(clock=lambda: now["t"], **kw)


def test_fires_after_sustained_darkness():
    now = {"t": 0.0}
    det = make_detector(now)
    assert det.update(BRIGHT) is False        # builds baseline
    assert det.update(DARK) is False          # 1 dark poll: debounced
    assert det.update(DARK) is True           # 2nd dark poll: fires
    assert det.consume() is True
    assert det.consume() is False             # read-and-clear


def test_stays_quiet_while_held_covered():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    det.update(DARK)
    assert det.update(DARK) is True
    assert det.update(DARK) is False          # disarmed until recovery
    assert det.update(DARK) is False


def test_rearms_after_recovery_and_cooldown():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    det.update(DARK)
    assert det.update(DARK) is True
    det.update(BRIGHT)                        # recovery re-arms
    now["t"] = 5.0                            # still inside cooldown
    det.update(DARK)
    assert det.update(DARK) is False
    det.update(BRIGHT)
    now["t"] = 20.0                           # cooldown expired
    det.update(DARK)
    assert det.update(DARK) is True


def test_dark_start_never_fires_without_a_bright_baseline():
    now = {"t": 0.0}
    det = make_detector(now)
    for _ in range(5):
        assert det.update(DARK) is False      # robot may boot covered/at night


def test_none_frames_are_ignored():
    now = {"t": 0.0}
    det = make_detector(now)
    det.update(BRIGHT)
    assert det.update(None) is False
    det.update(DARK)
    assert det.update(DARK) is True


def test_defaults_come_from_settings():
    det = OcclusionDetector()
    assert det._dark_floor == 28.0
    assert det._polls == 2

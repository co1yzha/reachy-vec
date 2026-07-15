import time

from reachy_vec.body.sway import SpeakingSway
from tests.test_body import FlakyBody


class SlowBody:
    """Records motions; each perform takes a beat, like real keyframes."""

    def __init__(self):
        self.motions: list[str] = []

    def perform(self, motion: str) -> None:
        self.motions.append(motion)
        time.sleep(0.01)


def _wait_for_motions(body, deadline_s=2.0):
    deadline = time.time() + deadline_s
    while not body.motions and time.time() < deadline:
        time.sleep(0.005)


def test_speaking_sway_performs_only_between_start_and_stop():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.start()
    _wait_for_motions(body)
    sway.stop()
    count = len(body.motions)
    assert count >= 1
    assert set(body.motions) == {"sway"}
    time.sleep(0.05)
    assert len(body.motions) == count  # nothing after stop (thread joined)


def test_speaking_sway_start_and_stop_are_idempotent():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.stop()  # stop before start: no-op
    sway.start()
    sway.start()  # second start: no second thread
    _wait_for_motions(body)
    sway.stop()
    sway.stop()  # double stop: no error


def test_speaking_sway_ends_quietly_when_body_raises():
    sway = SpeakingSway(FlakyBody(fail=99))  # every perform raises
    sway.start()
    time.sleep(0.05)
    sway.stop()  # thread already dead from the exception; join is clean

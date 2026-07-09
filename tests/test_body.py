from reachy_vec.body.motions import MOTIONS, Keyframe
from reachy_vec.body.robot import Body, NullBody, RobotBody, make_robot

EXPECTED = {"greet", "nod", "listen", "idle", "acknowledge", "goodbye"}


def test_all_motions_defined_and_well_formed():
    assert set(MOTIONS) == EXPECTED
    for name, frames in MOTIONS.items():
        assert frames, name
        for kf in frames:
            assert isinstance(kf, Keyframe)
            assert kf.duration > 0
            assert len(kf.antennas) == 2
            assert set(kf.head) <= {"x", "y", "z", "roll", "pitch", "yaw"}


def test_null_body_is_silent_noop():
    body: Body = NullBody()
    body.perform("greet")  # must not raise
    body.perform("nonexistent")  # unknown motions are ignored, not fatal


class RecordingMini:
    def __init__(self):
        self.calls = []
        self.modes: list[str] = []

    def goto_target(self, head=None, antennas=None, duration=0.5):
        self.calls.append((head is not None, tuple(antennas), duration))

    def goto_sleep(self):
        self.modes.append("sleep")

    def wake_up(self):
        self.modes.append("wake")


def test_robot_body_sleep_and_wake_use_sdk_modes():
    mini = RecordingMini()
    body = RobotBody(mini)
    body.perform("sleep")
    body.perform("wake")
    assert mini.modes == ["sleep", "wake"]
    assert mini.calls == []  # no keyframes for mode changes


def test_robot_body_plays_each_keyframe():
    mini = RecordingMini()
    body = RobotBody(mini)
    body.perform("nod")
    assert len(mini.calls) == len(MOTIONS["nod"])
    assert all(duration > 0 for _, _, duration in mini.calls)


def test_robot_body_ignores_unknown_motion():
    mini = RecordingMini()
    RobotBody(mini).perform("moonwalk")
    assert mini.calls == []


class FakeMini:
    def __init__(self):
        self.acquired = self.released = False
        self.media = object()

        class _Client:
            def disconnect(self_):
                pass

        self.client = _Client()

    def acquire_media(self):
        self.acquired = True

    def release_media(self):
        self.released = True


def test_make_robot_with_media_acquires_and_returns_media():
    mini = FakeMini()
    body, media = make_robot(with_media=True, connect=lambda **kw: mini)
    assert isinstance(body, RobotBody)
    assert media is mini.media
    assert mini.acquired is True


def test_make_robot_degrades_to_nullbody_on_connect_failure():
    def boom(**kw):
        raise RuntimeError("no daemon")

    body, media = make_robot(with_media=True, connect=boom)
    assert isinstance(body, NullBody)
    assert media is None

from reachy_vec.body.motions import MOTIONS, Keyframe
from reachy_vec.body.robot import Body, NullBody, RobotBody, make_robot
from reachy_vec.config import settings as _settings

EXPECTED = {"greet", "nod", "listen", "idle", "acknowledge", "goodbye", "look", "pose", "wakeup"}


def test_all_motions_defined_and_well_formed():
    assert set(MOTIONS) == EXPECTED
    for name, frames in MOTIONS.items():
        assert frames, name
        for kf in frames:
            assert isinstance(kf, Keyframe)
            assert kf.duration > 0
            assert len(kf.antennas) == 2
            assert set(kf.head) <= {"x", "y", "z", "roll", "pitch", "yaw"}


def test_look_and_pose_motions_exist_and_are_valid():
    from reachy_vec.body.motions import MOTIONS, Keyframe

    for name in ("look", "pose"):
        frames = MOTIONS[name]
        assert frames and all(isinstance(kf, Keyframe) for kf in frames)
        assert all(kf.duration > 0 for kf in frames)
        assert frames[-1].head == {} and frames[-1].antennas == (0.0, 0.0)  # ends neutral


def test_wakeup_motion_is_long_noticeable_and_ends_neutral():
    frames = MOTIONS["wakeup"]
    assert sum(kf.duration for kf in frames) >= 3.0          # noticeable, not a twitch
    assert any(kf.head.get("yaw", 0) > 0 for kf in frames)   # looks left...
    assert any(kf.head.get("yaw", 0) < 0 for kf in frames)   # ...and right
    assert frames[-1].head == {} and frames[-1].antennas == (0.0, 0.0)  # ends neutral


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


def test_make_robot_preacquires_media_over_rest_before_connecting():
    """Daemon 1.8.0 only starts the WebRTC signalling server after
    /api/media/acquire; the SDK media client needs it DURING construction,
    so the REST acquire must happen first (live incident, 2026-07-15)."""
    calls = []

    def connect(**kw):
        calls.append("connect")
        return FakeMini()

    make_robot(
        with_media=True,
        connect=connect,
        pre_acquire=lambda: calls.append("rest-acquire"),
    )
    assert calls == ["rest-acquire", "connect"]


def test_make_robot_skips_preacquire_without_media():
    calls = []
    make_robot(
        with_media=False,
        connect=lambda **kw: FakeMini(),
        pre_acquire=lambda: calls.append("rest-acquire"),
    )
    assert calls == []


def test_make_robot_preacquire_failure_is_not_fatal():
    def boom_acquire():
        raise OSError("connection refused")

    body, media = make_robot(
        with_media=True, connect=lambda **kw: FakeMini(), pre_acquire=boom_acquire
    )
    assert not isinstance(body, NullBody)  # still connected; SDK may cope
    assert media is not None


def test_make_robot_uses_network_mode_when_robot_host_set(monkeypatch):
    monkeypatch.setattr(_settings, "robot_host", "reachy.local")
    monkeypatch.setattr(_settings, "robot_port", 8123)
    captured = {}

    def connect(**kw):
        captured.update(kw)
        return FakeMini()

    make_robot(with_media=False, connect=connect)
    assert captured["connection_mode"] == "network"
    assert captured["host"] == "reachy.local"
    assert captured["port"] == 8123


def test_make_robot_local_when_no_robot_host(monkeypatch):
    monkeypatch.setattr(_settings, "robot_host", None)
    captured = {}

    def connect(**kw):
        captured.update(kw)
        return FakeMini()

    make_robot(with_media=False, connect=connect)
    assert "connection_mode" not in captured  # SDK default 'auto'
    assert "host" not in captured


class FlakyBody:
    """RobotBody stand-in: raises ConnectionError for the first `fail` performs."""

    def __init__(self, fail=0):
        self._fail = fail
        self.motions: list[str] = []

    def perform(self, motion):
        if self._fail > 0:
            self._fail -= 1
            raise ConnectionError("Lost connection with the server.")
        self.motions.append(motion)


def test_reconnecting_body_recovers_silently_from_a_blip():
    from reachy_vec.body.robot import ReconnectingBody

    # first inner body fails once; next connect yields a healthy body
    bodies = iter([FlakyBody(fail=1), FlakyBody(fail=0)])
    said: list[str] = []
    body = ReconnectingBody(
        connect_body=lambda: next(bodies), max_attempts=3, announce=said.append
    )
    body.perform("greet")  # inner #1 raises -> dropped, no announce
    body.perform("nod")    # reconnects to inner #2 -> records "nod"
    assert said == []      # transient blip stays silent


def test_reconnecting_body_gives_up_and_announces_once():
    from reachy_vec.body.robot import ReconnectingBody

    def always_fails():
        return FlakyBody(fail=99)

    said: list[str] = []
    body = ReconnectingBody(
        connect_body=always_fails, max_attempts=3, announce=said.append
    )
    for _ in range(5):
        body.perform("nod")  # never raises out
    assert len(said) == 1              # announced exactly once
    assert "body" in said[0].lower()


def test_reconnecting_body_treats_nullbody_reconnect_as_failure():
    """Production wires connect_body=make_robot, which degrades to NullBody
    instead of raising. A NullBody must count as a failed reconnect - not
    silently absorb motions forever - so a real body takes over when the
    daemon returns (live incident, 2026-07-15)."""
    from reachy_vec.body.robot import NullBody, ReconnectingBody

    healthy = FlakyBody(fail=0)
    # drop: first body fails once; reconnect #1 finds the daemon still down
    # (NullBody); reconnect #2 finds it back.
    bodies = iter([FlakyBody(fail=1), NullBody(), healthy])
    said: list[str] = []
    body = ReconnectingBody(
        connect_body=lambda: next(bodies), max_attempts=3, announce=said.append
    )
    body.perform("greet")   # raises inside -> failure 1
    body.perform("nod")     # NullBody reconnect -> must be failure 2, not success
    body.perform("look")    # daemon back -> healthy body performs
    assert healthy.motions == ["look"]
    assert said == []       # transient: recovered before max_attempts


def test_reconnecting_body_announces_when_reconnect_only_finds_nullbody():
    from reachy_vec.body.robot import NullBody, ReconnectingBody

    said: list[str] = []
    body = ReconnectingBody(
        connect_body=lambda: NullBody(), max_attempts=3, announce=said.append
    )
    for _ in range(5):
        body.perform("nod")
    assert len(said) == 1  # daemon never returns -> gives up audibly


def test_reconnecting_body_is_noop_after_death():
    from reachy_vec.body.robot import ReconnectingBody

    healthy = FlakyBody(fail=0)
    bodies = iter([FlakyBody(fail=99)] * 3 + [healthy])
    body = ReconnectingBody(connect_body=lambda: next(bodies), max_attempts=3)
    for _ in range(10):
        body.perform("nod")
    assert healthy.motions == []  # dead body never reaches a later healthy connection

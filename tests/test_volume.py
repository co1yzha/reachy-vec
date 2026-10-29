import pytest

from reachy_vec.audio.volume import MacVolume, RobotVolume, VolumeControl


class FakeHttp:
    def __init__(self, volume=68, fail=False):
        self.volume, self.fail, self.calls = volume, fail, []

    def __call__(self, url, *, method="GET", body=None, timeout=3.0):
        self.calls.append((method, url, body))
        if self.fail:
            raise OSError("unreachable")
        if method == "POST":
            self.volume = body["volume"]
        return 200, {"volume": self.volume}


@pytest.mark.parametrize("change, expected", [
    ("up", 83), ("down", 53), ("max", 100), ("quiet", 35), ("50", 50), ("louder", 83),
])
def test_changes_step_from_the_current_volume(change, expected):
    http = FakeHttp(68)
    control = VolumeControl(RobotVolume("http://robot:8000/api", http=http))
    control.change(change)
    assert http.volume == expected
    assert http.calls[-1] == ("POST", "http://robot:8000/api/volume/set", {"volume": expected})


def test_never_goes_silent_or_past_max():
    http = FakeHttp(20)
    control = VolumeControl(RobotVolume("http://r/api", http=http))
    control.change("down")
    assert http.volume == 15            # floor: still audible
    assert "quietest" in control.change("down")
    http.volume = 95
    control.change("up")
    assert http.volume == 100
    assert "loudest" in control.change("up")


def test_report_says_the_new_level():
    control = VolumeControl(RobotVolume("http://r/api", http=FakeHttp(68)))
    assert "83" in control.change("up")


def test_unknown_change_asks_back_without_touching_volume():
    http = FakeHttp(68)
    reply = VolumeControl(RobotVolume("http://r/api", http=http)).change("sideways")
    assert http.volume == 68 and "louder or quieter" in reply


def test_robot_unreachable_is_reported_not_raised():
    reply = VolumeControl(RobotVolume("http://r/api", http=FakeHttp(fail=True))).change("up")
    assert "couldn't" in reply.lower()


def test_mac_volume_uses_osascript():
    calls = []

    def run(script):
        calls.append(script)
        return "40" if script.startswith("output volume") else ""

    mac = MacVolume(run=run)
    assert mac.get() == 40
    mac.set(55)
    assert calls[-1] == "set volume output volume 55"

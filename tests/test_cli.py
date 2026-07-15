from typer.testing import CliRunner

from reachy_vec.cli import app
from reachy_vec.cli.run import resolve_media_source, wait_for_frame
from reachy_vec.store.db import Store
from tests.conftest import FakeEmbedder

runner = CliRunner()


def test_resolve_auto_prefers_robot_when_available():
    assert resolve_media_source("auto", media_available=True) == "robot"


def test_resolve_auto_falls_back_to_mac():
    assert resolve_media_source("auto", media_available=False) == "mac"


def test_resolve_explicit_robot_is_honored_even_if_unavailable():
    assert resolve_media_source("robot", media_available=False) == "robot"


def test_resolve_explicit_mac():
    assert resolve_media_source("mac", media_available=True) == "mac"


class _StubMini:
    def goto_target(self, **kw):
        pass

    def goto_sleep(self):
        pass

    def wake_up(self):
        pass


class _ScriptedCamera:
    def __init__(self, frames):
        self._frames = iter(frames)

    def read(self):
        return next(self._frames, None)


def test_wait_for_frame_retries_until_first_frame():
    # robot camera over WebRTC delivers its first frame after ~2s, not instantly
    cam = _ScriptedCamera([None, None, None, "frame"])
    t = {"now": 0.0}
    frame = wait_for_frame(
        cam,
        timeout_s=10.0,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
    )
    assert frame == "frame"


def test_wait_for_frame_gives_up_after_timeout():
    cam = _ScriptedCamera([])  # never a frame
    t = {"now": 0.0}
    frame = wait_for_frame(
        cam,
        timeout_s=2.0,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
    )
    assert frame is None


def test_wrap_reconnect_wraps_a_robot_body(monkeypatch):
    from reachy_vec.body.robot import ReconnectingBody, RobotBody
    from reachy_vec.cli.run import wrap_reconnect

    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", True)
    body = RobotBody(_StubMini())
    wrapped = wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None)
    assert isinstance(wrapped, ReconnectingBody)
    assert wrapped._inner is body  # seeded with the live connection - no re-dial at startup


def test_wrap_reconnect_leaves_nullbody_alone(monkeypatch):
    from reachy_vec.body.robot import NullBody
    from reachy_vec.cli.run import wrap_reconnect

    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", True)
    body = NullBody()
    assert wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None) is body


def test_wrap_reconnect_disabled_returns_body_unchanged(monkeypatch):
    from reachy_vec.body.robot import RobotBody
    from reachy_vec.cli.run import wrap_reconnect

    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", False)
    body = RobotBody(_StubMini())
    assert wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None) is body


def test_barge_in_factory_none_when_disabled(monkeypatch):
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr("reachy_vec.cli.run.settings.barge_in", False)
    assert make_barge_in_factory("mac", media=None) is None


def test_barge_in_factory_builds_monitor_when_enabled(monkeypatch):
    from reachy_vec.audio.listen import BargeInMonitor
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr("reachy_vec.cli.run.settings.barge_in", True)
    factory = make_barge_in_factory("mac", media=None)
    assert isinstance(factory(), BargeInMonitor)


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "ingest", "enroll", "run"):
        assert command in result.output


def test_ingest_command_writes_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha content")
    db_dir = tmp_path / "lancedb"

    monkeypatch.setattr("reachy_vec.cli.ingest.make_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("reachy_vec.cli.ingest.settings.data_dir", tmp_path)

    result = runner.invoke(app, ["ingest", str(docs)])
    assert result.exit_code == 0, result.output
    assert "1 chunk" in result.output
    assert Store(db_dir).doc_count() == 1


def test_ingest_command_rejects_missing_path(tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope")])
    assert result.exit_code != 0

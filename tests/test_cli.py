from pathlib import Path

import pytest
from typer.testing import CliRunner

from reachy_vec.cli import app
from reachy_vec.cli.run import resolve_media_source, wait_for_frame
from reachy_vec.config import settings
from reachy_vec.store.db import Store
from tests.conftest import FakeEmbedder, FakeSpeaker

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
    monkeypatch.setattr("reachy_vec.cli.run.settings.barge_in_mode", "speech")
    factory = make_barge_in_factory("mac", media=None)
    assert isinstance(factory(), BargeInMonitor)


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "ingest", "run", "record-voice"):
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


def test_self_voice_checker_none_without_clone_voice(monkeypatch):
    from reachy_vec.cli.run import make_self_voice_checker

    monkeypatch.setattr("reachy_vec.cli.run.settings.tts_backend", "say")
    assert make_self_voice_checker(store=None) is None


def test_self_voice_checker_compares_against_clone_sample(monkeypatch, tmp_path):
    import numpy as np

    from reachy_vec.cli.run import make_self_voice_checker

    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"stub")  # existence is all the factory checks
    monkeypatch.setattr("reachy_vec.cli.run.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.cli.run.settings.voice_sample", sample)
    monkeypatch.setattr("reachy_vec.cli.run.settings.voice_threshold", 0.30)

    class FakeIdent:
        def embed(self, audio):
            return None if audio is None else [float(audio[0])] * 3

    checker = make_self_voice_checker(
        store=None,
        identifier=FakeIdent(),
        read_sample=lambda path: np.ones(16000, dtype=np.float32),
    )
    assert checker is not None
    assert checker(np.ones(16000, dtype=np.float32)) is True    # same voice
    assert checker(-np.ones(16000, dtype=np.float32)) is False  # different voice
    assert checker(None) is False                               # can't tell -> not self


def test_make_hermes_none_when_disabled(monkeypatch, tmp_path):
    from reachy_vec.cli.run import make_hermes
    from reachy_vec.config import settings

    monkeypatch.setattr(settings, "hermes_enabled", False)
    assert make_hermes(Store(tmp_path / "db"), which=lambda b: "/x/hermes") is None


def test_make_hermes_none_when_binary_missing(monkeypatch, tmp_path):
    from reachy_vec.cli.run import make_hermes
    from reachy_vec.config import settings

    monkeypatch.setattr(settings, "hermes_enabled", True)
    monkeypatch.setattr(settings, "hermes_transport", "cli")  # independent of the real .env
    assert make_hermes(Store(tmp_path / "db"), which=lambda b: None) is None


def test_make_hermes_builds_jobs_with_settings(monkeypatch, tmp_path):
    from reachy_vec.brain.hermes import HermesJobs
    from reachy_vec.cli.run import make_hermes
    from reachy_vec.config import settings

    monkeypatch.setattr(settings, "hermes_enabled", True)
    monkeypatch.setattr(settings, "hermes_transport", "cli")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    jobs = make_hermes(Store(tmp_path / "db"), which=lambda b: "/x/hermes")
    assert isinstance(jobs, HermesJobs)
    assert jobs._agent.cwd == tmp_path / "hermes_cwd"
    assert jobs._agent.profile == "reachy"


def test_stt_client_reuses_cloud_client():
    from reachy_vec.cli.run import stt_client
    from reachy_vec.config import Settings

    llm = object()
    assert stt_client(llm, settings=Settings(llm_base_url=None, stt_backend="openai")) is llm


def test_stt_client_is_fresh_cloud_client_when_llm_local():
    from reachy_vec.cli.run import stt_client
    from reachy_vec.config import Settings

    llm = object()
    seen = {}

    def fake_cls(**kw):
        seen.update(kw)
        return "cloud"

    out = stt_client(llm, settings=Settings(llm_base_url="http://x/v1", stt_backend="openai"),
                     openai_cls=fake_cls)
    assert out == "cloud" and seen == {}


def test_stt_client_none_for_local_stt_without_building_a_cloud_client():
    """Local faster-whisper needs no client; building OpenAI() with no
    OPENAI_API_KEY raises, which crashed `run` on a fully local setup."""
    from reachy_vec.cli.run import stt_client
    from reachy_vec.config import Settings

    def boom(**kw):
        raise AssertionError("must not build a cloud client for local STT")

    for base_url in (None, "http://x/v1"):
        s = Settings(llm_base_url=base_url, stt_backend="local")
        assert stt_client(object(), settings=s, openai_cls=boom) is None


def test_prepare_fillers_off_returns_empty_and_skips_precache(monkeypatch):
    from reachy_vec.cli.run import prepare_fillers

    monkeypatch.setattr(settings, "ack_enabled", False)
    speaker = FakeSpeaker()
    assert prepare_fillers(speaker) == []
    assert speaker.precached == []


def test_prepare_fillers_on_pregenerates_them(monkeypatch):
    from reachy_vec.cli.run import prepare_fillers

    monkeypatch.setattr(settings, "ack_enabled", True)
    monkeypatch.setattr(settings, "ack_fillers", ["Mm.", "Right."])
    speaker = FakeSpeaker()
    assert prepare_fillers(speaker) == ["Mm.", "Right."]
    assert speaker.precached == [["Mm.", "Right."]]


def test_default_fillers():
    from reachy_vec.config import Settings

    assert Settings.model_fields["ack_fillers"].default == ["Mm.", "Right.", "Okay.", "Let me see."]
    assert Settings.model_fields["ack_enabled"].default is True


class _FakeGatewayAgent:
    instances: list = []

    def __init__(self, *, base_url, profile, api_key, fail_check=None):
        self.base_url, self.profile, self.api_key = base_url, profile, api_key
        self.fail_check = fail_check
        self.warmed = False
        _FakeGatewayAgent.instances.append(self)

    def check_toolsets(self, allowed):
        self.allowed = allowed
        if self.fail_check:
            from reachy_vec.brain.hermes import HermesError

            raise HermesError(self.fail_check)

    def warm_up(self):
        self.warmed = True


def _gateway_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "hermes_enabled", True)
    monkeypatch.setattr(settings, "hermes_transport", "gateway")
    monkeypatch.setattr(settings, "hermes_api_key", "k" * 32)
    monkeypatch.setattr(settings, "data_dir", tmp_path)


def test_make_hermes_uses_the_gateway(monkeypatch, tmp_path):
    from reachy_vec.cli.run import make_hermes

    _gateway_settings(monkeypatch, tmp_path)
    jobs = make_hermes(Store(tmp_path / "db"), which=lambda b: "/x/hermes",
                       gateway_cls=_FakeGatewayAgent, warm_in_background=False)
    agent = jobs._agent
    assert isinstance(agent, _FakeGatewayAgent)
    assert (agent.base_url, agent.profile) == ("http://127.0.0.1:8642", "reachy")
    assert agent.allowed == {"web", "browser", "memory", "skills", "session_search", "todo",
                             "cronjob", "kanban"}
    assert agent.warmed


def test_make_hermes_falls_back_to_cli_when_gateway_unsafe_or_down(monkeypatch, tmp_path):
    from functools import partial

    from reachy_vec.brain.hermes import HermesCli
    from reachy_vec.cli.run import make_hermes

    _gateway_settings(monkeypatch, tmp_path)
    jobs = make_hermes(Store(tmp_path / "db"), which=lambda b: "/x/hermes",
                       gateway_cls=partial(_FakeGatewayAgent, fail_check="terminal enabled"),
                       warm_in_background=False)
    assert isinstance(jobs._agent, HermesCli)


def test_make_hermes_gateway_without_key_falls_back(monkeypatch, tmp_path):
    from reachy_vec.brain.hermes import HermesCli
    from reachy_vec.cli.run import make_hermes

    _gateway_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "hermes_api_key", None)
    jobs = make_hermes(Store(tmp_path / "db"), which=lambda b: "/x/hermes",
                       warm_in_background=False)
    assert isinstance(jobs._agent, HermesCli)


def test_hermes_api_key_reads_the_profile_api_server_key(monkeypatch):
    from reachy_vec.config import Settings

    monkeypatch.delenv("REACHY_VEC_HERMES_API_KEY", raising=False)
    monkeypatch.setenv("API_SERVER_KEY", "x" * 32)
    assert Settings(_env_file=None).hermes_api_key == "x" * 32


@pytest.mark.parametrize(("media_source", "mic_setting", "expected"), [
    ("robot", "auto", "robot"),   # default: mic follows the media source
    ("mac", "auto", "mac"),
    ("robot", "mac", "mac"),      # robot speaker + Mac/USB mic
    ("mac", "robot", "mac"),      # can't use the robot mic without robot media
])
def test_resolve_mic(media_source, mic_setting, expected):
    from reachy_vec.cli.run import resolve_mic

    assert resolve_mic(media_source, mic_setting) == expected


def test_mic_source_default_is_auto():
    from reachy_vec.config import Settings

    assert Settings.model_fields["mic_source"].default == "auto"


def test_mac_mic_missing_falls_back_to_robot_mic():
    from reachy_vec.cli.run import check_mic

    assert check_mic("mac", robot_media=True, has_mac_input=lambda: False) == "robot"


def test_mac_mic_present_is_kept():
    from reachy_vec.cli.run import check_mic

    assert check_mic("mac", robot_media=True, has_mac_input=lambda: True) == "mac"


def test_no_mic_anywhere_is_a_clear_error():
    import typer

    from reachy_vec.cli.run import check_mic

    with pytest.raises(typer.Exit):
        check_mic("mac", robot_media=False, has_mac_input=lambda: False)


def test_robot_mic_needs_no_mac_check():
    from reachy_vec.cli.run import check_mic

    def boom():
        raise AssertionError("should not query Mac devices")

    assert check_mic("robot", robot_media=True, has_mac_input=boom) == "robot"


def test_camera_tools_off_without_robot_media(monkeypatch):
    from reachy_vec.cli.run import make_camera_tools

    monkeypatch.setattr(settings, "camera_tools", True)
    assert make_camera_tools(None, client=object(), body=None, speak=print) == {}


def test_camera_tools_off_when_disabled(monkeypatch):
    from reachy_vec.cli.run import make_camera_tools

    monkeypatch.setattr(settings, "camera_tools", False)
    assert make_camera_tools(object(), client=object(), body=None, speak=print) == {}


def test_camera_tools_built_when_a_frame_arrives(monkeypatch):
    from reachy_vec.cli.run import make_camera_tools
    from tests.conftest import FakeCamera

    monkeypatch.setattr(settings, "camera_tools", True)
    tools = make_camera_tools(object(), client=object(), body=None, speak=print,
                              camera_cls=lambda media: FakeCamera(["frame"]))
    assert set(tools) == {"look_fn", "selfie_fn", "look_around_fn"}


def test_camera_tools_skipped_when_camera_gives_no_frame(monkeypatch):
    from reachy_vec.cli.run import make_camera_tools
    from tests.conftest import FakeCamera

    monkeypatch.setattr(settings, "camera_tools", True)
    tools = make_camera_tools(object(), client=object(), body=None, speak=print,
                              camera_cls=lambda media: FakeCamera([]), frame_timeout_s=0.01)
    assert tools == {}


def test_keyword_barge_in_confirms_with_the_interrupt_phrase(monkeypatch):
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr(settings, "barge_in", True)
    monkeypatch.setattr(settings, "barge_in_mode", "keyword")
    heard = {"text": "the latest demo is"}
    factory = make_barge_in_factory("mac", None, transcribe=lambda audio: heard["text"])
    monitor = factory()
    assert monitor._confirm(None) is False
    heard["text"] = "Reachy, stop"
    assert monitor._confirm(None) is True


def test_speech_barge_in_has_no_confirm(monkeypatch):
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr(settings, "barge_in", True)
    monkeypatch.setattr(settings, "barge_in_mode", "speech")
    assert make_barge_in_factory("mac", None, transcribe=lambda a: "")()._confirm is None


def test_keyword_barge_in_without_transcriber_is_off(monkeypatch):
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr(settings, "barge_in", True)
    monkeypatch.setattr(settings, "barge_in_mode", "keyword")
    assert make_barge_in_factory("mac", None, transcribe=None) is None


def test_keyword_barge_in_ignores_reachys_own_reply_echoed(monkeypatch):
    from reachy_vec.cli.run import make_barge_in_factory

    monkeypatch.setattr(settings, "barge_in", True)
    monkeypatch.setattr(settings, "barge_in_mode", "keyword")
    heard = {"text": "But I'll stay quiet if you'd like to."}
    said = "I can't switch myself off, but I'll stay quiet if you'd like some peace."
    factory = make_barge_in_factory(
        "mac", None, transcribe=lambda audio: heard["text"],
        is_echo=lambda text: text == heard["text"] and "quiet" in said,
    )
    assert factory()._confirm(None) is False


def test_run_log_rotates_daily_and_keeps_history(tmp_path, monkeypatch):
    import logging
    import logging.handlers

    from reachy_vec.cli import run as run_cli

    monkeypatch.setattr(run_cli.settings, "data_dir", tmp_path)
    app_logger = logging.getLogger("reachy_vec")
    before = list(app_logger.handlers)
    try:
        run_cli._setup_logging()
        added = [h for h in app_logger.handlers if h not in before]
        files = [h for h in added if isinstance(h, logging.FileHandler)]
        assert len(files) == 1
        handler = files[0]
        assert isinstance(handler, logging.handlers.TimedRotatingFileHandler)
        assert handler.when == "MIDNIGHT"
        assert handler.backupCount == 0  # never delete old days
        assert Path(handler.baseFilename) == tmp_path / "reachy.log"
    finally:
        for h in app_logger.handlers[:]:
            if h not in before:
                app_logger.removeHandler(h)
                h.close()

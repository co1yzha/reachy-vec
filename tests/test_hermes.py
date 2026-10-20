import subprocess
import time
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from reachy_vec.brain.hermes import HermesCli, HermesError, HermesJobs, build_prompt
from reachy_vec.store.db import Store
from tests.conftest import FakeAgent


def _cli(runner, **kw):
    return HermesCli(bin="hermes", profile="reachy", toolsets="web,todo", runner=runner, **kw)


def test_args_exact_and_no_shell():
    seen = {}

    def runner(args, **kw):
        seen.update(args=args, **kw)
        return CompletedProcess(args, 0, "Done.\n", "")

    out = _cli(runner).run("check X; rm -rf ~ $(whoami)", requester="Bob", timeout_s=5)
    assert out == "Done."
    assert seen["args"][:4] == ["hermes", "-p", "reachy", "-z"]
    assert seen["args"][5:] == ["-t", "web,todo"]
    assert "check X; rm -rf ~ $(whoami)" in seen["args"][4] and "shell" not in seen
    assert seen["timeout"] == 5
    assert "cwd" not in seen


def test_prompt_copy_anonymous():
    assert build_prompt("x", None).startswith(
        "Task from an unidentified teammate, asked by voice via Reachy: x."
    )


def test_prompt_copy_exact_named():
    assert build_prompt("x", "Bob") == (
        "Task from Bob, asked by voice via Reachy: x. "
        "Reply in at most 3 short sentences suitable for speaking aloud."
    )


def test_cwd_passed_and_created(tmp_path):
    seen = {}

    def runner(args, **kw):
        seen.update(kw)
        return CompletedProcess(args, 0, "ok", "")

    cwd = tmp_path / "a" / "hermes_cwd"
    _cli(runner, cwd=cwd).run("x", requester=None, timeout_s=1)
    assert seen["cwd"] == str(cwd)
    assert Path(cwd).is_dir()


def _nonzero(args, **kw):
    return CompletedProcess(args, 1, "out", "boom")


def _empty(args, **kw):
    return CompletedProcess(args, 0, "  \n", "")


def _timeout(args, **kw):
    raise subprocess.TimeoutExpired(args, 1)


def _notfound(args, **kw):
    raise FileNotFoundError("hermes")


@pytest.mark.parametrize("runner", [_nonzero, _empty, _timeout, _notfound])
def test_failures_raise_hermes_error(runner):
    with pytest.raises(HermesError):
        _cli(runner).run("x", requester="Bob", timeout_s=1)


@pytest.mark.parametrize("profile", ["", "  "])
def test_empty_profile_rejected(profile):
    with pytest.raises(ValueError):
        HermesCli(bin="hermes", profile=profile, toolsets="web")


def test_default_setting_is_reachy_profile(monkeypatch):
    from reachy_vec.config import Settings

    monkeypatch.delenv("REACHY_VEC_HERMES_PROFILE", raising=False)
    assert Settings.model_fields["hermes_profile"].default == "reachy"


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _jobs(tmp_path, agent):
    store = Store(tmp_path / "db")
    return store, HermesJobs(agent, store, wait_s=0.05, task_timeout_s=5)


def test_fast_result_returned(tmp_path):
    _, jobs = _jobs(tmp_path, FakeAgent(["SDK v2 is out."]))
    assert jobs.ask("t", requester_id="p2", requester_name="Bob") == "SDK v2 is out."


def test_slow_result_becomes_message_for_requester(tmp_path):
    store, jobs = _jobs(tmp_path, FakeAgent(["  SDK v2 is out.\n"], delay_s=0.2))
    out = jobs.ask("t", requester_id="p2", requester_name="Bob")
    assert out.startswith("still working")
    assert wait_until(lambda: store.pending_messages_for("p2"))
    msg = store.pending_messages_for("p2")[0]
    assert (msg.from_person, msg.from_name, msg.to_name, msg.text) == (
        "hermes", "Hermes", "Bob", "SDK v2 is out.",
    )


def test_slow_anonymous_result_not_stored(tmp_path):
    agent = FakeAgent(["x"], delay_s=0.2)
    store, jobs = _jobs(tmp_path, agent)
    out = jobs.ask("t", requester_id=None, requester_name=None)
    assert out.startswith("this is taking too long")
    assert wait_until(lambda: not jobs.busy)
    time.sleep(0.05)
    assert store.pending_messages_for("") == []


def test_busy_refuses_second_task(tmp_path):
    agent = FakeAgent(["a", "b"], delay_s=0.2)
    _, jobs = _jobs(tmp_path, agent)
    jobs.ask("one", requester_id="p2", requester_name="Bob")
    assert jobs.busy
    assert jobs.ask("two", requester_id="p2", requester_name="Bob").startswith(
        "I'm still working"
    )
    assert len(agent.calls) == 1


def test_error_returns_spoken_failure(tmp_path):
    _, jobs = _jobs(tmp_path, FakeAgent([HermesError("boom")]))
    assert jobs.ask("t", requester_id="p2", requester_name="Bob") == (
        "Hermes couldn't finish that"
    )


def test_late_error_writes_no_message(tmp_path):
    store, jobs = _jobs(tmp_path, FakeAgent([HermesError("boom")], delay_s=0.2))
    assert jobs.ask("t", requester_id="p2", requester_name="Bob").startswith("still working")
    assert wait_until(lambda: not jobs.busy)
    time.sleep(0.05)
    assert store.pending_messages_for("p2") == []

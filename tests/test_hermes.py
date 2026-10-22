import subprocess
import threading
import time
from pathlib import Path

import pytest

from reachy_vec.brain.hermes import HermesCli, HermesError, HermesJobs, build_prompt
from reachy_vec.store.db import Store
from tests.conftest import FakeAgent


class FakePopen:
    """Popen stand-in; communicate() returns canned output or raises."""

    def __init__(self, args, returncode=0, out="", err="", raises=None, **kw):
        self.args, self.kw = args, kw
        self.returncode, self._out, self._err, self._raises = returncode, out, err, raises
        self.timeout = None
        self.killed = False

    def communicate(self, timeout=None):
        self.timeout = timeout
        if self._raises is not None:
            raise self._raises
        return self._out, self._err

    def kill(self):
        self.killed = True


def _cli(popen, **kw):
    return HermesCli(bin="hermes", profile="reachy", toolsets="web,todo", popen=popen, **kw)


def test_args_exact_and_no_shell():
    made = []

    def popen(args, **kw):
        made.append(FakePopen(args, out="Done.\n", **kw))
        return made[0]

    out = _cli(popen).run("check X; rm -rf ~ $(whoami)", requester="Bob", timeout_s=5)
    p = made[0]
    assert out == "Done."
    assert p.args[:4] == ["hermes", "-p", "reachy", "-z"]
    assert p.args[5:] == ["-t", "web,todo"]
    assert "check X; rm -rf ~ $(whoami)" in p.args[4] and "shell" not in p.kw
    assert p.timeout == 5
    assert "cwd" not in p.kw


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

    def popen(args, **kw):
        seen.update(kw)
        return FakePopen(args, out="ok")

    cwd = tmp_path / "a" / "hermes_cwd"
    _cli(popen, cwd=cwd).run("x", requester=None, timeout_s=1)
    assert seen["cwd"] == str(cwd)
    assert Path(cwd).is_dir()


def _nonzero(args, **kw):
    return FakePopen(args, returncode=1, out="out", err="boom")


def _empty(args, **kw):
    return FakePopen(args, out="  \n")


def _timeout(args, **kw):
    return FakePopen(args, raises=subprocess.TimeoutExpired(args, 1))


def _notfound(args, **kw):
    raise FileNotFoundError("hermes")


@pytest.mark.parametrize("popen", [_nonzero, _empty, _timeout, _notfound])
def test_failures_raise_hermes_error(popen):
    with pytest.raises(HermesError):
        _cli(popen).run("x", requester="Bob", timeout_s=1)


def test_timeout_kills_process():
    made = []

    def popen(args, **kw):
        made.append(_timeout(args))
        return made[0]

    with pytest.raises(HermesError):
        _cli(popen).run("x", requester="Bob", timeout_s=1)
    assert made[0].killed


def test_terminate_without_process_is_noop():
    _cli(_nonzero).terminate()


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


def test_close_returns_promptly_while_job_running(tmp_path):
    _, jobs = _jobs(tmp_path, FakeAgent(["x"], delay_s=2.0))
    jobs.ask("t", requester_id="p2", requester_name="Bob")
    start = time.monotonic()
    jobs.close()
    assert time.monotonic() - start < 0.5


class BlockingPopen(FakePopen):
    def __init__(self, args, **kw):
        super().__init__(args, **kw)
        self.stopped = threading.Event()
        self.terminated = False

    def communicate(self, timeout=None):
        self.stopped.wait(10)
        return "", ""

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stopped.set()

    def wait(self, timeout=None):
        return self.returncode


def test_close_terminates_in_flight_hermes_process(tmp_path):
    made = []

    def popen(args, **kw):
        made.append(BlockingPopen(args, **kw))
        return made[0]

    store = Store(tmp_path / "db")
    jobs = HermesJobs(_cli(popen), store, wait_s=0.05, task_timeout_s=30)
    assert jobs.ask("slow", requester_id="p2", requester_name="Bob").startswith("still working")
    assert jobs.busy
    jobs.close()
    assert made[0].terminated
    assert wait_until(lambda: not jobs.busy, 1.0)


def test_status_reports_running_task_then_clears(tmp_path):
    clock = {"t": 100.0}
    jobs = HermesJobs(FakeAgent(["done"], delay_s=0.3), Store(tmp_path / "db"),
                      wait_s=0.05, task_timeout_s=5, clock=lambda: clock["t"])
    assert jobs.status() is None                       # idle
    jobs.ask("research X", requester_id="p1", requester_name="Yang")
    clock["t"] = 145.0
    assert jobs.status() == ("research X", 45.0)       # handed off, still running
    wait_until(lambda: not jobs.busy)
    assert jobs.status() is None                       # finished


def test_anonymous_requester_waits_longer_instead_of_dropping(tmp_path):
    """Nothing can be delivered later to someone unrecognised, so wait longer."""
    jobs = HermesJobs(FakeAgent(["The headlines are X."], delay_s=0.15), Store(tmp_path / "db"),
                      wait_s=0.05, wait_anonymous_s=1.0, task_timeout_s=5)
    assert jobs.ask("latest AI news", requester_id=None, requester_name=None) == \
        "The headlines are X."


def test_recognised_requester_keeps_the_short_wait(tmp_path):
    jobs = HermesJobs(FakeAgent(["late"], delay_s=0.15), Store(tmp_path / "db"),
                      wait_s=0.05, wait_anonymous_s=1.0, task_timeout_s=5)
    assert jobs.ask("x", requester_id="p1", requester_name="Yang").startswith("still working")


def test_progress_updates_while_waiting(tmp_path):
    ticks = []
    jobs = HermesJobs(FakeAgent(["Done."], delay_s=0.35), Store(tmp_path / "db"),
                      wait_s=2.0, task_timeout_s=5, progress_s=0.1)
    out = jobs.ask("t", requester_id="p1", requester_name="Yang",
                   on_progress=lambda: ticks.append(1) or True)
    assert out == "Done." and len(ticks) >= 2


def test_progress_callback_can_stop_the_wait(tmp_path):
    jobs = HermesJobs(FakeAgent(["late"], delay_s=0.5), Store(tmp_path / "db"),
                      wait_s=5.0, task_timeout_s=5, progress_s=0.05)
    out = jobs.ask("t", requester_id="p1", requester_name="Yang", on_progress=lambda: False)
    assert out.startswith("still working")          # handed off at once
    wait_until(lambda: Store(tmp_path / "db").pending_messages_for("p1"))

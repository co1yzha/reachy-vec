import subprocess
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from reachy_vec.brain.hermes import HermesCli, HermesError, build_prompt


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

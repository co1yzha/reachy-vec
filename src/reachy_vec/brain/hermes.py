"""Agent adapter: delegates spoken tasks to the local Hermes CLI."""

import subprocess
from pathlib import Path
from typing import Protocol


class HermesError(Exception):
    """Hermes failed, timed out, or returned nothing."""


class Agent(Protocol):
    def run(self, task: str, *, requester: str | None, timeout_s: float) -> str: ...


def build_prompt(task: str, requester: str | None) -> str:
    who = requester or "an unidentified teammate"
    return (
        f"Task from {who}, asked by voice via Reachy: {task}. "
        "Reply in at most 3 short sentences suitable for speaking aloud."
    )


class HermesCli:
    def __init__(
        self,
        *,
        bin: str,
        profile: str,
        toolsets: str,
        runner=subprocess.run,
        cwd: Path | None = None,
    ):
        if not profile or not profile.strip():
            # Never fall back to Hermes's default profile by dropping -p.
            raise ValueError("Hermes profile must be a non-empty name")
        self.bin = bin
        self.profile = profile
        self.toolsets = toolsets
        self.runner = runner
        self.cwd = cwd

    def run(self, task: str, *, requester: str | None, timeout_s: float) -> str:
        args = [
            self.bin, "-p", self.profile, "-z", build_prompt(task, requester),
            "-t", self.toolsets,
        ]
        kwargs: dict = {"capture_output": True, "text": True, "timeout": timeout_s}
        try:
            if self.cwd is not None:
                self.cwd.mkdir(parents=True, exist_ok=True)
                kwargs["cwd"] = str(self.cwd)
            proc = self.runner(args, **kwargs)
        except subprocess.TimeoutExpired as e:
            raise HermesError(f"hermes timed out after {timeout_s}s") from e
        except OSError as e:
            raise HermesError(f"hermes could not run: {e}") from e
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()[:200]
            raise HermesError(f"hermes exited {proc.returncode}: {err}")
        out = (proc.stdout or "").strip()
        if not out:
            raise HermesError("hermes returned no output")
        return out

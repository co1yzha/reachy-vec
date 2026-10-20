"""Agent adapter: delegates spoken tasks to the local Hermes CLI."""

import logging
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from reachy_vec.store.db import Store
from reachy_vec.store.schemas import MessageRow

logger = logging.getLogger(__name__)

HANDED_OFF = "still working; tell the user you'll let them know when it's ready"
TOO_SLOW_ANON = "this is taking too long, and I can't get back to someone I don't recognise"
BUSY = "I'm still working on your last request"
FAILED = "Hermes couldn't finish that"


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


class HermesJobs:
    """One Hermes task at a time: wait briefly, else hand off as a message."""

    def __init__(self, agent: Agent, store: Store, *, wait_s: float, task_timeout_s: float):
        self._agent = agent
        self._store = store
        self._wait_s = wait_s
        self._task_timeout_s = task_timeout_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes")
        self._future: Future | None = None

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def _run(self, task: str, requester_name: str | None) -> str:
        start = time.monotonic()
        logger.info("hermes task from %s: %r", requester_name, task)
        try:
            result = self._agent.run(
                task, requester=requester_name, timeout_s=self._task_timeout_s
            )
        except Exception as e:
            logger.warning("hermes task failed after %.1fs: %s", time.monotonic() - start, e)
            raise
        logger.info("hermes result after %.1fs: %r", time.monotonic() - start, result)
        return result

    def ask(self, task: str, *, requester_id: str | None, requester_name: str | None) -> str:
        if self.busy:
            return BUSY
        future = self._pool.submit(self._run, task, requester_name)
        self._future = future
        try:
            return future.result(timeout=self._wait_s).strip()
        except FutureTimeout:
            pass
        except Exception:
            return FAILED
        future.add_done_callback(
            lambda f: self._deliver(f, requester_id, requester_name)
        )
        return HANDED_OFF if requester_id else TOO_SLOW_ANON

    def _deliver(self, future: Future, requester_id: str | None, requester_name: str | None):
        try:
            if future.exception() is not None:
                return  # already logged in _run
            text = future.result().strip()
            if not requester_id:
                logger.info("hermes late result dropped (anonymous requester)")
                return
            self._store.add_message(
                MessageRow(
                    message_id=f"msg-{uuid.uuid4().hex[:10]}",
                    from_person="hermes",
                    from_name="Hermes",
                    to_person=requester_id,
                    to_name=requester_name or "",
                    text=text,
                    created_at=datetime.now(UTC).isoformat(),
                    delivered_at="",
                )
            )
            logger.info("hermes result queued for %s", requester_name or requester_id)
        except Exception:
            logger.exception("hermes late-result delivery failed")

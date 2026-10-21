"""Agent adapters: delegate spoken tasks to Hermes - via the running gateway's
HTTP API (HermesGateway, preferred) or a one-shot CLI process (HermesCli)."""

import json
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
    def run(
        self, task: str, *, requester: str | None, timeout_s: float,
        requester_id: str | None = None,
    ) -> str: ...


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
        popen=subprocess.Popen,
        cwd: Path | None = None,
    ):
        if not profile or not profile.strip():
            # Never fall back to Hermes's default profile by dropping -p.
            raise ValueError("Hermes profile must be a non-empty name")
        self.bin = bin
        self.profile = profile
        self.toolsets = toolsets
        self.popen = popen
        self.cwd = cwd
        self._proc = None

    def run(
        self, task: str, *, requester: str | None, timeout_s: float,
        requester_id: str | None = None,  # the CLI has no per-person session key
    ) -> str:
        args = [
            self.bin, "-p", self.profile, "-z", build_prompt(task, requester),
            "-t", self.toolsets,
        ]
        kwargs: dict = {
            "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True,
        }
        proc = None
        try:
            if self.cwd is not None:
                self.cwd.mkdir(parents=True, exist_ok=True)
                kwargs["cwd"] = str(self.cwd)
            proc = self._proc = self.popen(args, **kwargs)
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            self._kill(proc)
            raise HermesError(f"hermes timed out after {timeout_s}s") from e
        except OSError as e:
            raise HermesError(f"hermes could not run: {e}") from e
        finally:
            self._proc = None
        if proc.returncode != 0:
            err = (stderr or "").strip()[:200]
            raise HermesError(f"hermes exited {proc.returncode}: {err}")
        out = (stdout or "").strip()
        if not out:
            raise HermesError("hermes returned no output")
        return out

    @staticmethod
    def _kill(proc) -> None:
        try:
            proc.kill()
            proc.communicate(timeout=2)
        except Exception:
            pass

    def terminate(self) -> None:
        """Stop the in-flight Hermes process, if any. Never raises."""
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            logger.warning("hermes terminate failed", exc_info=True)


TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}


def _urllib_http(method: str, url: str, headers: dict, body: bytes | None):
    """(status, parsed JSON) - HTTP errors come back as their status, not raised."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


class HermesGateway:
    """Hermes via the always-running gateway's API server (Hermes >= 0.22).

    Starts a run on /p/<profile>/v1/runs, polls it to a terminal state, and
    stops it on timeout or when it parks waiting for a tool approval (never
    approves). No per-request toolset flag exists, so check_toolsets() verifies
    the profile's api_server toolsets instead. `http`, `clock`, `sleep` are
    injectable for tests.
    """

    def __init__(
        self,
        *,
        base_url: str,
        profile: str,
        api_key: str | None,
        http=_urllib_http,
        poll_s: float = 0.5,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        if not profile.strip():
            raise ValueError("Hermes profile must not be empty")  # no default fallback
        if not api_key:
            raise ValueError("Hermes gateway needs the profile's API_SERVER_KEY")
        self._base = f"{base_url.rstrip('/')}/p/{profile.strip()}"
        self._key = api_key
        self._http = http
        self._poll_s = poll_s
        self._clock = clock
        self._sleep = sleep
        self._run_id: str | None = None

    def _call(self, method: str, path: str, body: dict | None = None, extra=None):
        headers = {"Authorization": f"Bearer {self._key}", **(extra or {})}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        try:
            return self._http(method, self._base + path, headers, data)
        except OSError as e:
            raise HermesError(f"Hermes gateway unreachable: {e}") from e

    def run(
        self, task: str, *, requester: str | None, timeout_s: float,
        requester_id: str | None = None,
    ) -> str:
        extra = {"Idempotency-Key": f"reachy-{uuid.uuid4().hex}"}
        if requester_id:
            extra["X-Hermes-Session-Key"] = f"reachy:{requester_id}"  # per-person memory
        body = {"input": build_prompt(task, requester)}
        if requester:
            body["author"] = {"name": requester}
        status, data = self._call("POST", "/v1/runs", body, extra)
        if status == 401:
            raise HermesError("Hermes gateway rejected the key (auth 401)")
        if status == 429:
            raise HermesError("Hermes gateway is busy (429)")
        if status not in (200, 202) or "run_id" not in data:
            raise HermesError(f"Hermes gateway refused the run ({status}): {data}")
        self._run_id = run_id = data["run_id"]
        deadline = self._clock() + timeout_s
        try:
            while True:
                _, run = self._call("GET", f"/v1/runs/{run_id}")
                state = run.get("status")
                if state in TERMINAL_STATES:
                    output = (run.get("output") or "").strip()
                    if state == "completed" and output:
                        return output
                    raise HermesError(f"Hermes run {state}: {run.get('error') or 'no output'}")
                if state == "waiting_for_approval":
                    self._stop(run_id)
                    raise HermesError("Hermes run needed a tool approval; stopped it")
                if self._clock() >= deadline:
                    self._stop(run_id)
                    raise HermesError(f"Hermes run timed out after {timeout_s:.0f}s")
                self._sleep(self._poll_s)
        finally:
            self._run_id = None

    def _stop(self, run_id: str) -> None:
        try:
            self._call("POST", f"/v1/runs/{run_id}/stop", {})
        except HermesError:
            logger.warning("could not stop Hermes run %s", run_id)

    def terminate(self) -> None:
        """Stop the in-flight run (on quit); a no-op when idle. Never raises."""
        run_id = self._run_id
        if run_id is not None:
            self._stop(run_id)

    def check_toolsets(self, allowed: set[str]) -> None:
        """Raise unless the profile's API runs have only `allowed` toolsets."""
        status, data = self._call("GET", "/v1/toolsets")
        if status != 200:
            raise HermesError(f"could not read Hermes toolsets ({status})")
        enabled = {t["name"] for t in data.get("data", []) if t.get("enabled")}
        extra = sorted(enabled - allowed)
        if extra:
            raise HermesError(f"Hermes API toolsets beyond the allowed set: {', '.join(extra)}")

    def warm_up(self) -> None:
        """One tiny run so oMLX caches Hermes's ~14k-token prompt (cold ~13 s,
        warm ~0.7 s). Best-effort."""
        try:
            self.run("Reply with the single word ready. Do not use tools.",
                     requester=None, timeout_s=120)
        except HermesError as e:
            logger.warning("Hermes gateway warm-up failed: %s", e)


class HermesJobs:
    """One Hermes task at a time: wait briefly, else hand off as a message."""

    def __init__(
        self,
        agent: Agent,
        store: Store,
        *,
        wait_s: float,
        wait_anonymous_s: float | None = None,
        task_timeout_s: float,
        clock=time.monotonic,
    ):
        self._agent = agent
        self._clock = clock
        self._task: str | None = None
        self._started = 0.0
        self._store = store
        self._wait_s = wait_s
        # an unrecognised speaker can't get a late result, so wait longer for them
        self._wait_anonymous_s = wait_s if wait_anonymous_s is None else wait_anonymous_s
        self._task_timeout_s = task_timeout_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes")
        self._future: Future | None = None

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def status(self) -> tuple[str, float] | None:
        """(task, seconds running) while a job is in flight, else None - so
        Reachy can say truthfully what Hermes is doing."""
        if not self.busy or self._task is None:
            return None
        return self._task, self._clock() - self._started

    def close(self) -> None:
        """Kill the in-flight Hermes process and stop accepting work.

        Does not block; without the kill the non-daemon worker would hold
        interpreter exit until the task finished or timed out."""
        terminate = getattr(self._agent, "terminate", None)
        if terminate is not None:
            terminate()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, task: str, requester_name: str | None, requester_id: str | None) -> str:
        start = time.monotonic()
        logger.info("hermes task from %s: %r", requester_name, task)
        try:
            result = self._agent.run(
                task, requester=requester_name, timeout_s=self._task_timeout_s,
                requester_id=requester_id,
            )
        except Exception as e:
            logger.warning("hermes task failed after %.1fs: %s", time.monotonic() - start, e)
            raise
        logger.info("hermes result after %.1fs: %r", time.monotonic() - start, result)
        return result

    def ask(self, task: str, *, requester_id: str | None, requester_name: str | None) -> str:
        if self.busy:
            return BUSY
        self._task, self._started = task, self._clock()
        future = self._pool.submit(self._run, task, requester_name, requester_id)
        self._future = future
        try:
            wait = self._wait_s if requester_id else self._wait_anonymous_s
            return future.result(timeout=wait).strip()
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

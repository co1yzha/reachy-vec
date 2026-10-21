"""HermesGateway against a scripted fake of the Hermes gateway HTTP API.

Response shapes are the live ones from Hermes 0.22.2 (POST /v1/runs -> 202
{"run_id", "status": "started"}; GET /v1/runs/{id} -> {"status", "output",
"error"?}; GET /v1/toolsets -> {"data": [{"name", "enabled"}]}).
"""

import json

import pytest

from reachy_vec.brain.hermes import HermesError, HermesGateway

TOOLSETS_OK = {"object": "list", "data": [
    {"name": n, "enabled": True} for n in ("web", "browser", "memory", "skills",
                                          "session_search", "todo", "cronjob", "kanban")
] + [{"name": "terminal", "enabled": False}, {"name": "file", "enabled": False}]}


class FakeGateway:
    """Records requests; GET /v1/runs/{id} walks through `statuses`."""

    def __init__(self, statuses=None, create=(202, {"run_id": "run_1", "status": "started"}),
                 toolsets=TOOLSETS_OK):
        self.statuses = list(statuses or [{"status": "completed", "output": " Canberra. "}])
        self.create = create
        self.toolsets = toolsets
        self.requests: list[tuple[str, str, dict, dict | None]] = []

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url, headers, json.loads(body) if body else None))
        if url.endswith("/v1/runs") and method == "POST":
            return self.create
        if url.endswith("/stop"):
            return 200, {"status": "stopping"}
        if url.endswith("/v1/toolsets"):
            return 200, self.toolsets
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return 200, {"object": "hermes.run", "run_id": "run_1", **status}


def gateway(fake, **kw):
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    return HermesGateway(base_url="http://127.0.0.1:8642", profile="reachy", api_key="k" * 32,
                         http=fake, poll_s=0.5, clock=lambda: clock["t"], sleep=sleep, **kw)


def test_run_posts_to_the_reachy_profile_and_returns_output():
    fake = FakeGateway(statuses=[{"status": "running"}, {"status": "completed",
                                                          "output": " Canberra. "}])
    out = gateway(fake).run("capital of Australia", requester="Yang", timeout_s=60,
                            requester_id="p1")
    assert out == "Canberra."
    method, url, headers, body = fake.requests[0]
    assert (method, url) == ("POST", "http://127.0.0.1:8642/p/reachy/v1/runs")
    assert headers["Authorization"] == "Bearer " + "k" * 32
    assert headers["X-Hermes-Session-Key"] == "reachy:p1"     # per-teammate Hermes memory
    assert headers["Idempotency-Key"]
    assert body["author"] == {"name": "Yang"}
    assert "capital of Australia" in body["input"] and "Yang" in body["input"]  # build_prompt
    assert fake.requests[1][1] == "http://127.0.0.1:8642/p/reachy/v1/runs/run_1"


def test_anonymous_requester_gets_no_session_key():
    fake = FakeGateway()
    gateway(fake).run("x", requester=None, timeout_s=60)
    assert "X-Hermes-Session-Key" not in fake.requests[0][2]


@pytest.mark.parametrize("final", [
    {"status": "failed", "error": "tool crashed"},
    {"status": "cancelled"},
    {"status": "interrupted", "error": "Gateway shutdown interrupted the run."},
    {"status": "completed", "output": "   "},
])
def test_unsuccessful_runs_raise(final):
    with pytest.raises(HermesError):
        gateway(FakeGateway(statuses=[final])).run("x", requester=None, timeout_s=60)


def test_timeout_stops_the_run():
    fake = FakeGateway(statuses=[{"status": "running"}])
    with pytest.raises(HermesError, match="timed out"):
        gateway(fake).run("x", requester=None, timeout_s=3)
    assert fake.requests[-1][:2] == ("POST", "http://127.0.0.1:8642/p/reachy/v1/runs/run_1/stop")


def test_waiting_for_approval_is_stopped_never_approved():
    fake = FakeGateway(statuses=[{"status": "waiting_for_approval"}])
    with pytest.raises(HermesError, match="approval"):
        gateway(fake).run("x", requester=None, timeout_s=60)
    assert not any("/approval" in url for _, url, _, _ in fake.requests)
    assert fake.requests[-1][1].endswith("/run_1/stop")


@pytest.mark.parametrize(("code", "match"), [(401, "auth"), (429, "busy"), (404, "404"),
                                             (500, "500")])
def test_create_errors_raise(code, match):
    fake = FakeGateway(create=(code, {"error": {"message": "nope"}}))
    with pytest.raises(HermesError, match=match):
        gateway(fake).run("x", requester=None, timeout_s=60)


def test_connection_failure_raises_hermes_error():
    def down(method, url, headers, body):
        raise OSError("connection refused")

    with pytest.raises(HermesError, match="unreachable"):
        gateway(down).run("x", requester=None, timeout_s=60)


def test_terminate_stops_the_in_flight_run_and_is_safe_when_idle():
    fake = FakeGateway()
    gw = gateway(fake)
    gw.terminate()                                  # idle: no request, no error
    assert fake.requests == []
    gw._run_id = "run_9"                            # as if a run were in flight
    gw.terminate()
    assert fake.requests[-1][1].endswith("/p/reachy/v1/runs/run_9/stop")


def test_check_toolsets_accepts_the_allowed_set():
    gateway(FakeGateway()).check_toolsets(allowed={"web", "browser", "memory", "skills",
                                                    "session_search", "todo", "cronjob",
                                                    "kanban"})


def test_check_toolsets_rejects_anything_extra():
    risky = {"data": TOOLSETS_OK["data"] + [{"name": "terminal", "enabled": True}]}
    with pytest.raises(HermesError, match="terminal"):
        gateway(FakeGateway(toolsets=risky)).check_toolsets(allowed={"web"} | {
            "browser", "memory", "skills", "session_search", "todo", "cronjob", "kanban"})


def test_empty_profile_or_key_rejected():
    with pytest.raises(ValueError):
        HermesGateway(base_url="http://x", profile=" ", api_key="k" * 32)
    with pytest.raises(ValueError):
        HermesGateway(base_url="http://x", profile="reachy", api_key="")

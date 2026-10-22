from types import SimpleNamespace

from reachy_vec.cli import doctor as d


def cfg(**kw):
    base = dict(llm_base_url="http://x/v1", llm_api_key="k", llm_model="M",
                lancedb_dir=None, stt_backend="parakeet", tts_backend="kokoro",
                kokoro_voice="bm_george", voice_sample=None, mic_source="mac",
                media_source="auto", robot_host="reachy-mini.local", robot_port=8000,
                hermes_enabled=True, hermes_transport="gateway", hermes_bin="hermes",
                hermes_gateway_url="http://127.0.0.1:8642", hermes_profile="reachy",
                hermes_api_key="h" * 32,
                hermes_toolsets="web,browser,memory,skills,session_search,todo,cronjob,kanban")
    return SimpleNamespace(**{**base, **kw})


def routes(table):
    """Fake http: URL suffix -> (status, json) or an exception to raise."""
    calls = []

    def http(url, method="GET", headers=None, body=None, timeout=5.0):
        calls.append((method, url))
        for suffix, result in table.items():
            if url.endswith(suffix):
                if isinstance(result, Exception):
                    raise result
                return result
        return 404, {}

    http.calls = calls
    return http


def test_llm_ok_and_timed():
    http = routes({"/models": (200, {"data": [{"id": "M"}]}),
                   "/chat/completions": (200, {"choices": []})})
    ticks = iter([0.0, 1.4])
    [c] = d.check_llm(cfg(), http=http, clock=lambda: next(ticks))
    assert c.status == d.OK and "1.4s" in c.detail


def test_llm_model_missing_and_unreachable_and_oom():
    assert d.check_llm(cfg(), http=routes({"/models": (200, {"data": [{"id": "Other"}]})})
                       )[0].status == d.FAIL
    assert "start oMLX" in d.check_llm(cfg(), http=routes({"/models": OSError("refused")})
                                       )[0].fix
    oom = routes({"/models": (200, {"data": [{"id": "M"}]}),
                  "/chat/completions": (507, {"error": {"message": "projected memory 46GB"}})})
    assert "free RAM" in d.check_llm(cfg(), http=oom)[0].fix


def test_llm_key_never_appears_in_output():
    http = routes({"/models": (401, {})})
    c = d.check_llm(cfg(llm_api_key="SECRET-KEY"), http=http)[0]
    assert "SECRET-KEY" not in c.detail + c.fix


def test_store_counts(tmp_path):
    class FakeStore:
        def __init__(self, path):
            pass

        def doc_count(self):
            return 40

        def dump_tables(self):
            return {"voices": [{"name": "Yang"}, {"name": "Yang"}]}

    docs, voices = d.check_store(cfg(lancedb_dir=tmp_path), store_factory=FakeStore)
    assert docs.status == d.OK and voices.detail == "1 enrolled: Yang"


def test_packages_missing_misaki_and_parakeet():
    out = d.check_packages(cfg(), importable=lambda m: False)
    assert [c.status for c in out] == [d.FAIL, d.FAIL]
    assert "misaki" in out[1].fix and "parakeet-mlx" in out[0].fix


def test_mic_missing_falls_back_warning_with_robot():
    assert d.check_mic(cfg(), input_name=lambda: None)[0].status == d.WARN
    assert d.check_mic(cfg(robot_host=None), input_name=lambda: None)[0].status == d.FAIL
    assert "Yeti X" in d.check_mic(cfg(), input_name=lambda: "Yeti X")[0].detail


def test_robot_motors_off_warns_and_fix_enables():
    state = {"mode": "disabled"}

    def http(url, method="GET", headers=None, body=None, timeout=5.0):
        if url.endswith("/daemon/status"):
            return 200, {"state": "running"}
        if url.endswith("/motors/set_mode/enabled"):
            state["mode"] = "enabled"
            return 200, {}
        return 200, {"mode": state["mode"]}

    assert d.check_robot(cfg(), http=http)[1].status == d.WARN
    assert d.check_robot(cfg(), http=http, fix=True)[1].status == d.OK


def test_robot_unreachable():
    assert d.check_robot(cfg(), http=routes({"/daemon/status": OSError("no route")})
                         )[0].status == d.FAIL


def test_duplicate_runs():
    one = ["38620 uv run --no-sync reachy-vec run"]
    two = one + ["32663 uv run reachy-vec run --source robot"]
    assert d.check_duplicates(list_runs=lambda: one)[0].status == d.OK
    assert d.check_duplicates(list_runs=lambda: two)[0].status == d.FAIL


def test_hermes_gateway_safe_and_unsafe_toolsets():
    safe = [{"name": n, "enabled": True} for n in ("web", "browser", "memory")]
    http = routes({"/health": (200, {}), "/v1/toolsets": (200, {"data": safe})})
    assert d.check_hermes(cfg(), http=http)[0].status == d.OK
    risky = safe + [{"name": "terminal", "enabled": True}]
    http = routes({"/health": (200, {}), "/v1/toolsets": (200, {"data": risky})})
    c = d.check_hermes(cfg(), http=http)[0]
    assert c.status == d.FAIL and "terminal" in c.detail


def test_a_crashing_check_does_not_hide_the_others():
    def boom(**kw):
        raise RuntimeError("bad")

    def fine():
        return [d.Check("x", d.OK, "fine")]

    out = d.run_checks(checks=[boom, fine])
    assert [c.status for c in out] == [d.FAIL, d.OK]

"""reachy-vec doctor: check everything a working Reachy needs, say what to fix.

Read-only by default (safe while `run` is running: the robot is queried over
its REST API only, never through a second SDK media connection). --fix applies
the one safe repair: switching the robot's motors on.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import typer

from reachy_vec.config import settings

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "✓", WARN: "!", FAIL: "✗"}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def http_json(url: str, *, method: str = "GET", headers=None, body=None, timeout=5.0):
    """(status, parsed JSON) - HTTP errors come back as their status."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


# -- individual checks (dependencies injectable for tests) -----------------------


def check_llm(cfg=settings, http=http_json, clock=time.monotonic) -> list[Check]:
    if not cfg.llm_base_url:
        return [Check("LLM", WARN, f"OpenAI cloud ({cfg.llm_model}) - no local server set",
                      "set REACHY_VEC_LLM_BASE_URL to use oMLX")]
    base = cfg.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}
    try:
        status, data = http(f"{base}/models", headers=headers)
    except OSError as e:
        return [Check("LLM server", FAIL, f"{base} unreachable ({e})",
                      "start oMLX.app and wait for the model to load")]
    if status == 401:
        return [Check("LLM server", FAIL, f"{base} rejected the key",
                      "set REACHY_VEC_LLM_API_KEY to the oMLX API key")]
    models = [m.get("id") for m in data.get("data", [])]
    if cfg.llm_model not in models:
        return [Check("LLM model", FAIL, f"{cfg.llm_model!r} not offered by oMLX",
                      f"pick one of {models} in REACHY_VEC_LLM_MODEL")]
    t0 = clock()
    try:
        status, reply = http(f"{base}/chat/completions", method="POST", headers=headers,
                             timeout=60, body={
                                 "model": cfg.llm_model, "max_tokens": 3,
                                 "messages": [{"role": "user", "content": "Say ok."}],
                                 "chat_template_kwargs": {"enable_thinking": False}})
    except OSError as e:
        return [Check("LLM reply", FAIL, f"no reply ({e})", "check oMLX memory / logs")]
    took = clock() - t0
    if status != 200:
        message = reply.get("error", {}).get("message", "")[:120]
        fix = ("free RAM: unload other oMLX models, close apps"
               if "memory" in message.lower() else "check the oMLX logs")
        return [Check("LLM reply", FAIL, f"HTTP {status}: {message}", fix)]
    return [Check("LLM", OK, f"{cfg.llm_model} on oMLX answered in {took:.1f}s")]


def check_store(cfg=settings, store_factory=None) -> list[Check]:
    from reachy_vec.store.db import Store

    store = (store_factory or Store)(cfg.lancedb_dir)
    docs = store.doc_count()
    voices = store.dump_tables().get("voices", [])
    people = sorted({v["name"] for v in voices})
    out = [Check("Knowledge base", OK if docs else FAIL, f"{docs} chunks",
                 "" if docs else "run: uv run --no-sync reachy-vec sync-mongo")]
    out.append(Check("Voices", OK if people else WARN,
                     f"{len(people)} enrolled: {', '.join(people)}" if people
                     else "nobody enrolled", "" if people else 'say "Reachy, remember me"'))
    return out


def check_packages(cfg=settings, importable: Callable[[str], bool] | None = None,
                   exists: Callable[[str], bool] | None = None) -> list[Check]:
    import importlib.util
    from pathlib import Path

    importable = importable or (lambda m: importlib.util.find_spec(m) is not None)
    exists = exists or (lambda p: Path(p).is_file())
    out = []
    needs = {"parakeet": ("parakeet_mlx", "parakeet-mlx"),
             "local": ("faster_whisper", "faster-whisper")}.get(cfg.stt_backend)
    if needs:
        mod, pkg = needs
        out.append(Check(f"STT ({cfg.stt_backend})", OK if importable(mod) else FAIL,
                         f"{pkg} installed" if importable(mod) else f"{pkg} missing",
                         "" if importable(mod)
                         else f"uv pip install --python .venv/bin/python {pkg}"))
    if cfg.tts_backend == "kokoro":
        ok = importable("misaki")
        out.append(Check("TTS (kokoro)", OK if ok else FAIL,
                         f"voice {cfg.kokoro_voice}" + ("" if ok else ", misaki missing"),
                         "" if ok else 'uv pip install --python .venv/bin/python "misaki[en]"'))
    elif cfg.tts_backend == "qwen-tts":
        ok = bool(cfg.voice_sample) and exists(str(cfg.voice_sample))
        out.append(Check("TTS (qwen-tts)", OK if ok else FAIL,
                         f"voice sample {cfg.voice_sample}",
                         "" if ok else "uv run --no-sync reachy-vec record-voice"))
    else:
        out.append(Check(f"TTS ({cfg.tts_backend})", OK, "built in"))
    return out


def check_mic(cfg=settings, input_name: Callable[[], str | None] | None = None) -> list[Check]:
    if cfg.mic_source != "mac" and cfg.media_source == "robot":
        return [Check("Microphone", OK, "robot mic")]

    def default_input() -> str | None:
        import sounddevice as sd

        try:
            return sd.query_devices(kind="input")["name"]
        except Exception:
            return None

    name = (input_name or default_input)()
    if name:
        return [Check("Microphone", OK, f"Mac input: {name}")]
    status = WARN if cfg.robot_host else FAIL
    return [Check("Microphone", status, "no Mac microphone (USB mic unplugged?)",
                  "plug the mic straight into the Mac" +
                  (" - run will fall back to the robot mic" if cfg.robot_host else ""))]


def check_robot(cfg=settings, http=http_json, fix: bool = False) -> list[Check]:
    if not cfg.robot_host:
        return [Check("Robot", WARN, "REACHY_VEC_ROBOT_HOST not set - body-less / simulator")]
    base = f"http://{cfg.robot_host}:{cfg.robot_port}/api"
    try:
        _, daemon = http(f"{base}/daemon/status")
        _, motors = http(f"{base}/motors/status")
    except OSError as e:
        return [Check("Robot", FAIL, f"{cfg.robot_host} unreachable ({e})",
                      "switch the robot on; check it is on the same Wi-Fi")]
    out = [Check("Robot daemon", OK if daemon.get("state") == "running" else FAIL,
                 f"state {daemon.get('state')!r}",
                 "" if daemon.get("state") == "running" else
                 f"curl -X POST {base}/daemon/restart")]
    mode = motors.get("mode")
    if mode != "enabled" and fix:
        try:
            http(f"{base}/motors/set_mode/enabled", method="POST", timeout=20)
            mode = http(f"{base}/motors/status")[1].get("mode")
        except OSError:
            pass
    out.append(Check("Motors", OK if mode == "enabled" else WARN, f"mode {mode!r}",
                     "" if mode == "enabled" else
                     "reachy-vec doctor --fix (run also switches them on at connect)"))
    return out


def check_duplicates(list_runs: Callable[[], list[str]] | None = None) -> list[Check]:
    def pgrep() -> list[str]:
        import subprocess

        out = subprocess.run(["pgrep", "-fl", "reachy-vec run"], capture_output=True,
                             text=True).stdout
        return [line for line in out.splitlines() if "python" in line or "uv run" in line]

    runs = [r for r in (list_runs or pgrep)() if " uv run" in r or r.split()[1:2] == ["uv"]]
    if len(runs) > 1:
        return [Check("Reachy copies", FAIL, f"{len(runs)} `reachy-vec run` running - they "
                      "fight over the robot's mic/speaker",
                      "stop all but one (check Hermes did not start one)")]
    return [Check("Reachy copies", OK, f"{len(runs)} running")]


def check_hermes(cfg=settings, http=http_json, which=None) -> list[Check]:
    import shutil

    if not cfg.hermes_enabled:
        return [Check("Hermes", WARN, "disabled", "REACHY_VEC_HERMES_ENABLED=true")]
    if cfg.hermes_transport != "gateway":
        found = (which or shutil.which)(cfg.hermes_bin)
        return [Check("Hermes (cli)", OK if found else FAIL, found or "not on PATH",
                      "" if found else "install Hermes or fix REACHY_VEC_HERMES_BIN")]
    base = f"{cfg.hermes_gateway_url.rstrip('/')}"
    try:
        status, _ = http(f"{base}/health")
    except OSError as e:
        return [Check("Hermes gateway", FAIL, f"{base} unreachable ({e})",
                      "hermes gateway restart  (run falls back to the hermes CLI)")]
    if not cfg.hermes_api_key:
        return [Check("Hermes gateway", FAIL, "no API key",
                      f"API_SERVER_KEY in ~/.hermes/profiles/{cfg.hermes_profile}/.env")]
    status, data = http(f"{base}/p/{cfg.hermes_profile}/v1/toolsets",
                        headers={"Authorization": f"Bearer {cfg.hermes_api_key}"})
    if status != 200:
        return [Check("Hermes gateway", FAIL, f"/p/{cfg.hermes_profile} answered {status}",
                      "check API_SERVER_KEY / API_SERVER_ENABLED for the profile")]
    enabled = {t["name"] for t in data.get("data", []) if t.get("enabled")}
    extra = sorted(enabled - set(cfg.hermes_toolsets.split(",")))
    if extra:
        return [Check("Hermes toolsets", FAIL, f"unsafe extras enabled: {', '.join(extra)}",
                      f"hermes -p {cfg.hermes_profile} config set platform_toolsets.api_server "
                      f'"[{cfg.hermes_toolsets.replace(",", ", ")}]"')]
    return [Check("Hermes", OK, f"gateway, profile {cfg.hermes_profile}, "
                                f"{len(enabled)} read-only/scheduling toolsets")]


CHECKS: list[Callable[..., list[Check]]] = [
    check_llm, check_store, check_packages, check_mic, check_robot, check_duplicates,
    check_hermes,
]


def run_checks(checks=None, fix: bool = False) -> list[Check]:
    results = []
    for check in checks or CHECKS:
        try:
            results += check(fix=fix) if check is check_robot else check()
        except Exception as e:  # a broken check must not hide the others
            results.append(Check(check.__name__.removeprefix("check_"), FAIL, f"crashed: {e}"))
    return results


def doctor(fix: bool = typer.Option(False, "--fix", help="Switch the robot's motors on.")):
    """Check oMLX, knowledge base, speech packages, mic, robot, Hermes - and say what to fix."""
    from dotenv import load_dotenv

    load_dotenv()
    results = run_checks(fix=fix)
    width = max(len(r.name) for r in results)
    for r in results:
        typer.echo(f"{MARK[r.status]} {r.name:<{width}}  {r.detail}")
        if r.fix and r.status != OK:
            typer.echo(f"  {'':<{width}}  -> {r.fix}")
    failed = [r for r in results if r.status == FAIL]
    typer.echo(f"\n{len(failed)} problem(s), "
               f"{sum(r.status == WARN for r in results)} warning(s).")
    raise typer.Exit(code=1 if failed else 0)

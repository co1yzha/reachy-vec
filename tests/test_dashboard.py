import json
import threading
import urllib.error
import urllib.request

import pytest

from reachy_vec.cli.dashboard import make_server, render_page
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk, MemoryRow, VoiceRow
from tests.conftest import FakeEmbedder


def seeded_store(tmp_path) -> Store:
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder()
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id="d1",
                text="the nightly pipeline runs at 02:00",
                vector=embedder.embed(["pipeline"])[0],
                source="notes.md",
                ingested_at="2026-07-08T00:00:00+00:00",
            )
        ]
    )
    store.add_voice_rows(
        [
            VoiceRow(
                voice_id="p1:abc",
                person_id="p1",
                name="Yang",
                vector=[0.1] * 192,
                created_at="2026-07-08T00:00:00+00:00",
                source="passive",
            )
        ]
    )
    store.add_memories(
        [
            MemoryRow(
                memory_id="m1",
                person_id="p1",
                text="Yang prefers short answers",
                vector=embedder.embed(["short"])[0],
                created_at="2026-07-08T00:00:00+00:00",
            )
        ]
    )
    return store


def test_dump_tables_covers_all_tables_and_truncates_vectors(tmp_path):
    dump = seeded_store(tmp_path).dump_tables()
    assert set(dump) == {"docs", "voices", "memories", "greetings", "messages"}
    assert dump["messages"] == []
    doc = dump["docs"][0]
    assert doc["text"] == "the nightly pipeline runs at 02:00"
    assert doc["vector"]["dim"] == 384
    assert len(doc["vector"]["head"]) <= 4
    voice = dump["voices"][0]
    assert voice["source"] == "passive"
    assert voice["vector"] == {"dim": 192, "head": [0.1, 0.1, 0.1, 0.1]}


def test_render_page_embeds_table_metadata():
    page = render_page()
    assert "__META__" not in page
    assert "ECAPA" in page and "Reachy" in page


def test_server_serves_page_and_live_json(tmp_path):
    store = seeded_store(tmp_path)
    server = make_server(store, "127.0.0.1", 0)  # port 0 = any free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            assert "text/html" in response.headers["Content-Type"]
            assert "Reachy" in response.read().decode()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tables") as response:
            data = json.loads(response.read())
        assert data["memories"][0]["text"] == "Yang prefers short answers"
        # a second fetch re-reads the store: new rows appear without restart
        store.add_memories(
            [
                MemoryRow(
                    memory_id="m2",
                    person_id="p1",
                    text="Yang is preparing demo day",
                    vector=FakeEmbedder().embed(["demo"])[0],
                    created_at="2026-07-08T01:00:00+00:00",
                )
            ]
        )
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tables") as response:
            assert len(json.loads(response.read())["memories"]) == 2
    finally:
        server.shutdown()
        server.server_close()


# -- Live tab: tailing data/reachy.log ---------------------------------------------

LOG = """\
2026-10-22 14:15:53,799 reachy_vec.audio.listen INFO heard (parakeet): 'Can you play music?'
2026-10-22 14:15:55,561 reachy_vec.brain.router INFO route open ('https://youtube.com') for 'x'
2026-10-22 14:15:55,664 reachy_vec.brain.chat INFO open_url: https://youtube.com
2026-10-22 14:15:56,500 reachy_vec.brain.chat INFO reply to Yang: "I've opened YouTube."
2026-10-22 14:16:00,000 reachy_vec.brain.oracle ERROR brain.respond failed
Traceback (most recent call last):
  File "chat.py", line 1
2026-10-22 14:17:23,203 reachy_vec.audio.listen INFO barge-in fired: 0.7s of speech
"""


def test_parse_log_classifies_events_and_folds_tracebacks():
    from reachy_vec.cli.dashboard import parse_log

    events = parse_log(LOG)
    assert [e["kind"] for e in events] == ["heard", "route", "action", "reply", "error", "barge"]
    assert events[0] == {"time": "2026-10-22 14:15:53", "level": "INFO", "kind": "heard",
                         "source": "audio.listen",
                         "text": "heard (parakeet): 'Can you play music?'"}
    assert "Traceback" in events[4]["text"] and "chat.py" in events[4]["text"]


def test_mic_lines_are_their_own_kind():
    from reachy_vec.cli.dashboard import parse_log

    line = ("2026-10-22 14:00:00,000 reachy_vec.audio.listen INFO "
            "mic: Mac input 'Wireless Microphone RX'\n")
    assert parse_log(line)[0]["kind"] == "mic"


START = "2026-10-22 14:15:00,000 reachy_vec.cli.run INFO run started: source=mac\n"


def test_live_view_starts_at_the_current_run(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    path.write_text(LOG + START + LOG)          # an old run, then the current one
    first = read_log(path, offset=-1)
    assert [e["kind"] for e in first["events"]][:2] == ["start", "heard"]
    assert len(first["events"]) == 7            # the marker + the current run only


def test_live_view_without_a_run_marker_only_streams_new_lines(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    path.write_text(LOG)                        # history from before markers existed
    first = read_log(path, offset=-1)
    assert first["events"] == []
    with path.open("a") as f:
        f.write("2026-10-22 14:18:00,000 reachy_vec.brain.oracle INFO wake word heard: 'Reachy'\n")
    second = read_log(path, offset=first["offset"])
    assert [e["kind"] for e in second["events"]] == ["wake"]
    assert read_log(path, offset=second["offset"])["events"] == []


def test_read_log_holds_back_a_half_written_line(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    path.write_text(START + LOG + "2026-10-22 14:18:00,000 reachy_vec.x INFO partial")
    result = read_log(path, offset=-1)
    assert len(result["events"]) == 7
    assert result["offset"] == len((START + LOG).encode())


def test_read_log_restarts_when_the_log_is_truncated_or_missing(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    assert read_log(path, offset=-1) == {"offset": 0, "events": [], "reset": True}
    path.write_text(LOG)
    result = read_log(path, offset=10**9)       # file was deleted and restarted
    assert result["reset"] and len(result["events"]) == 6


@pytest.mark.parametrize(("message", "kind"), [
    ("run started: source=mac", "start"),
    ("interaction ended: conversation", "end"),
    ("not called for 300s - going to sleep", "sleep"),
    ("wake word while asleep - waking up", "wake"),
])
def test_session_boundaries_get_their_own_kinds(message, kind):
    from reachy_vec.cli.dashboard import parse_log

    line = f"2026-10-22 14:00:00,000 reachy_vec.brain.oracle INFO {message}\n"
    assert parse_log(line)[0]["kind"] == kind


def test_server_serves_live_page_and_log(tmp_path):
    log = tmp_path / "reachy.log"
    log.write_text(START + LOG)
    server = make_server(seeded_store(tmp_path), "127.0.0.1", 0, log_path=log)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/live").read().decode()
        assert "Reachy live" in page
        data = json.loads(urllib.request.urlopen(f"{base}/api/log?offset=-1").read())
        assert [e["kind"] for e in data["events"]][:3] == ["start", "heard", "route"]
    finally:
        server.shutdown()
        server.server_close()


def test_start_dashboard_serves_in_the_background_and_opens_live(tmp_path, monkeypatch):
    from reachy_vec.cli import dashboard

    opened = []
    monkeypatch.setattr(dashboard.webbrowser, "open", opened.append)
    server = dashboard.start_dashboard(seeded_store(tmp_path), "127.0.0.1", 0,
                                       page="/live", open_browser=True, delay_s=0)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/live"
        assert urllib.request.urlopen(url).status == 200
        assert opened == [url]
    finally:
        server.shutdown()
        server.server_close()


def test_start_dashboard_returns_none_when_the_port_is_taken(tmp_path):
    from reachy_vec.cli.dashboard import start_dashboard

    first = start_dashboard(seeded_store(tmp_path), "127.0.0.1", 0, open_browser=False)
    try:
        port = first.server_address[1]
        assert start_dashboard(seeded_store(tmp_path), "127.0.0.1", port,
                               open_browser=False) is None
    finally:
        first.shutdown()
        first.server_close()


# -- ports: Reachy's services and everything listening on the Mac ------------------

LSOF = """\
COMMAND     PID            USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1  1511 vec-datascience    9u  IPv4 0x1      0t0  TCP 127.0.0.1:8001 (LISTEN)
python3.1 84745 vec-datascience    7u  IPv4 0x2      0t0  TCP 127.0.0.1:8642 (LISTEN)
python3.1 84745 vec-datascience    8u  IPv6 0x3      0t0  TCP [::1]:8642 (LISTEN)
ControlCe   612 vec-datascience   10u  IPv4 0x4      0t0  TCP *:7000 (LISTEN)
"""


def test_parse_lsof_lists_each_listening_port_once_per_process():
    from reachy_vec.cli.dashboard import parse_lsof

    assert parse_lsof(LSOF) == [
        {"port": 7000, "address": "*", "command": "ControlCe", "pid": 612},
        {"port": 8001, "address": "127.0.0.1", "command": "python3.1", "pid": 1511},
        {"port": 8642, "address": "127.0.0.1, [::1]", "command": "python3.1", "pid": 84745},
    ]


def test_reachy_services_report_their_ports_and_whether_they_answer(monkeypatch):
    from reachy_vec.cli import dashboard

    monkeypatch.setattr(dashboard.settings, "robot_host", "reachy-mini.local")
    monkeypatch.setattr(dashboard.settings, "llm_base_url", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(dashboard.settings, "hermes_gateway_url", "http://127.0.0.1:8642")
    up = {("127.0.0.1", 8001), ("127.0.0.1", 8400)}
    services = dashboard.reachy_services(8400, is_open=lambda host, port: (host, port) in up)
    assert [(s["name"], s["host"], s["port"], s["up"]) for s in services] == [
        ("Robot daemon", "reachy-mini.local", 8000, False),
        ("LLM (oMLX)", "127.0.0.1", 8001, True),
        ("Hermes gateway", "127.0.0.1", 8642, False),
        ("Dashboard", "127.0.0.1", 8400, True),
    ]


def test_llm_on_openai_cloud_is_not_a_local_port(monkeypatch):
    from reachy_vec.cli import dashboard

    monkeypatch.setattr(dashboard.settings, "llm_base_url", None)
    services = dashboard.reachy_services(8400, is_open=lambda host, port: True)
    llm = next(s for s in services if s["name"].startswith("LLM"))
    assert llm["host"] == "api.openai.com" and llm["port"] == 443


def test_kill_port_only_kills_the_process_listening_there(monkeypatch):
    from reachy_vec.cli import dashboard

    monkeypatch.setattr(dashboard, "listening_ports", lambda: dashboard.parse_lsof(LSOF))
    killed = []
    kill = lambda pid, sig: killed.append(pid)  # noqa: E731
    assert dashboard.kill_port(8001, 1511, kill=kill) == "stopped python3.1 (pid 1511) on 8001"
    assert dashboard.kill_port(8001, 612, kill=kill).startswith("refused")   # wrong pid
    assert dashboard.kill_port(9999, 1511, kill=kill).startswith("refused")  # not listening
    assert killed == [1511]


def test_kill_port_refuses_the_dashboard_itself(monkeypatch):
    import os

    from reachy_vec.cli import dashboard

    me = os.getpid()
    monkeypatch.setattr(dashboard, "listening_ports", lambda: [
        {"port": 8400, "address": "127.0.0.1", "command": "python", "pid": me}])
    assert dashboard.kill_port(8400, me, kill=lambda p, s: None).startswith("refused")


def test_kill_endpoint_rejects_cross_site_requests(tmp_path, monkeypatch):
    from reachy_vec.cli import dashboard

    calls = []
    monkeypatch.setattr(dashboard, "kill_port", lambda port, pid: calls.append(pid) or "ok")
    server = make_server(seeded_store(tmp_path), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(origin):
        req = urllib.request.Request(f"{base}/api/ports/kill", method="POST",
                                     data=json.dumps({"port": 8001, "pid": 1511}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Origin": origin})
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as err:
            return err.code

    try:
        assert post("https://evil.example") == 403
        assert post(base) == 200
        assert calls == [1511]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("args", "label"), [
    ("omlx-server", "oMLX server (LLM)"),
    ("/x/.venv/bin/python /x/.venv/bin/reachy-vec run --dashboard", "reachy-vec run --dashboard"),
    ("/x/.venv/bin/python /x/.venv/bin/reachy-vec dashboard --live", "reachy-vec dashboard --live"),
    ("/Users/me/.hermes/tools/python-3.14/bin/python3 -I -c import os", "Hermes"),
    ("/usr/bin/python3 -m http.server 8499", "python3 -m http.server 8499"),
    ("/Applications/Cursor.app/Contents/MacOS/Cursor --type=renderer", "Cursor"),
])
def test_process_labels_say_what_the_process_is(args, label):
    from reachy_vec.cli.dashboard import process_label

    assert process_label(args) == label


def test_show_earlier_loads_the_previous_run_each_time(tmp_path):
    from reachy_vec.cli.dashboard import read_earlier, read_log

    path = tmp_path / "reachy.log"
    old = LOG.replace("14:1", "09:1")
    path.write_text(START.replace("14:15", "09:00") + old + START + LOG)
    current = read_log(path, offset=-1)
    assert current["start"] == len((START + old).encode())     # where this run begins
    earlier = read_earlier(path, before=current["start"])
    assert earlier["events"][0]["kind"] == "start"
    assert earlier["events"][1]["time"].startswith("2026-10-22 09:1")
    assert earlier["start"] == 0 and earlier["more"] is False


def test_show_earlier_without_markers_pages_back_in_chunks(tmp_path):
    from reachy_vec.cli.dashboard import read_earlier

    path = tmp_path / "reachy.log"
    path.write_text(LOG * 3)
    size = path.stat().st_size
    page = read_earlier(path, before=size, chunk_bytes=len(LOG.encode()) + 10)
    assert 0 < page["start"] < size and page["more"] is True
    assert page["events"] and page["events"][0]["kind"] in ("heard", "route", "action")


def test_show_earlier_continues_into_the_previous_days_file(tmp_path):
    from reachy_vec.cli.dashboard import read_earlier_logs, read_log

    path = tmp_path / "reachy.log"
    (tmp_path / "reachy.log.2026-10-20").write_text(START.replace("14:15", "08:00") + LOG)
    (tmp_path / "reachy.log.2026-10-21").write_text(START.replace("14:15", "09:00") + LOG)
    path.write_text(START + LOG)
    current = read_log(path, offset=-1)
    assert current["start"] == 0 and current["older"] is True

    page = read_earlier_logs(path, "", before=0)        # today is used up: go to 10-21
    assert page["file"] == "reachy.log.2026-10-21"
    assert page["events"][0]["time"].startswith("2026-10-22 09:00")
    assert page["start"] == 0 and page["more"] is True

    page = read_earlier_logs(path, page["file"], before=page["start"])
    assert page["file"] == "reachy.log.2026-10-20" and page["more"] is False


def test_show_earlier_ignores_files_that_are_not_rotated_logs(tmp_path):
    from reachy_vec.cli.dashboard import read_earlier_logs

    path = tmp_path / "reachy.log"
    path.write_text(START + LOG)
    (tmp_path / "reachy.log.bak").write_text(LOG)
    page = read_earlier_logs(path, "../secrets", before=0)   # unknown name = today
    assert page["file"] == "" and page["events"] == [] and page["more"] is False

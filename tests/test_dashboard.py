import json
import threading
import urllib.request

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


def test_read_log_returns_only_new_lines(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    path.write_text(LOG)
    first = read_log(path, offset=-1)
    assert len(first["events"]) == 6
    with path.open("a") as f:
        f.write("2026-10-22 14:18:00,000 reachy_vec.brain.oracle INFO wake word heard: 'Reachy'\n")
    second = read_log(path, offset=first["offset"])
    assert [e["kind"] for e in second["events"]] == ["wake"]
    assert read_log(path, offset=second["offset"])["events"] == []


def test_read_log_holds_back_a_half_written_line(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    path.write_text(LOG + "2026-10-22 14:18:00,000 reachy_vec.x INFO partial")
    result = read_log(path, offset=-1)
    assert len(result["events"]) == 6
    assert result["offset"] == len(LOG.encode())


def test_read_log_restarts_when_the_log_is_truncated_or_missing(tmp_path):
    from reachy_vec.cli.dashboard import read_log

    path = tmp_path / "reachy.log"
    assert read_log(path, offset=-1) == {"offset": 0, "events": [], "reset": True}
    path.write_text(LOG)
    result = read_log(path, offset=10**9)       # file was deleted and restarted
    assert result["reset"] and len(result["events"]) == 6


def test_server_serves_live_page_and_log(tmp_path):
    log = tmp_path / "reachy.log"
    log.write_text(LOG)
    server = make_server(seeded_store(tmp_path), "127.0.0.1", 0, log_path=log)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/live").read().decode()
        assert "Reachy live" in page
        data = json.loads(urllib.request.urlopen(f"{base}/api/log?offset=-1").read())
        assert [e["kind"] for e in data["events"]][:2] == ["heard", "route"]
    finally:
        server.shutdown()
        server.server_close()

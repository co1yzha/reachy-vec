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
    assert set(dump) == {"docs", "people", "voices", "memories", "greetings", "messages"}
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

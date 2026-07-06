from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import FakeEmbedder


def chunk(cid: str, text: str, source: str) -> DocChunk:
    return DocChunk(
        chunk_id=cid,
        text=text,
        vector=FakeEmbedder().embed([text])[0],
        source=source,
        ingested_at="2026-07-06T00:00:00+00:00",
    )


def test_scored_search_returns_high_score_for_exact_match(tmp_path):
    store = Store(tmp_path / "db")
    store.add_doc_chunks(
        [chunk("c1", "food mapping demo", "demo: Food"), chunk("c2", "other", "notes.md")]
    )
    results = store.search_docs_scored(FakeEmbedder().embed(["food mapping demo"])[0], k=1)
    (hit, score), = results
    assert hit.chunk_id == "c1"
    assert score > 0.99


def test_scored_search_empty_table(tmp_path):
    assert Store(tmp_path / "db").search_docs_scored(FakeEmbedder().embed(["x"])[0]) == []


def test_delete_by_source_prefix_removes_only_matching(tmp_path):
    store = Store(tmp_path / "db")
    store.add_doc_chunks(
        [
            chunk("c1", "alpha", "demo: Food Mapping"),
            chunk("c2", "beta", "demo: Robot Arm"),
            chunk("c3", "gamma", "notes.md"),
        ]
    )
    store.delete_docs_by_source_prefix("demo: ")
    assert store.doc_count() == 1
    remaining = store.search_docs(FakeEmbedder().embed(["gamma"])[0], k=5)
    assert [r.chunk_id for r in remaining] == ["c3"]

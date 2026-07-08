from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk
from tests.conftest import FakeEmbedder


def make_chunk(chunk_id: str, text: str, embedder: FakeEmbedder) -> DocChunk:
    return DocChunk(
        chunk_id=chunk_id,
        text=text,
        vector=embedder.embed([text])[0],
        source="notes.md",
        ingested_at="2026-07-06T00:00:00+00:00",
    )


def test_add_and_count(tmp_path):
    store = Store(tmp_path / "lancedb")
    embedder = FakeEmbedder()
    assert store.doc_count() == 0
    store.add_doc_chunks([make_chunk("c1", "the pipeline runs nightly", embedder)])
    assert store.doc_count() == 1


def test_search_returns_nearest_chunk(tmp_path):
    store = Store(tmp_path / "lancedb")
    embedder = FakeEmbedder()
    store.add_doc_chunks(
        [
            make_chunk("c1", "the pipeline runs nightly", embedder),
            make_chunk("c2", "lunch is at noon on fridays", embedder),
        ]
    )
    query_vector = embedder.embed(["the pipeline runs nightly"])[0]
    hits = store.search_docs(query_vector, k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"


def test_search_empty_table_returns_no_hits(tmp_path):
    store = Store(tmp_path / "lancedb")
    query_vector = FakeEmbedder().embed(["anything"])[0]
    assert store.search_docs(query_vector) == []

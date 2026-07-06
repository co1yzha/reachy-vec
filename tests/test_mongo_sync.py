from reachy_vec.store.db import Store
from reachy_vec.store.mongo_sync import format_demo, sync_demos

from tests.conftest import FakeEmbedder

DEMO = {
    "_id": "abc123",
    "title": "Liverpool City Food Mapping",
    "project": "R&D",
    "url": "https://example.org/food",
    "authors": ["Yang Zhang"],
    "tags": ["foodmapping", "geospatial"],
    "note": "### Case Study: fighting food insecurity across Liverpool.",
}


def test_format_demo_includes_key_fields():
    text = format_demo(DEMO)
    assert "Liverpool City Food Mapping" in text
    assert "R&D" in text
    assert "Yang Zhang" in text
    assert "foodmapping" in text
    assert "https://example.org/food" in text
    assert "food insecurity" in text


def test_format_demo_tolerates_missing_fields():
    text = format_demo({"_id": "x", "title": "Bare Demo"})
    assert "Bare Demo" in text


def test_sync_is_idempotent_and_preserves_other_docs(tmp_path):
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder()
    # a pre-existing non-demo doc must survive syncs
    from reachy_vec.store.schemas import DocChunk

    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id="n1",
                text="regular note",
                vector=embedder.embed(["regular note"])[0],
                source="notes.md",
                ingested_at="2026-07-06T00:00:00+00:00",
            )
        ]
    )

    first = sync_demos([DEMO], store, embedder)
    assert first >= 1
    count_after_first = store.doc_count()

    second = sync_demos([DEMO], store, embedder)
    assert second == first
    assert store.doc_count() == count_after_first  # no duplicates

    hit = store.search_docs(embedder.embed([format_demo(DEMO)])[0], k=1)[0]
    assert hit.source == "demo: Liverpool City Food Mapping"
    assert store.doc_count() >= 2  # notes.md survived
    assert store.demo_titles() == ["Liverpool City Food Mapping"]

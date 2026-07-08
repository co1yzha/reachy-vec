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


def test_add_doc_chunks_creates_fts_index(tmp_path):
    store = Store(tmp_path / "db")
    store.add_doc_chunks([chunk("c1", "food mapping demo", "demo: Food")])
    names = [i.name for i in store._docs().list_indices()]
    assert "text_idx" in names


def test_hybrid_search_surfaces_keyword_only_match(tmp_path):
    # c1's vector is unrelated to the query vector; only BM25 on "Jane Smith"
    # can rank it first. RRF: c1 wins the text leg outright (c2 doesn't match),
    # so c1 ranks first regardless of the vector leg.
    store = Store(tmp_path / "db")
    store.add_doc_chunks(
        [
            chunk("c1", "Demo: Robot Arm Teleop\nAuthors: Jane Smith", "demo: Arm"),
            chunk("c2", "Demo: Food Mapping\nAuthors: Bob", "demo: Food"),
        ]
    )
    query_vector = FakeEmbedder().embed_query("completely unrelated words")
    results = store.search_docs_scored(query_vector, k=2, query_text="Jane Smith")
    assert results[0][0].chunk_id == "c1"


def test_hybrid_scores_are_cosine_zero_to_one(tmp_path):
    store = Store(tmp_path / "db")
    store.add_doc_chunks(
        [chunk("c1", "food mapping demo", "demo: Food"), chunk("c2", "other", "n.md")]
    )
    query_vector = FakeEmbedder().embed_query("food mapping demo")
    results = store.search_docs_scored(query_vector, k=2, query_text="food mapping demo")
    scores = {hit.chunk_id: score for hit, score in results}
    assert scores["c1"] > 0.99  # identical text -> identical fake vector
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_vector_fallback_when_no_fts_index(tmp_path):
    # Simulate a DB created before FTS indexing existed: write rows directly,
    # bypassing add_doc_chunks (which now builds the index).
    store = Store(tmp_path / "db")
    store._docs().add([chunk("c1", "food mapping demo", "demo: Food")])
    results = store.search_docs_scored(
        FakeEmbedder().embed_query("food mapping demo"), k=1, query_text="food"
    )
    (hit, score), = results
    assert hit.chunk_id == "c1"
    assert score > 0.99


def test_no_query_text_uses_vector_path(tmp_path):
    store = Store(tmp_path / "db")
    store.add_doc_chunks([chunk("c1", "food mapping demo", "demo: Food")])
    results = store.search_docs_scored(FakeEmbedder().embed(["food mapping demo"])[0])
    assert results[0][0].chunk_id == "c1"

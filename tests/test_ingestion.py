from reachy_vec.store.db import Store
from reachy_vec.store.ingestion import chunk_text, ingest_path
from tests.conftest import FakeEmbedder


def test_chunk_text_packs_paragraphs_under_limit():
    text = "para one.\n\npara two.\n\npara three."
    chunks = chunk_text(text, max_chars=22)
    assert chunks == ["para one.\n\npara two.", "para three."]


def test_chunk_text_splits_oversized_paragraph():
    text = "x" * 250
    chunks = chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_text_skips_blank_input():
    assert chunk_text("   \n\n  ") == []


def test_ingest_directory_writes_chunks_and_returns_count(tmp_path):
    docs_dir = tmp_path / "team-docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("alpha doc content")
    (docs_dir / "b.txt").write_text("beta doc content")
    (docs_dir / "ignored.pdf").write_text("binary-ish")

    store = Store(tmp_path / "lancedb")
    count = ingest_path(docs_dir, store, FakeEmbedder())

    assert count == 2
    assert store.doc_count() == 2
    hit = store.search_docs(FakeEmbedder().embed(["alpha doc content"])[0], k=1)[0]
    assert hit.source.endswith("a.md")
    assert hit.text == "alpha doc content"


def test_ingest_single_file(tmp_path):
    f = tmp_path / "solo.md"
    f.write_text("solo content")
    store = Store(tmp_path / "lancedb")
    assert ingest_path(f, store, FakeEmbedder()) == 1

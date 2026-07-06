# Phase 0b: Ingestion Pipeline + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reachy-vec ingest <path>` reads `.md`/`.txt` team docs, chunks them, embeds them, and writes them to the LanceDB `docs` table.

**Architecture:** A pure chunker (paragraph packing with a hard-split fallback) feeds `ingest_path`, which walks files, embeds via the `Embedder` protocol, and writes `DocChunk` rows through the `Store`. The typer command is a thin wrapper with a `make_embedder()` seam so tests inject the fake.

**Tech Stack:** Python 3.12, uv, lancedb, typer, pytest.

**Depends on:** plan 0a (`Embedder`, `EMBEDDING_DIM`, `Store`, `DocChunk`, `FakeEmbedder` in conftest). **Unblocks:** 0c's end-to-end smoke test.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- All persistent data lives in one LanceDB database at `settings.lancedb_dir` (default `data/lancedb`); settings come from `reachy_vec.config.settings` (pydantic-settings, `REACHY_VEC_` env prefix).
- Tests must not hit the network: use `FakeEmbedder` from `tests/conftest.py`; never instantiate `BgeEmbedder` in tests.
- Commit after every green test cycle; conventional-commit messages (`feat:`, `test:`, `chore:`).

---

### Task 1: Chunking and ingestion pipeline

**Files:**
- Create: `src/reachy_vec/store/ingestion.py`
- Test: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: `Store.add_doc_chunks`, `DocChunk`, `Embedder` (plan 0a).
- Produces: `chunk_text(text: str, max_chars: int = 1000) -> list[str]`; `ingest_path(path: Path, store: Store, embedder: Embedder) -> int` (returns number of chunks written).

- [ ] **Step 1: Write the failing test**

`tests/test_ingestion.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingestion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reachy_vec.store.ingestion'`.

- [ ] **Step 3: Write minimal implementation**

`src/reachy_vec/store/ingestion.py`:

```python
"""Ingestion pipeline: read .md/.txt files, chunk, embed, write to the store."""

from datetime import datetime, timezone
from pathlib import Path

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.schemas import DocChunk

SUPPORTED_SUFFIXES = {".md", ".txt"}


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars.

    Oversized single paragraphs are hard-split at max_chars boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        while len(para) > max_chars:
            pieces.append(para[:max_chars])
            para = para[max_chars:]
        if para:
            pieces.append(para)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def ingest_path(path: Path, store: Store, embedder: Embedder) -> int:
    """Ingest one file or every supported file under a directory tree."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*") if p.suffix in SUPPORTED_SUFFIXES and p.is_file()
        )

    now = datetime.now(timezone.utc).isoformat()
    written = 0
    for file in files:
        texts = chunk_text(file.read_text())
        if not texts:
            continue
        vectors = embedder.embed(texts)
        store.add_doc_chunks(
            [
                DocChunk(
                    chunk_id=f"{file}:{i}",
                    text=text,
                    vector=vector,
                    source=str(file),
                    ingested_at=now,
                )
                for i, (text, vector) in enumerate(zip(texts, vectors))
            ]
        )
        written += len(texts)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingestion.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/store/ingestion.py tests/test_ingestion.py
git commit -m "feat: chunking and ingestion pipeline for team docs"
```

---

### Task 2: Wire the `ingest` CLI command

**Files:**
- Modify: `src/reachy_vec/cli/ingest.py`
- Test: `tests/test_cli.py` (extend the existing file)

**Interfaces:**
- Consumes: `ingest_path` (Task 1), `Store`, `BgeEmbedder` (plan 0a), `settings` from `reachy_vec.config`.
- Produces: working `reachy-vec ingest PATH` command; `make_embedder() -> Embedder` seam in `reachy_vec.cli.ingest` (monkeypatched by tests, reused as the pattern for 0c's chat command).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
from reachy_vec.store.db import Store

from tests.conftest import FakeEmbedder


def test_ingest_command_writes_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha content")
    db_dir = tmp_path / "lancedb"

    monkeypatch.setattr("reachy_vec.cli.ingest.make_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("reachy_vec.cli.ingest.settings.data_dir", tmp_path)

    result = runner.invoke(app, ["ingest", str(docs)])
    assert result.exit_code == 0, result.output
    assert "1 chunk" in result.output
    assert Store(db_dir).doc_count() == 1


def test_ingest_command_rejects_missing_path(tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope")])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: new tests FAIL (`AttributeError`: `reachy_vec.cli.ingest` has no `make_embedder`); the original `--help` test still passes.

- [ ] **Step 3: Implement the ingest command**

Replace `src/reachy_vec/cli/ingest.py` with:

```python
from pathlib import Path

import typer

from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder, Embedder
from reachy_vec.store.ingestion import ingest_path


def make_embedder() -> Embedder:
    return BgeEmbedder(settings.embedding_model)


def ingest(path: Path) -> None:
    """Ingest team documents at PATH (.md/.txt file or directory) into the knowledge base."""
    if not path.exists():
        typer.echo(f"error: {path} does not exist", err=True)
        raise typer.Exit(code=1)
    store = Store(settings.lancedb_dir)
    count = ingest_path(path, store, make_embedder())
    suffix = "" if count == 1 else "s"
    typer.echo(f"Ingested {count} chunk{suffix} into {settings.lancedb_dir}.")
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 5: Guard the data directory**

`reachy-vec ingest` creates `data/lancedb` in the repo by default. Check it is ignored:

Run: `git check-ignore data/ || echo NOT-IGNORED`
If `NOT-IGNORED`: append a `data/` line to `.gitignore` and include it in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/reachy_vec/cli/ingest.py tests/test_cli.py .gitignore
git commit -m "feat: wire the ingest CLI command"
```

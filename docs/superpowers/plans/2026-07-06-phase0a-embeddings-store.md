# Phase 0a: Embeddings & Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The vector foundation — an `Embedder` protocol (real BGE + test fakes) and a LanceDB `Store` with a searchable `docs` table.

**Architecture:** Everything that embeds depends on the `Embedder` protocol, never on sentence-transformers directly, so tests substitute a deterministic fake. The `Store` wraps one embedded LanceDB database; Phase 0 implements only the `docs` table.

**Tech Stack:** Python 3.12, uv, lancedb (pydantic integration), sentence-transformers (BAAI/bge-small-en-v1.5, 384-dim), pytest.

**Depends on:** nothing. **Unblocks:** 0b (ingestion), 0c (RAG chat).

## Global Constraints

- Python `>=3.12`; run everything through `uv run`; add deps with `uv add`.
- Embedding model: `BAAI/bge-small-en-v1.5`, dimension **384** (constant `EMBEDDING_DIM = 384`).
- All persistent data lives in one LanceDB database at `settings.lancedb_dir` (default `data/lancedb`).
- Tests must not hit the network: use `FakeEmbedder` from `tests/conftest.py`; never instantiate `BgeEmbedder` in tests.
- Commit after every green test cycle; conventional-commit messages (`feat:`, `test:`, `chore:`).

---

### Task 1: Embedder protocol + test fakes

**Files:**
- Create: `src/reachy_vec/store/embeddings.py`
- Create: `tests/conftest.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `Embedder` protocol with `embed(texts: list[str]) -> list[list[float]]`; `BgeEmbedder(model_name: str)` implementing it; `EMBEDDING_DIM = 384` constant importable from `reachy_vec.store.embeddings`; `FakeEmbedder` and `FakeLLMClient` importable from `tests.conftest` (the LLM fake is consumed by plan 0c but lives here so conftest is written once).

- [x] **Step 1: Add the sentence-transformers dependency**

```bash
uv add "sentence-transformers>=3.0"
```

Expected: `uv.lock` updated, resolve succeeds.

- [x] **Step 2: Write the failing test**

`tests/conftest.py`:

```python
"""Shared test fakes: deterministic embedder and canned LLM client."""

import hashlib

from reachy_vec.store.embeddings import EMBEDDING_DIM


class FakeEmbedder:
    """Deterministic embeddings: same text -> same vector, no model download."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Repeat the 32-byte digest to fill EMBEDDING_DIM floats in [0, 1).
            raw = (digest * (EMBEDDING_DIM // len(digest) + 1))[:EMBEDDING_DIM]
            vectors.append([b / 256 for b in raw])
        return vectors


class FakeChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeChoiceMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(self._reply)


class FakeChat:
    def __init__(self, reply: str):
        self.completions = FakeCompletions(reply)


class FakeLLMClient:
    """Mimics the openai client surface used by brain.rag: client.chat.completions.create()."""

    def __init__(self, reply: str = "canned answer"):
        self.chat = FakeChat(reply)
```

`tests/test_embeddings.py`:

```python
from reachy_vec.store.embeddings import EMBEDDING_DIM, Embedder

from tests.conftest import FakeEmbedder


def test_embedding_dim_is_384():
    assert EMBEDDING_DIM == 384


def test_fake_embedder_is_deterministic_and_conforms():
    embedder: Embedder = FakeEmbedder()
    first = embedder.embed(["hello", "world"])
    second = embedder.embed(["hello", "world"])
    assert first == second
    assert len(first) == 2
    assert all(len(vec) == EMBEDDING_DIM for vec in first)
    assert first[0] != first[1]
```

- [x] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL with `ImportError: cannot import name 'EMBEDDING_DIM'` (module has no such names yet).

- [x] **Step 4: Write minimal implementation**

Replace `src/reachy_vec/store/embeddings.py` (create it) with:

```python
"""Text embeddings: protocol + sentence-transformers implementation.

BAAI/bge-small-en-v1.5 produces 384-dim normalized vectors. Anything that
embeds (ingestion, RAG search, later face/voice matching) depends on the
Embedder protocol, not on sentence-transformers directly, so tests can
substitute a deterministic fake.
"""

from typing import Protocol

EMBEDDING_DIM = 384


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class BgeEmbedder:
    """sentence-transformers embedder; loads the model lazily on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()
```

(`BgeEmbedder` is exercised for real in plan 0c's manual smoke test, not in unit tests — the model download is ~130 MB.)

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all PASS (including the existing CLI smoke test).

- [x] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/reachy_vec/store/embeddings.py tests/conftest.py tests/test_embeddings.py
git commit -m "feat: embedder protocol with BGE implementation and test fakes"
```

---

### Task 2: LanceDB store — schemas and docs table

**Files:**
- Modify: `src/reachy_vec/store/schemas.py` (currently a docstring stub)
- Modify: `src/reachy_vec/store/db.py` (currently a docstring stub)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `EMBEDDING_DIM` from Task 1.
- Produces: `DocChunk(LanceModel)` with fields `chunk_id: str, text: str, vector: Vector(384), source: str, ingested_at: str`; `Store(db_path: Path)` with methods `add_doc_chunks(chunks: list[DocChunk]) -> None`, `search_docs(query_vector: list[float], k: int = 5) -> list[DocChunk]`, `doc_count() -> int`.

- [x] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ImportError` (no `Store` / `DocChunk` defined yet).

- [x] **Step 3: Write the schemas**

Replace `src/reachy_vec/store/schemas.py` with:

```python
"""LanceDB table schemas.

Phase 0 uses only `docs`. The people/memories/messages tables arrive with
Phases 1-3 and will be defined here when implemented.
"""

from lancedb.pydantic import LanceModel, Vector

from reachy_vec.store.embeddings import EMBEDDING_DIM


class DocChunk(LanceModel):
    chunk_id: str
    text: str
    vector: Vector(EMBEDDING_DIM)
    source: str
    ingested_at: str  # ISO-8601 UTC timestamp
```

- [x] **Step 4: Write the store**

Replace `src/reachy_vec/store/db.py` with:

```python
"""LanceDB connection and vector-search helpers."""

from pathlib import Path

import lancedb

from reachy_vec.store.schemas import DocChunk

DOCS_TABLE = "docs"


class Store:
    """One embedded LanceDB database holding all reachy-vec tables."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(db_path)

    def _docs(self) -> lancedb.table.Table:
        if DOCS_TABLE not in self._db.table_names():
            self._db.create_table(DOCS_TABLE, schema=DocChunk)
        return self._db.open_table(DOCS_TABLE)

    def add_doc_chunks(self, chunks: list[DocChunk]) -> None:
        if chunks:
            self._docs().add(chunks)

    def search_docs(self, query_vector: list[float], k: int = 5) -> list[DocChunk]:
        return self._docs().search(query_vector).limit(k).to_pydantic(DocChunk)

    def doc_count(self) -> int:
        return self._docs().count_rows()
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 3 PASS. If `to_pydantic` is not available in the installed lancedb version, use `.to_list()` and construct `DocChunk(**{k: row[k] for k in DocChunk.model_fields})` per row instead — keep the return type `list[DocChunk]`.

- [x] **Step 6: Commit**

```bash
git add src/reachy_vec/store/schemas.py src/reachy_vec/store/db.py tests/test_store.py
git commit -m "feat: LanceDB store with docs table and vector search"
```

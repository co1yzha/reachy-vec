# Phase 0: RAG Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest team documents into LanceDB and answer questions about them in a terminal chat loop via OpenAI RAG — no robot hardware.

**Architecture:** An `Embedder` protocol (real: sentence-transformers BGE; tests: deterministic fake) feeds a LanceDB `Store` (embedded, at `data/lancedb`). Ingestion chunks `.md`/`.txt` files and writes embedded chunks to the `docs` table. The RAG layer embeds a question, vector-searches `docs`, and prompts the OpenAI chat API with the retrieved context. A REPL loop and two typer commands (`ingest`, `chat`) wire it together.

**Tech Stack:** Python 3.12, uv, lancedb (pydantic integration), sentence-transformers (BAAI/bge-small-en-v1.5, 384-dim), openai SDK, typer, pytest.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`; add deps with `uv add`.
- Embedding model: `BAAI/bge-small-en-v1.5`, dimension **384** (constant `EMBEDDING_DIM = 384`).
- LLM: OpenAI chat completions, model from `settings.llm_model` (default `"gpt-4o"`); API key from `OPENAI_API_KEY` env var, read implicitly by the openai SDK.
- All persistent data lives in one LanceDB database at `settings.lancedb_dir` (default `data/lancedb`).
- Tests must not hit the network: use `FakeEmbedder` and `FakeLLMClient` from `tests/conftest.py`; never instantiate `BgeEmbedder` or `openai.OpenAI` in tests.
- Existing settings come from `reachy_vec.config.settings` (pydantic-settings, `REACHY_VEC_` env prefix) — extend, don't replace.
- Commit after every green test cycle; messages in conventional-commit style (`feat:`, `test:`, `chore:`).

---

### Task 1: Embedder protocol + test fakes

**Files:**
- Create: `src/reachy_vec/store/embeddings.py`
- Create: `tests/conftest.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `Embedder` protocol with `embed(texts: list[str]) -> list[list[float]]`; `BgeEmbedder(model_name: str)` implementing it; `EMBEDDING_DIM = 384` constant importable from `reachy_vec.store.embeddings`; `FakeEmbedder` fixture-importable from `tests.conftest`.

- [ ] **Step 1: Add the sentence-transformers dependency**

```bash
uv add "sentence-transformers>=3.0"
```

Expected: `uv.lock` updated, resolve succeeds.

- [ ] **Step 2: Write the failing test**

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

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL with `ImportError: cannot import name 'EMBEDDING_DIM'` (module has no such names yet).

- [ ] **Step 4: Write minimal implementation**

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

(`BgeEmbedder` is exercised for real in Phase 0's final manual smoke test, not in unit tests — the model download is ~130 MB.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all PASS (including the existing CLI smoke test).

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ImportError` (no `Store` / `DocChunk` defined yet).

- [ ] **Step 3: Write the schemas**

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

- [ ] **Step 4: Write the store**

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

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 3 PASS. If `to_pydantic` is not available in the installed lancedb version, use `.to_list()` and construct `DocChunk(**{k: row[k] for k in DocChunk.model_fields})` per row instead — keep the return type `list[DocChunk]`.

- [ ] **Step 6: Commit**

```bash
git add src/reachy_vec/store/schemas.py src/reachy_vec/store/db.py tests/test_store.py
git commit -m "feat: LanceDB store with docs table and vector search"
```

---

### Task 3: Chunking and ingestion pipeline

**Files:**
- Create: `src/reachy_vec/store/ingestion.py`
- Test: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: `Store.add_doc_chunks`, `DocChunk` (Task 2); `Embedder` (Task 1).
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

### Task 4: RAG answer

**Files:**
- Modify: `src/reachy_vec/brain/rag.py` (currently a docstring stub)
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `Store.search_docs` (Task 2), `Embedder` (Task 1), `FakeLLMClient` (Task 1 conftest).
- Produces: `Answer` dataclass with `text: str, sources: list[str]`; `answer(question: str, store: Store, embedder: Embedder, client, model: str, k: int = 5) -> Answer`. `client` is any object with `chat.completions.create(model=..., messages=...)` returning an openai-shaped response.

- [ ] **Step 1: Write the failing test**

`tests/test_rag.py`:

```python
from reachy_vec.brain.rag import Answer, answer
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import FakeEmbedder, FakeLLMClient


def seeded_store(tmp_path) -> Store:
    store = Store(tmp_path / "lancedb")
    embedder = FakeEmbedder()
    texts = {"c1": "the pipeline runs nightly", "c2": "lunch is at noon"}
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id=cid,
                text=text,
                vector=embedder.embed([text])[0],
                source=f"{cid}.md",
                ingested_at="2026-07-06T00:00:00+00:00",
            )
            for cid, text in texts.items()
        ]
    )
    return store


def test_answer_returns_llm_text_and_sources(tmp_path):
    store = seeded_store(tmp_path)
    client = FakeLLMClient(reply="It runs nightly.")
    result = answer(
        "when does the pipeline run?",
        store=store,
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        k=1,
    )
    assert isinstance(result, Answer)
    assert result.text == "It runs nightly."
    assert result.sources == ["c1.md"]


def test_answer_puts_retrieved_context_and_question_in_prompt(tmp_path):
    store = seeded_store(tmp_path)
    client = FakeLLMClient()
    answer(
        "when does the pipeline run?",
        store=store,
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        k=1,
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    user_message = kwargs["messages"][-1]["content"]
    assert "the pipeline runs nightly" in user_message
    assert "when does the pipeline run?" in user_message


def test_answer_with_empty_store_says_no_context(tmp_path):
    store = Store(tmp_path / "lancedb")
    result = answer(
        "anything?",
        store=store,
        embedder=FakeEmbedder(),
        client=FakeLLMClient(),
        model="gpt-4o",
    )
    assert result.sources == []
    assert "knowledge base is empty" in result.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL with `ImportError: cannot import name 'Answer'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/reachy_vec/brain/rag.py` with:

```python
"""Retrieval-augmented generation: search docs, prompt the LLM."""

from dataclasses import dataclass

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

SYSTEM_PROMPT = (
    "You are Reachy, a friendly team assistant. Answer using ONLY the provided "
    "team-knowledge context. If the context does not contain the answer, say you "
    "don't know. Keep answers short and conversational - one or two sentences."
)

USER_TEMPLATE = """Context from the team knowledge base:

{context}

Question: {question}"""


@dataclass
class Answer:
    text: str
    sources: list[str]


def answer(
    question: str,
    *,
    store: Store,
    embedder: Embedder,
    client,
    model: str,
    k: int = 5,
) -> Answer:
    query_vector = embedder.embed([question])[0]
    hits = store.search_docs(query_vector, k=k)
    if not hits:
        return Answer(
            text="My knowledge base is empty - ingest some documents first.",
            sources=[],
        )

    context = "\n\n".join(f"[{hit.source}]\n{hit.text}" for hit in hits)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
    )
    return Answer(
        text=response.choices[0].message.content,
        sources=sorted({hit.source for hit in hits}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/rag.py tests/test_rag.py
git commit -m "feat: RAG answer over the docs table via OpenAI chat API"
```

---

### Task 5: Terminal chat loop

**Files:**
- Modify: `src/reachy_vec/brain/loop.py` (currently a docstring stub)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `answer` / `Answer` (Task 4), `Store` (Task 2), `Embedder` (Task 1).
- Produces: `chat_loop(store, embedder, client, model, input_fn=input, print_fn=print) -> None` — REPL that exits on `exit`, `quit`, or EOF, and prints each answer followed by its sources.

- [ ] **Step 1: Write the failing test**

`tests/test_loop.py`:

```python
from reachy_vec.brain.loop import chat_loop
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import FakeEmbedder, FakeLLMClient


def make_store(tmp_path) -> Store:
    store = Store(tmp_path / "lancedb")
    text = "the pipeline runs nightly"
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id="c1",
                text=text,
                vector=FakeEmbedder().embed([text])[0],
                source="notes.md",
                ingested_at="2026-07-06T00:00:00+00:00",
            )
        ]
    )
    return store


def run_loop(store, inputs: list[str]) -> list[str]:
    inputs_iter = iter(inputs)
    printed: list[str] = []
    chat_loop(
        store=store,
        embedder=FakeEmbedder(),
        client=FakeLLMClient(reply="Nightly."),
        model="gpt-4o",
        input_fn=lambda prompt="": next(inputs_iter),
        print_fn=printed.append,
    )
    return printed


def test_loop_answers_then_exits(tmp_path):
    printed = run_loop(make_store(tmp_path), ["when does it run?", "exit"])
    joined = "\n".join(printed)
    assert "Nightly." in joined
    assert "notes.md" in joined


def test_loop_exits_on_eof(tmp_path):
    inputs_iter = iter(["one question"])

    def input_fn(prompt=""):
        try:
            return next(inputs_iter)
        except StopIteration:
            raise EOFError

    printed: list[str] = []
    chat_loop(
        store=make_store(tmp_path),
        embedder=FakeEmbedder(),
        client=FakeLLMClient(),
        model="gpt-4o",
        input_fn=input_fn,
        print_fn=printed.append,
    )  # must return instead of raising
    assert printed  # answered the one question before EOF


def test_loop_skips_blank_lines(tmp_path):
    printed = run_loop(make_store(tmp_path), ["", "  ", "quit"])
    assert all("canned answer" not in line for line in printed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'chat_loop'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/reachy_vec/brain/loop.py` with:

```python
"""Terminal conversation loop (Phase 0). Phase 1 swaps input/output for robot audio."""

from reachy_vec.brain.rag import answer
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

EXIT_COMMANDS = {"exit", "quit"}


def chat_loop(
    *,
    store: Store,
    embedder: Embedder,
    client,
    model: str,
    input_fn=input,
    print_fn=print,
) -> None:
    print_fn("Reachy KB chat - ask about your team docs ('exit' to leave).")
    while True:
        try:
            question = input_fn("you> ").strip()
        except EOFError:
            return
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return
        result = answer(question, store=store, embedder=embedder, client=client, model=model)
        print_fn(f"reachy> {result.text}")
        if result.sources:
            print_fn(f"        (sources: {', '.join(result.sources)})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loop.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/loop.py tests/test_loop.py
git commit -m "feat: terminal chat loop over the RAG layer"
```

---

### Task 6: Wire the CLI commands and smoke-test end to end

**Files:**
- Modify: `src/reachy_vec/cli/ingest.py`
- Modify: `src/reachy_vec/cli/chat.py`
- Test: `tests/test_cli.py` (extend the existing file)

**Interfaces:**
- Consumes: `ingest_path` (Task 3), `chat_loop` (Task 5), `Store` (Task 2), `BgeEmbedder` (Task 1), `settings` from `reachy_vec.config`.
- Produces: working `reachy-vec ingest PATH` and `reachy-vec chat` commands.

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

- [ ] **Step 4: Implement the chat command**

Replace `src/reachy_vec/cli/chat.py` with:

```python
import typer

from reachy_vec.config import settings
from reachy_vec.brain.loop import chat_loop
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder


def chat() -> None:
    """Chat with the team knowledge base in the terminal (no robot needed)."""
    from openai import OpenAI

    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)
    chat_loop(
        store=store,
        embedder=BgeEmbedder(settings.embedding_model),
        client=OpenAI(),
        model=settings.llm_model,
    )
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 6: Manual end-to-end smoke test (real models, needs OPENAI_API_KEY)**

```bash
mkdir -p /tmp/team-docs
printf 'The nightly data pipeline kicks off at 02:00 UTC and takes about 40 minutes.\n' > /tmp/team-docs/pipeline.md
uv run reachy-vec ingest /tmp/team-docs
uv run reachy-vec chat
# you> when does the pipeline run?
# expect an answer mentioning 02:00 UTC with (sources: /tmp/team-docs/pipeline.md), then: exit
```

Expected: first run downloads the BGE model (~130 MB), then answers correctly. Note: `data/lancedb` (created in the repo) is the default store; verify `data/` is gitignored — if not, add `data/` to `.gitignore` in this commit.

- [ ] **Step 7: Commit**

```bash
git add src/reachy_vec/cli/ingest.py src/reachy_vec/cli/chat.py tests/test_cli.py .gitignore
git commit -m "feat: wire ingest and chat CLI commands - Phase 0 milestone"
```

---

## Self-Review Notes

- **Spec coverage:** Phase 0 = scaffold (done previously), store module, config (exists), `ingest` CLI, text-only RAG loop, milestone "correct RAG answers over a sample doc set in the terminal" → Tasks 2, 3, 6, 4+5, and Task 6 Step 6 respectively. People/memories/messages tables are deliberately deferred to Phases 1-3 (schemas note this).
- **Type consistency:** `Embedder.embed(list[str]) -> list[list[float]]` used identically in Tasks 1-4; `DocChunk` field set identical across Tasks 2-5; `answer(question, *, store, embedder, client, model, k)` matches its call in `chat_loop`.
- **No placeholders:** every code step contains complete code; the only conditional instruction (lancedb `to_pydantic` fallback in Task 2 Step 5) includes its concrete alternative.

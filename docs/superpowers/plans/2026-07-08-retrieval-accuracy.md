# Retrieval Accuracy (BGE query prefix + hybrid search) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix docs-search accuracy by embedding queries with the BGE instruction prefix and combining vector search with BM25 (LanceDB hybrid search) over the `docs` table.

**Architecture:** Query-side-only changes: the `Embedder` protocol gains `embed_query()` (prefix applied at query time; stored vectors untouched), and `Store.search_docs_scored()` gains a `query_text` parameter that switches to `query_type="hybrid"` when the docs table has an FTS index. Scores reported to the LLM stay cosine 0..1 (recomputed per returned chunk). `add_doc_chunks()` refreshes a native Lance FTS index on `text` after every write, so `ingest`/`sync-mongo` pick everything up with zero changes.

**Tech Stack:** Python 3.12, LanceDB 0.34 (embedded; native FTS via `lancedb.index.FTS`, hybrid search with default RRF reranker), sentence-transformers (BGE), pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-retrieval-accuracy-design.md`

## Global Constraints

- All tests run offline with no model downloads (fakes for embedding/LLM; the *real* embedded LanceDB is fine in tests, including FTS/hybrid).
- Heavy imports stay deferred inside methods (sentence-transformers must NOT be imported at module import time). `lancedb.index` is light and may be imported at the top of `db.py`.
- New config knobs go in `config.py` (`REACHY_VEC_` env prefix), not module constants.
- Query prefix default (copy verbatim): `"Represent this sentence for searching relevant passages: "` — empty string disables it.
- Verified API for lancedb 0.34: `table.create_index("text", config=FTS(), replace=True)` creates index named `text_idx`; `table.search(query_type="hybrid").vector(qv).text(q).limit(k).to_list()` returns rows with all schema fields plus `_relevance_score`. (`create_fts_index` is deprecated — do not use it.)
- Run `uv run ruff check src tests` before every commit; both ruff and pytest must pass.

---

### Task 1: `embed_query` on the Embedder protocol + config knob

**Files:**
- Modify: `src/reachy_vec/store/embeddings.py`
- Modify: `src/reachy_vec/config.py` (add one setting after `embedding_model`, line 24)
- Modify: `tests/conftest.py` (FakeEmbedder, line 9)
- Modify: `src/reachy_vec/cli/chat.py:23`, `src/reachy_vec/cli/run.py:60`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: existing `Embedder.embed(texts: list[str]) -> list[list[float]]`.
- Produces: `Embedder.embed_query(text: str) -> list[float]` (protocol method, implemented by `BgeEmbedder` and `FakeEmbedder`); `BgeEmbedder(model_name: str, query_prefix: str = BGE_QUERY_PREFIX)`; `settings.embedding_query_prefix: str`. Later tasks rely on `FakeEmbedder().embed_query(text) == FakeEmbedder().embed([text])[0]` (no prefix in the fake, so tests can seed stores with `embed()` and query with `embed_query()`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
def test_fake_embedder_embed_query_matches_embed():
    embedder = FakeEmbedder()
    assert embedder.embed_query("hello") == embedder.embed(["hello"])[0]


def test_bge_embed_query_prepends_prefix(monkeypatch):
    from reachy_vec.store.embeddings import BgeEmbedder

    embedder = BgeEmbedder("any-model", query_prefix="QP: ")
    captured = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return [[0.0] * EMBEDDING_DIM]

    monkeypatch.setattr(embedder, "embed", fake_embed)
    vector = embedder.embed_query("hello")
    assert captured["texts"] == ["QP: hello"]
    assert len(vector) == EMBEDDING_DIM


def test_bge_default_prefix_is_the_bge_instruction():
    from reachy_vec.store.embeddings import BGE_QUERY_PREFIX, BgeEmbedder

    assert BGE_QUERY_PREFIX == (
        "Represent this sentence for searching relevant passages: "
    )
    assert BgeEmbedder("any-model")._query_prefix == BGE_QUERY_PREFIX


def test_settings_default_query_prefix():
    from reachy_vec.config import Settings

    assert Settings().embedding_query_prefix == (
        "Represent this sentence for searching relevant passages: "
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings.py -q`
Expected: FAIL — `AttributeError: 'FakeEmbedder' object has no attribute 'embed_query'`, `TypeError: BgeEmbedder.__init__() got an unexpected keyword argument 'query_prefix'`, `ImportError: cannot import name 'BGE_QUERY_PREFIX'`, `AttributeError: 'Settings' object has no attribute 'embedding_query_prefix'`.

- [ ] **Step 3: Implement**

In `src/reachy_vec/store/embeddings.py`, replace the protocol and `BgeEmbedder` with:

```python
EMBEDDING_DIM = 384

# BGE models expect this instruction on *queries* (not documents); see the
# bge-small-en-v1.5 model card. Applied at query time only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    """sentence-transformers embedder; loads the model lazily on first use."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self._query_prefix + text])[0]
```

In `src/reachy_vec/config.py`, directly under `embedding_model` (line 24) add:

```python
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )  # BGE query instruction; set empty to disable for non-BGE models
```

In `tests/conftest.py`, add to `FakeEmbedder` (after `embed`):

```python
    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

In `src/reachy_vec/cli/chat.py:23` change the embedder argument to:

```python
        embedder=BgeEmbedder(
            settings.embedding_model, query_prefix=settings.embedding_query_prefix
        ),
```

In `src/reachy_vec/cli/run.py:60` change to:

```python
    embedder = BgeEmbedder(
        settings.embedding_model, query_prefix=settings.embedding_query_prefix
    )
```

(`cli/ingest.py` and `cli/sync.py` only embed documents — leave them as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings.py -q && uv run pytest -q`
Expected: all PASS (full suite too — nothing calls `embed_query` yet).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/store/embeddings.py src/reachy_vec/config.py tests/conftest.py tests/test_embeddings.py src/reachy_vec/cli/chat.py src/reachy_vec/cli/run.py
git commit -m "feat: embed_query with BGE instruction prefix (query-side only)"
```

---

### Task 2: FTS index + hybrid `search_docs_scored` in Store

**Files:**
- Modify: `src/reachy_vec/store/db.py` (imports; `add_doc_chunks` line 41; `search_docs_scored` line 51)
- Test: `tests/test_store_scored.py`

**Interfaces:**
- Consumes: `DocChunk` schema; existing `Store._docs()` / `add_doc_chunks` / `search_docs_scored`.
- Produces: `Store.search_docs_scored(query_vector: list[float], k: int = 5, query_text: str | None = None) -> list[tuple[DocChunk, float]]` — float is cosine similarity 0..1 in BOTH paths. `add_doc_chunks` now (re)creates the FTS index `text_idx` after every write.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store_scored.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store_scored.py -q`
Expected: the two new hybrid tests and the index test FAIL (`list_indices()` empty; `search_docs_scored() got an unexpected keyword argument 'query_text'`). Fallback/no-query-text tests may already pass.

- [ ] **Step 3: Implement**

In `src/reachy_vec/store/db.py`, add to the imports at the top:

```python
from lancedb.index import FTS
```

Add a module-level helper below the table-name constants:

```python
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

Replace `add_doc_chunks` (db.py:41):

```python
    def add_doc_chunks(self, chunks: list[DocChunk]) -> None:
        if chunks:
            table = self._docs()
            table.add(chunks)
            # BM25 leg of hybrid search; cheap to rebuild at this corpus size.
            table.create_index("text", config=FTS(), replace=True)
```

Replace `search_docs_scored` (db.py:51):

```python
    def search_docs_scored(
        self, query_vector: list[float], k: int = 5, query_text: str | None = None
    ) -> list[tuple[DocChunk, float]]:
        """Top-k docs with cosine similarity scores (1.0 = identical).

        With query_text and an FTS index present, ranking is hybrid
        (vector + BM25, RRF); the reported score is still cosine so its
        meaning never changes. Falls back to vector-only otherwise.
        """
        if self.doc_count() == 0:
            return []
        table = self._docs()
        if query_text and any(i.name == "text_idx" for i in table.list_indices()):
            rows = (
                table.search(query_type="hybrid")
                .vector(query_vector)
                .text(query_text)
                .limit(k)
                .to_list()
            )
            return [
                (
                    DocChunk(**{k_: row[k_] for k_ in DocChunk.model_fields}),
                    _cosine(query_vector, list(row["vector"])),
                )
                for row in rows
            ]
        rows = table.search(query_vector).metric("cosine").limit(k).to_list()
        return [
            (
                DocChunk(**{k_: row[k_] for k_ in DocChunk.model_fields}),
                1.0 - row["_distance"],
            )
            for row in rows
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store_scored.py tests/test_store.py tests/test_ingestion.py tests/test_mongo_sync.py -q && uv run pytest -q`
Expected: all PASS (ingestion/sync go through `add_doc_chunks`, so they exercise index creation too).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/store/db.py tests/test_store_scored.py
git commit -m "feat: hybrid (BM25 + vector) docs search with cosine scores"
```

---

### Task 3: ChatBrain uses embed_query + passes question text

**Files:**
- Modify: `src/reachy_vec/brain/chat.py` (`respond` line 235; `_retrieve_docs` line 262 and its call site line 240)
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Consumes: `Embedder.embed_query(text) -> list[float]` (Task 1); `Store.search_docs_scored(vector, k, query_text)` (Task 2).
- Produces: no new public surface — `respond()` behavior only.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_brain.py` (module already imports `ChatBrain`, `TurnIdentity`, `Store`, `DocChunk`, `FakeEmbedder`, `FakeLLMClient`, and defines `YANG` and `seeded_store`):

```python
class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.query_texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return super().embed_query(text)


def test_question_is_embedded_as_query(tmp_path):
    embedder = RecordingEmbedder()
    brain = ChatBrain(
        store=seeded_store(tmp_path),
        embedder=embedder,
        client=FakeLLMClient(reply="ok"),
        model="gpt-4o",
        opener=lambda url: None,
    )
    brain.respond("any demos about food?", identity=YANG)
    assert embedder.query_texts == ["any demos about food?"]


def test_keyword_only_match_ranks_first_in_context(tmp_path):
    # The question's fake vector is unrelated to both chunks; only BM25 on
    # "Jane Smith" can put the Robot Arm chunk first in the injected context.
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder()
    texts = {
        "arm": "Demo: Robot Arm Teleop\nAuthors: Jane Smith",
        "food": "Demo: Food Mapping\nAuthors: Bob",
    }
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id=key,
                text=text,
                vector=embedder.embed([text])[0],
                source=f"demo: {key}",
                ingested_at="2026-07-08T00:00:00+00:00",
            )
            for key, text in texts.items()
        ]
    )
    client = FakeLLMClient(reply="ok")
    brain = ChatBrain(
        store=store,
        embedder=embedder,
        client=client,
        model="gpt-4o",
        opener=lambda url: None,
    )
    brain.respond("who is Jane Smith?", identity=YANG)
    context = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert context.index("Robot Arm Teleop") < context.index("Food Mapping")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_brain.py -q`
Expected: `test_question_is_embedded_as_query` FAILS (`query_texts == []` — brain still calls `embed`). `test_keyword_only_match_ranks_first_in_context` FAILS (vector-only ranking puts an arbitrary chunk first; if it passes by luck, it locks in the behavior — keep it).

- [ ] **Step 3: Implement**

In `src/reachy_vec/brain/chat.py`:

Line 235, replace:

```python
        vector = self._embedder.embed([question])[0]
```

with:

```python
        vector = self._embedder.embed_query(question)
```

Line 240, replace the `context=` argument:

```python
                    context=self._retrieve_docs(vector, question)
                    or "(nothing relevant found)",
```

Line 262, replace `_retrieve_docs`:

```python
    def _retrieve_docs(self, vector: list[float], question: str) -> str:
        scored = self._store.search_docs_scored(vector, k=self._k, query_text=question)
        return "\n\n".join(
            f"[{chunk.source} | score {score:.2f}]\n{chunk.text}"
            for chunk, score in scored
        )
```

Leave `_retrieve_memories`, `end_conversation`'s note embedding, and the duplicate check untouched — they are document-side (`embed()`) or already receive the query vector.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_brain.py tests/test_loop.py tests/test_memories_store.py -q && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: brain retrieves docs via hybrid search with prefixed query embedding"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/architecture.md` (THINKING line, ~line 65)
- Modify: `docs/pipelines.md` (embeddings row line 12; RAG section near line 34)
- Modify: `docs/configuration.md` (models table, after `EMBEDDING_MODEL` line 20)

**Interfaces:** none — prose only.

- [ ] **Step 1: Update the docs**

`docs/architecture.md` line 65: change

```
THINKING (ChatBrain): embed question → scored LanceDB search → context +
```

to

```
THINKING (ChatBrain): embed question (BGE query prefix) → hybrid LanceDB
search (vector + BM25, cosine scores) → context +
```

`docs/pipelines.md` line 12: change the embeddings row's second column from `` `BAAI/bge-small-en-v1.5`, 384-dim normalized `` to `` `BAAI/bge-small-en-v1.5`, 384-dim normalized; queries get the BGE instruction prefix (`REACHY_VEC_EMBEDDING_QUERY_PREFIX`) ``. After the paragraph ending "everything lives in one local BGE space." (line 34) add:

```markdown
Docs retrieval is hybrid: every `add_doc_chunks` write refreshes a native
Lance FTS index on `text`, and per-turn search combines BM25 with the
query vector (RRF). Scores shown to the LLM stay cosine 0..1. A DB
without the index (created before this feature) transparently falls back
to vector-only until the next ingest/sync rebuilds it.
```

`docs/configuration.md`: after the `EMBEDDING_MODEL` row (line 20) add:

```markdown
| `EMBEDDING_QUERY_PREFIX` | `Represent this sentence for searching relevant passages: ` | BGE query instruction, applied to search queries only (never documents); set empty when using a non-BGE embedding model |
```

(Match the table's existing column layout; the env var is `REACHY_VEC_EMBEDDING_QUERY_PREFIX` — the table omits the prefix like the other rows if that's the established style; check row 20 and copy its convention.)

- [ ] **Step 2: Full verification**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all PASS, no lint errors.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md docs/pipelines.md docs/configuration.md
git commit -m "docs: hybrid retrieval + BGE query prefix"
```

---

## Manual verification (after all tasks)

1. `uv run reachy-vec sync-mongo` — rebuilds demo chunks AND creates the FTS index on the real DB.
2. Replay previously-failing questions in `uv run reachy-vec chat`.
3. Score debug for any stragglers:

```bash
uv run python -c "
from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder
q = 'who did the robot arm demo?'
e = BgeEmbedder(settings.embedding_model, query_prefix=settings.embedding_query_prefix)
for chunk, score in Store(settings.lancedb_dir).search_docs_scored(
    e.embed_query(q), k=10, query_text=q
):
    print(f'{score:.3f}  [{chunk.source}]  {chunk.text[:70]!r}')
"
```

# Retrieval accuracy — BGE query prefix + hybrid (BM25 + vector) docs search

**Date:** 2026-07-08 · **Status:** approved in conversation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (improves
the Phase 0 RAG pipeline; no new capability, better answers from the
existing one).

## Problem

Search over Mongo-synced demos is inaccurate. Two root causes, both on the
query side — the stored data is fine:

1. **Missing BGE query instruction.** `bge-small-en-v1.5` expects queries
   (not documents) to be prefixed with
   `"Represent this sentence for searching relevant passages: "`. We embed
   the user's question verbatim, which measurably hurts short queries —
   exactly what spoken questions are.
2. **Metadata-heavy corpus, dense-only search.** Demo chunks are mostly
   titles, author names, tags, and URLs. A 384-dim dense model is weak at
   matching a natural question against sparse keyword-ish text; exact
   tokens (names, project names, tags) are what BM25 is good at.

## Decisions

1. **Query prefix via `Embedder.embed_query()`** — prefix applied at query
   time only; stored vectors are untouched (no re-ingestion needed).
   Prefix is a config knob (`REACHY_VEC_EMBEDDING_QUERY_PREFIX`), default
   is the BGE instruction; empty string disables it for non-BGE models.
2. **Hybrid docs search in LanceDB** — `query_type="hybrid"` combining the
   query vector with BM25 over a native Lance FTS index on `docs.text`
   (default RRF reranker, no new dependency, fully local). Verified
   against lancedb 0.34 (installed).
3. **Scores stay cosine 0..1.** Hybrid decides *which* chunks and their
   order; the score shown to the LLM is cosine similarity recomputed
   between the query vector and each returned chunk's stored vector
   (both normalized → dot product). The `CONTEXT_TEMPLATE` prompt keeps
   its meaning unchanged.
4. **Graceful fallback.** No FTS index (old DB) or no `query_text` given →
   current vector-only path. Nothing breaks until the next ingest/sync
   creates the index.
5. **Memories stay vector-only** — tiny per-person sets don't need BM25;
   they do benefit from the prefixed query vector for free.
6. Rejected: a `Retriever` layer between `ChatBrain` and `Store` (adds an
   abstraction with no behavioral difference); a `hybrid_search` on/off
   flag (fallback already covers the failure mode); switching to direct
   MongoDB/Atlas retrieval (network in the conversation loop, only covers
   demos, doesn't address either root cause).

## Changes by module

### `store/embeddings.py` + `config.py`

- `Embedder` protocol gains `embed_query(text: str) -> list[float]`.
- `BgeEmbedder(model_name, query_prefix)` implements it as
  `self.embed([query_prefix + text])[0]`.
- New setting `embedding_query_prefix: str` (default the BGE instruction);
  passed to `BgeEmbedder` where it is constructed (`cli/run.py`,
  `cli/chat.py`).

### `store/db.py`

- `search_docs_scored(query_vector, k=5, query_text: str | None = None)`:
  - `query_text` given **and** docs table has an FTS index → hybrid:
    `search(query_type="hybrid").vector(qv).text(query_text).limit(k)`,
    then recompute cosine per row from the stored vector.
  - otherwise → existing vector-only cosine path.
- `add_doc_chunks()` refreshes the index after writing:
  `create_fts_index("text", replace=True)`. Cheap at this corpus size;
  `ingest` and `sync-mongo` need zero changes. Existing DBs get the index
  on their next ingest/sync.

### `brain/chat.py`

- Per-turn question: `embed_query()` instead of `embed()`, and the raw
  question text is passed through `_retrieve_docs` into
  `search_docs_scored`.
- Memories search reuses the same (prefixed) query vector.
- Unchanged: distilled-note embedding at save time and the note-vs-note
  duplicate check — both are document-to-document, plain `embed()`.

## Testing (fakes-first, all offline)

- Fake embedder in `conftest.py` gains `embed_query` (delegates to its
  `embed`).
- `BgeEmbedder` prefix logic asserted without loading the model.
- Store tests run the **real** embedded LanceDB hybrid path:
  - keyword-only match: a chunk whose fake vector is unrelated to the
    query must still surface via BM25 (e.g. an author name);
  - returned scores are cosine 0..1;
  - vector-only fallback works on a table with no FTS index and when
    `query_text` is omitted.
- Chat-brain test: the question goes through `embed_query` and the raw
  text reaches the store.
- Docs: a paragraph each in `architecture.md` and `pipelines.md`.

## Manual verification

Re-run `sync-mongo` (builds the FTS index), then replay previously-failing
questions via `reachy-vec chat` and the score-debug one-liner
(`search_docs_scored` top-10 with scores) to confirm the right demos rank
in the top k.

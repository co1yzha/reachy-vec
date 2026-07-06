# Phase 0: RAG Foundation — Plan Index

Phase 0 is split into three independently executable sub-plans. Each produces
working, tested software on its own and carries its own task checklist.

| Sub-plan | Delivers | Depends on |
|---|---|---|
| [0a — Embeddings & Store](2026-07-06-phase0a-embeddings-store.md) | `Embedder` protocol + BGE implementation, test fakes, LanceDB `Store` with searchable `docs` table | — |
| [0b — Ingestion + CLI](2026-07-06-phase0b-ingestion.md) | Chunker, `ingest_path` pipeline, working `reachy-vec ingest` command | 0a |
| [0c — RAG Chat + CLI](2026-07-06-phase0c-rag-chat.md) | `answer()` RAG layer, terminal REPL, working `reachy-vec chat` command, end-to-end smoke test | 0a (0b for the final smoke test only) |

Execute in order 0a → 0b → 0c. The Phase 0 milestone (from the
[design spec](../specs/2026-07-06-team-familiar-design.md)) is 0c's final
smoke test: correct spoken-style RAG answers over a sample doc set, in the
terminal.

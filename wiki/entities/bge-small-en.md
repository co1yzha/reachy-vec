---
title: BGE-small-en-v1.5 (embeddings)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [rag, lancedb]
sources: [raw/docs/pipelines.md]
---

`BAAI/bge-small-en-v1.5` (384-dim, normalized; local via sentence-transformers)
embeds **everything textual** in one shared space in the [[lancedb-store]]: the `docs` chunks, the
`memories` notes, and the per-turn query.

- Queries get the BGE instruction prefix (`REACHY_VEC_EMBEDDING_QUERY_PREFIX`).
- Configured via `REACHY_VEC_EMBEDDING_MODEL`.
- Mongo's own 1536-dim `embedding` field is **ignored** — `sync-mongo` re-embeds
  in this local space so retrieval stays in one metric space (the space the [[rag-pipeline]] searches every turn).


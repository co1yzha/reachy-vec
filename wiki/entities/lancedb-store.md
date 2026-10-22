---
title: LanceDB store
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [lancedb, rag]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

A single **embedded LanceDB** database at `data/lancedb` holds all vectors.
All access goes through `store/db.py:Store`; schemas in `store/schemas.py`.
`data/` is git-ignored (privacy: transcripts, embeddings of enrolled people).

| Table | Contents | Vector |
|---|---|---|
| `docs` | knowledge chunks (demo: / ingested) | 384-dim [[bge-small-en]] |
| `people` | face embeddings, **one row per captured frame** | 512-dim [[insightface]] |
| `voices` | speaker embeddings (enrolled + passive) | 192-dim [[ecapa-tdnn]] |
| `greetings` | per-person last-greeted timestamps | — |
| `memories` | per-person notes (saved + distilled) | 384-dim BGE |
| `messages` | queued relays (messenger) | — |

Retrieval is **hybrid**: every write refreshes a native Lance FTS index on
`text`, and per-turn search combines BM25 with the query vector (RRF); scores
shown to the LLM stay cosine 0..1. ^[raw/docs/pipelines.md]


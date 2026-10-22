---
title: Memory pipeline (per-person notes)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [memory, rag]
sources: [raw/docs/pipelines.md]
---

Phase 2. Memories are per-person notes in the `memories` table.

- **Explicit:** the model calls `save_note` when asked to remember something.
- **Implicit:** when a conversation ends (`end_conversation`), one extra
  [[gpt-5-mini]] call **per enrolled person who spoke** reviews the visit and
  distills up to 3 third-person notes each (or `NONE` for chit-chat).
- Both paths embed the note with [[bge-small-en]], skip near-duplicates
  (cosine ≥ 0.97 vs the person's nearest existing memory), and write rows.
- **Recall is automatic:** step 2 of every [[rag-pipeline]] turn searches this
  person's memories with the question vector.


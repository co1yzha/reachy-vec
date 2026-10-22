---
title: Roadmap (Phases 0–4)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [roadmap, robot]
sources: [raw/docs/readme.md, raw/docs/architecture.md]
---

The project is built in phases (design specs in `docs/superpowers/specs/`):

- **Phase 0 — Skeleton & simulator:** text-only RAG loop, doc ingestion, no hardware.
- **Phase 1 — Oracle:** walk up, get greeted by name, ask, hear the answer.
- **Phase 2 — Memory keeper:** per-person attribution of notes, recall by person.
- **Phase 3 — Messenger:** "tell Bob…" relayed when Bob is next seen.
- **Phase 4 — On-robot deployment** *(in progress)*: **4a** on-robot media,
  **4b** `ROBOT_HOST` + reconnect, **4c** [[barge-in]] have shipped; **4d**
  autostart + offline degradation remains. See [[media-source]].

**Known gaps** (from `docs/architecture.md`): no offline LLM fallback (answers
need the OpenAI API); robot media does not hot-recover from a stream drop
(soft-degrades until it returns); attention is face-only (no wake word / sound
localization); far-field mic gain untuned; no autostart/service; identity
management (delete/rename/merge, consent + retention) not exposed beyond the raw
store.


---
title: Mac brain (Apple silicon)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [robot, protocol, llm]
sources: [raw/docs/architecture.md, raw/docs/readme.md]
---

The **brain runs on a Mac (Apple silicon)**; the robot is a thin body. All
heavy local models (insightface, ECAPA, faster-whisper, BGE, Qwen3-TTS)
lazy-load on the Mac and are warmed at `reachy-vec run` startup so the first
conversation isn't slow.

- Runs the full perception + speech loop, the [[lancedb-store]], and the
  OpenAI-based [[rag-pipeline]].
- `MEDIA_SOURCE` selects whether I/O comes from the Mac's own devices or the
  robot's — see [[media-source]].
- Cloud answers need the OpenAI API; there is deliberately **no offline
  fallback** (a known gap). ^[raw/docs/architecture.md]


---
title: gpt-5-mini (OpenAI)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [llm, rag, tools]
sources: [raw/docs/pipelines.md]
---

`gpt-5-mini` (OpenAI API, cloud) is the reasoning engine. It does the
[[rag-pipeline]] turn, executes [[tools]], and distills per-person memories at
the end of a conversation.

- Configured via `REACHY_VEC_LLM_MODEL` (default `gpt-5-mini`).
- Replies **stream** — each sentence is spoken as it generates (~1 s to first
  sound). Tool calls cost a second LLM round-trip.
- **Requires the API key** (`OPENAI_API_KEY`); a WiFi/API outage yields a spoken
  apology, not a degraded local answer (no offline fallback — deliberate).


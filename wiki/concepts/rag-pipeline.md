---
title: RAG pipeline (one turn)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [rag, lancedb, tools]
sources: [raw/docs/pipelines.md]
---

`brain/chat.py:ChatBrain.respond` — one turn:

0. The fused per-turn [[identity-rule]] arrives as `identity=`. Retrieval,
   `save_note`, and `send_message` all follow **the turn's speaker**. Anonymous
   turns get docs-only retrieval and side-effect tools politely refuse.
1. Embed the question with [[bge-small-en]] (same space as the docs).
2. Retrieve locally every turn (no retrieval tool-calls): top-5 `docs` chunks
   (cosine) + top-3 of **the speaker's** `memories`.
3. Build one user message: `[context + scores] + [memories] + "Name: question"`,
   appended to the visit's history (max 20 messages, reset per visit).
4. One **streaming** [[gpt-5-mini]] call with the Reachy personality prompt +
   the [[tools]] list. Completed sentences are spoken as they arrive. The model
   judges relevance: a grounded answer naming the demo, or general knowledge
   with a casual "off the top of my head" signal (none for chit-chat).
5. If the model called tools: execute them, append results, make a **second**
   LLM call for the final spoken reply (sentence audio suppressed while tool
   calls are pending).


# Conversational Brain + Demo-Opening Tool — Design

**Date:** 2026-07-07 · **Status:** approved in conversation
**Supersedes:** the score-gated fallback (decision 2) in
[Phase 1.5](2026-07-06-phase1.5-mongo-fallback-stt-design.md) and the
single-shot `rag.answer` design.

## Goal

Turn the Oracle from a stateless Q&A pipeline into a personable chatbot with
conversation memory, where retrieval is always-on context (no extra LLM
round-trip) and real actions are LLM tools — starting with opening a demo's
URL in the browser on the lab PC.

## Design

**`brain/chat.py` — `ChatBrain`** (replaces `brain/rag.py`):

- **Personality system prompt:** Reachy, the team's desk robot; warm, a
  little playful, professional; answers are spoken aloud so 1–2 short
  sentences; names demos when citing them.
- **Always-on retrieval (pushed, not pulled):** every turn, embed the user's
  words, run the scored LanceDB search (~50 ms, local), and inject the top-k
  chunks with scores into the turn's user message. The *prompt* instructs:
  use context when relevant; otherwise answer generally but start with
  "Not from our team docs, but". The code-level `rag_min_score` gate is
  removed (setting deleted).
- **Conversation history:** `ChatBrain` keeps the message list (trimmed to
  the last ~20 messages); `reset()` clears it. The Oracle resets when a
  conversation starts; the speaker's name prefixes their turns so the model
  knows who's talking. Follow-ups ("tell me more about that one") now work.
- **Tool loop:** OpenAI tool-calling with one tool for now —
  `open_url(url, title)` → opens the URL in the default browser on the Mac
  (`open <url>`), http/https only. The model reads demo URLs from the
  retrieved context. Tool turns cost a second LLM call — acceptable, they're
  actions, not answers. Phase 2 adds `save_note` / `send_message` here.

**Integration:** `OracleLoop` takes a `brain` (respond/reset) instead of
`answer_fn`; terminal `chat` uses the same brain (history + tools work in
text mode too, which is also how open_url is smoke-tested).

## Error handling

- Tool asked to open a non-http(s) URL → refused, tool result says so.
- Opener/LLM failure → spoken apology (existing Oracle behavior), history
  intact.

## Out of scope

save_note / send_message tools (Phase 2), multi-demo disambiguation UI,
fish-speech TTS.

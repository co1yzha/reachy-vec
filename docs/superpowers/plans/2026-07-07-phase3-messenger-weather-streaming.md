# Phase 3 Package: Messenger + Weather Tool + Streaming Speech — Plan

> Executed inline. Approved in conversation 2026-07-07.

## 0. Housekeeping (before features)

- Merge `phase2-companion` → `main`; continue on `phase3-messenger`.
- Fix enrollment name: people row `person-983afcb0` "Yum" → "Yang" (LanceDB update).
- Memory dedupe: `ChatBrain._store_memories` skips a note whose cosine
  similarity to an existing memory of the same person is ≥ 0.97.

## 1. Messenger (Phase 3 of the parent spec)

- `MessageRow`: `message_id, from_person, from_name, to_person, to_name,
  text, created_at, delivered_at` (empty string = pending).
- `Store`: `add_message(row)`, `pending_messages_for(person_id)`,
  `mark_delivered(message_id)`, `find_person_by_name(name)` (case-insensitive
  match over enrolled people; None if unknown).
- `send_message` tool: `{to_name, message}` — resolves recipient among
  *enrolled people only*; unknown → tool result explains it can only relay to
  people it knows. Requires a recognized sender (like save_note).
- Delivery: in `OracleLoop._converse`, right after the greeting, fetch
  pending messages, speak each ("By the way, <from> left you a message: …"),
  mark delivered.

## 2. Weather tool

- `get_weather` tool (no params): fetches Open-Meteo current weather (no API
  key) for `settings.weather_lat/lon` (default Liverpool 53.41, -2.99),
  returns a compact summary string (temp, feels-like, condition, wind).
- Injected `weather_fetch` callable on ChatBrain for tests; real
  implementation uses urllib with a 5 s timeout; failure → friendly tool
  result, never an exception.

## 3. Streaming speech

- `ChatBrain.respond(..., on_sentence=None)`: with a callback, the LLM call
  streams; completed sentences are emitted to `on_sentence` as they arrive,
  so speech starts after the first sentence (~1 s) instead of the full
  response. Tool-call deltas are accumulated silently; the follow-up call
  after tool execution streams too. Return value stays the full text.
- Oracle passes `on_sentence=speaker.speak` and no longer re-speaks the
  returned text; terminal chat stays non-streaming.
- Fakes: `FakeCompletions` gains stream support (content split into deltas,
  tool-call delta chunks) to test sentence emission and tool accumulation.

Each numbered item = test cycle + commit.

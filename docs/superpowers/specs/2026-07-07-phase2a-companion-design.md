# Phase 2a "Companion" — Memory Across Visits, Wider Personality, Wake/Sleep

**Date:** 2026-07-07 · **Status:** approved in conversation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (delivers
the memory half of Phase 2; voice/speaker-ID and messaging remain Phase 2b/3).

## Goal

Reachy remembers people between visits, chats like a companion rather than a
librarian, and visibly sleeps when alone / wakes when someone appears.

## Design

### 1. Persistent per-person memory

- **`memories` table** (finally lands, per the parent spec):
  `memory_id, person_id, text, vector(384), created_at`.
- **Three ways memories are made:**
  1. **Auto-summary:** when a conversation ends, the brain asks the LLM for
     up to 3 short notes worth remembering about that person (nothing for
     trivial chit-chat) and stores them.
  2. **`save_note` tool:** "remember that I prefer short answers" → stored
     immediately, attributed to the recognized speaker.
  3. (Phase 2b: "note that X" team memos — out of scope here.)
- **Recall:** every turn, alongside the docs search, the question is
  searched against the *current speaker's* memories; hits are injected as
  "[What I remember about <name>]" context. The greeting flow can therefore
  produce "welcome back" continuity naturally.
- **Brain API:** `begin_conversation(person_id, name)` (reset + identity),
  `end_conversation()` (summarize → store → reset). The Oracle calls these
  at greet/goodbye. Terminal `chat` runs anonymous (no person) — docs-only
  retrieval, notes disabled.

### 2. Wider personality

Prompt v2: a curious desk companion — has opinions, banters, occasionally
asks a question back, references what it remembers about you; still the
demo-library expert; still signals off-library answers casually; still 1–2
spoken sentences.

### 3. Wake / sleep mode

- No face for `idle_sleep_s` (default 300 s, `REACHY_VEC_IDLE_SLEEP_S`) →
  the robot performs **sleep** (SDK `goto_sleep()` — a visible slump).
- Face appears while asleep → **wake** (SDK `wake_up()`) before greeting.
- `Body.perform` gains special motions `"sleep"`/`"wake"` mapped to the SDK
  calls (keyframes don't apply); `NullBody` no-ops as usual.

## Error handling

- Summarization failure at conversation end → log, drop the summary, never
  block the goodbye.
- `save_note` with no recognized speaker → tool result explains it can't
  save without knowing who's asking.

## Privacy

Memories are per-person, local (LanceDB), and inspectable/deletable like
everything else in `data/lancedb`. Enrollment consent already covers "I will
remember you"; the offer phrasing stays explicit.

## Out of scope

Speaker/voice ID, cross-person memory queries ("what did Alice say"),
messages, barge-in/streaming speech, fish-speech.

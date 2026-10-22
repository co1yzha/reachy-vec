---
title: Barge-in (interruption)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [barge-in, tts]
sources: [raw/docs/architecture.md]
---

Phase 4c. Talking over a reply interrupts it, gated by `BARGE_IN` (on by
default).

- Talking over a reply stops the current sentence (`Speaker.stop()`), abandons
  the in-flight LLM stream (the **partial reply is kept in history**), and makes
  the interruption the next turn inside the [[oracle-loop]].
- Implemented via a `BargeInMonitor` watching the mic + `SpeechInterrupted` in
  `ChatBrain`.
- **Caveat:** on-robot audio-out `stop()` is best-effort (per-sentence buffer);
  the `say`/local backends cut mid-sentence. See [[tts-backends]].


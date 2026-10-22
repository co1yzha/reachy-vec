---
title: Oracle loop
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [state-machine, protocol]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

`brain/oracle.py:OracleLoop` is a **synchronous state machine** — all
dependencies injected, so the whole thing runs in pytest on fakes (the
[[protocol-pattern]]).

```
idle (poll faces) → greet (spoken at most every GREET_COOLDOWN_S/person,
silent nod otherwise) → deliver queued [[messenger-pipeline]] messages →
listen/think/speak loop → 30 s silence ends the visit (memories distilled) →
back to idle.
```

- THINKING is the [[rag-pipeline]] (`ChatBrain.respond`).
- Identity each turn comes from the [[identity-rule]].
- No faces for `IDLE_SLEEP_S` (5 min) → sleep motion; wakes on the next face.
- OpenAI errors produce a spoken apology and keep listening.
- Everything heard/said is logged to `data/reachy.log` (delete to forget).


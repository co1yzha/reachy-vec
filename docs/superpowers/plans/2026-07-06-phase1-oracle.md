# Phase 1: Oracle — Plan Index

Four independently executable sub-plans implementing the
[Phase 1 design spec](../specs/2026-07-06-phase1-oracle-design.md).

| Sub-plan | Delivers | Depends on |
|---|---|---|
| [1a — Body](2026-07-06-phase1a-body.md) | Motion keyframes, `Body` protocol, sim playback, null fallback | — |
| [1b — Voice](2026-07-06-phase1b-voice.md) | `Speaker` (`say` backend), `Transcriber` (VAD + faster-whisper) | — |
| [1c — Faces](2026-07-06-phase1c-faces.md) | people/greetings tables, `FaceMatcher` (insightface), webcam `enroll` CLI | Phase 0 store |
| [1d — Oracle loop](2026-07-06-phase1d-oracle-loop.md) | `OracleLoop` state machine, robot-led enrollment, `reachy-vec run` | 1a + 1b + 1c |

1a, 1b, 1c are mutually independent — any order (or parallel). 1d ties them
together and ends with the Phase 1 milestone smoke test: walk up, get greeted
by name in the sim viewer, ask a question out loud, hear the answer.

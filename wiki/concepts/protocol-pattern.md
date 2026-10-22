---
title: Protocol pattern (heavy deps behind fakes)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [protocol, testing]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

The core pattern of reachy-vec: **every heavy dependency sits behind a small
`Protocol` with a test fake** — `Embedder`, `Body`, `Transcriber`, `Speaker`,
`FaceMatcher`, `SpeakerIdentifier`, `Camera` (plus `AudioSource`/`AudioSink`
for on-robot media).

- The whole [[oracle-loop]] runs in pytest with **no devices, models, or
  network** — fakes live in `tests/conftest.py`.
- Real implementations **lazy-load** their model on the [[mac-brain]] inside the method, so
  `import reachy_vec` and the test suite stay fast (torch, insightface,
  sentence-transformers, faster-whisper, mlx-audio are all deferred).
- **When adding a capability, keep this shape:** protocol in the module, real
  impl lazy-loads, fake in `conftest.py`. Note: pytest fakes can pass while
  production wiring is broken — hardware-facing changes still need the
  `docs/testing.md` smoke checklist on the real robot.


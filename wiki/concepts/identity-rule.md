---
title: Identity rule (never guess)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [face-id, speaker-id, state-machine]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

The design-spec rule: **never guess.** Every utterance is voice-identified
([[ecapa-tdnn]]) and fused with the face observation ([[insightface]]) into a
per-turn `TurnIdentity` via `perception/fusion.py: fuse(face_obs, voice_obs)`:

- voice **knows** a person → that person (they may be off-camera)
- voice **confidently unknown** → anonymous (side-effect tools refuse)
- voice **can't tell** → the recognized face, else anonymous

Threshold behaviour (both modalities): score ≥ threshold → known; within
**0.05 below** threshold → treated as **no face/voice at all** (a bad angle of
a known person is neither greeted nor re-enrolled); lower → unknown.

Notes, memories, and messages follow **whoever actually spoke** — an unknown
voice is answered but **never written to the store**.


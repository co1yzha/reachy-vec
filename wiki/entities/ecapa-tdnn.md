---
title: ECAPA-TDNN (speaker ID)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [speaker-id]
sources: [raw/docs/pipelines.md]
---

speechbrain ECAPA-TDNN `spkrec-ecapa-voxceleb` (192-dim, local) in
`perception/voice.py` answers *who is talking*.

- Every utterance's raw audio rides along with its transcript; audio shorter
  than `VOICE_MIN_UTTERANCE_S` (1 s) is "can't tell".
- ECAPA embeds the audio; k-NN majority vote (k=5, cosine) against the `voices` table in the [[lancedb-store]], gated by `VOICE_THRESHOLD` (0.30) with the same 0.05 borderline margin
  as faces.
- Voice profiles come from two sources: **enrolled** (one phrase after face
  enrollment) and **passive** (banked after a confident solo face match, capped
  at 10 rows per person). Raw audio is never persisted — only embeddings.
- The result is fused with the face observation in the [[identity-rule]].


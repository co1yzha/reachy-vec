---
title: insightface (face ID)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [face-id]
sources: [raw/docs/pipelines.md, raw/docs/architecture.md]
---

insightface `buffalo_s` (512-dim, local via onnxruntime) in `perception/face.py`
answers *who is this*.

1. Poll the webcam (`CAMERA_INDEX`); detect faces (det size 640×640); the
   largest bounding box wins → normalized 512-dim embedding.
2. k-NN (k=5, cosine) against the `people` table in the [[lancedb-store]], with a **majority vote** over
   person_ids; the winner's best similarity is the score.
3. Decision under the [[identity-rule]]: score ≥ `FACE_THRESHOLD` (0.45) →
   known; within 0.05 below → treated as **no face at all**; lower → unknown
   (enrollment offered after 3 stable polls).

**Enrollment** (`enroll_person`): 5 guided captures, each stored as its own
`people` row; frames saved to `data/faces/` for audit/re-embedding.


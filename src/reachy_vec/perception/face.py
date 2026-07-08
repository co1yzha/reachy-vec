"""Face recognition: detect -> 512-dim embedding -> store match.

Never guesses: below-threshold matches come back as unknown observations;
borderline matches (just under threshold) come back as "no face" so the
loop neither greets nor offers enrollment (probably a bad angle of a
known person).
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import FaceRow

logger = logging.getLogger(__name__)

BORDERLINE_MARGIN = 0.05


@dataclass(frozen=True)
class Observation:
    person_id: str | None  # None = face present but unknown
    name: str | None
    score: float
    face_count: int = 1  # faces detected in the frame this came from


class FaceMatcher(Protocol):
    def observe(self, frame) -> Observation | None: ...
    def embed(self, frame) -> list[float] | None: ...


class InsightFaceMatcher:
    """buffalo_s embeddings matched against the people table (lazy model load)."""

    def __init__(self, store: Store, threshold: float | None = None):
        self._store = store
        self._threshold = threshold if threshold is not None else settings.face_threshold
        self._app = None
        self.last_bbox: tuple[int, int, int, int] | None = None  # for preview overlay
        self.last_face_count: int = 0  # faces in the last frame (passive backfill gate)

    def _load(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name="buffalo_s")
            self._app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, frame) -> list[float] | None:
        self._load()
        faces = self._app.get(frame)
        self.last_face_count = len(faces)
        if not faces:
            self.last_bbox = None
            return None
        largest = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        self.last_bbox = tuple(int(v) for v in largest.bbox)
        return largest.normed_embedding.tolist()

    def observe(self, frame) -> Observation | None:
        vector = self.embed(frame)
        if vector is None:
            return None
        match = self._store.match_face(vector)
        count = self.last_face_count
        if match is None:
            return Observation(person_id=None, name=None, score=0.0, face_count=count)
        person_id, name, score = match
        if score >= self._threshold:
            return Observation(person_id=person_id, name=name, score=score, face_count=count)
        if score >= self._threshold - BORDERLINE_MARGIN:
            return None
        return Observation(person_id=None, name=None, score=score, face_count=count)


ENROLL_PROMPTS = [
    "Look straight at the camera.",
    "Turn slightly left.",
    "Turn slightly right.",
    "Tilt your head up a little.",
    "One more, straight ahead.",
]


def enroll_person(
    name: str,
    camera,
    matcher: FaceMatcher,
    store: Store,
    prompt: Callable[[str], None],
    n_frames: int = 5,
    faces_dir: Path | None = None,
) -> str | None:
    """Capture n_frames embeddings; returns person_id or None if no usable face.

    With faces_dir set, each accepted frame is also saved there as
    {person_id}-{i}.jpg for later audit or re-embedding.
    """
    person_id = f"person-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC).isoformat()
    rows: list[FaceRow] = []
    for i in range(n_frames):
        prompt(ENROLL_PROMPTS[i % len(ENROLL_PROMPTS)])
        frame = camera.read()
        vector = matcher.embed(frame) if frame is not None else None
        if vector is None:
            continue
        if faces_dir is not None:
            import cv2

            faces_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(faces_dir / f"{person_id}-{i}.jpg"), frame)
        rows.append(
            FaceRow(
                embedding_id=f"{person_id}:{i}",
                person_id=person_id,
                name=name,
                vector=vector,
                created_at=now,
            )
        )
    if not rows:
        logger.warning("Enrollment for %r captured no usable faces.", name)
        return None
    store.add_face_rows(rows)
    return person_id

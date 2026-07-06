# Phase 1c: Faces — Recognition & Enrollment Storage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `people` + `greetings` tables in LanceDB, a `FaceMatcher` protocol (insightface implementation) over the Mac webcam, and a keyboard-driven `reachy-vec enroll` that makes you recognizable.

**Architecture:** Face embeddings are 512-dim (insightface), one LanceDB row per captured frame; matching is k-NN majority vote with a cosine threshold. `perception/camera.py` isolates frame capture (`Camera` protocol; `WebcamCamera` via OpenCV). `perception/face.py` isolates detection/embedding/matching (`FaceMatcher` protocol; `InsightFaceMatcher` lazy-loads the buffalo_s model). The `enroll` CLI captures N frames and writes rows; greeting cooldown state lives in a `greetings` table.

**Tech Stack:** insightface (buffalo_s), onnxruntime, opencv-python, lancedb, pytest.

**Depends on:** Phase 0 store. **Unblocks:** 1d.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- Face embedding dimension **512** (`FACE_EMBEDDING_DIM = 512`) — distinct from the 384-dim text embeddings.
- Face threshold from `settings.face_threshold` (default **0.45** cosine similarity); below = unknown; never guess.
- Tests never load insightface or open the webcam; fakes only.
- Commit after every green test cycle; conventional-commit messages.

---

### Task 1: People + greetings tables

**Files:**
- Modify: `src/reachy_vec/store/schemas.py`
- Modify: `src/reachy_vec/store/db.py`
- Test: `tests/test_people_store.py`

**Interfaces:**
- Consumes: existing `Store` (Phase 0).
- Produces: `FACE_EMBEDDING_DIM = 512` (in `schemas.py`); `FaceRow(LanceModel)`: `embedding_id: str, person_id: str, name: str, vector: Vector(512), created_at: str`; `Store.add_face_rows(rows: list[FaceRow])`, `Store.match_face(vector: list[float], k: int = 5) -> tuple[str, str, float] | None` ((person_id, name, cosine score of best row of majority person) or None if table empty), `Store.people_count() -> int` (distinct persons), `Store.get_last_greeted(person_id: str) -> str | None`, `Store.set_last_greeted(person_id: str, when_iso: str)`.

- [x] **Step 1: Write the failing test**

`tests/test_people_store.py`:

```python
import random

from reachy_vec.store.db import Store
from reachy_vec.store.schemas import FACE_EMBEDDING_DIM, FaceRow


def unit_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(FACE_EMBEDDING_DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def rows_for(person_id: str, name: str, seeds: list[int]) -> list[FaceRow]:
    return [
        FaceRow(
            embedding_id=f"{person_id}:{s}",
            person_id=person_id,
            name=name,
            vector=unit_vector(s),
            created_at="2026-07-06T00:00:00+00:00",
        )
        for s in seeds
    ]


def test_match_face_returns_majority_person_with_score(tmp_path):
    store = Store(tmp_path / "db")
    store.add_face_rows(rows_for("p1", "Alice", [1, 2, 3]))
    store.add_face_rows(rows_for("p2", "Bob", [10, 11, 12]))
    person_id, name, score = store.match_face(unit_vector(1), k=3)
    assert (person_id, name) == ("p1", "Alice")
    assert score > 0.99  # exact vector -> cosine ~1


def test_match_face_empty_table_returns_none(tmp_path):
    assert Store(tmp_path / "db").match_face(unit_vector(1)) is None


def test_people_count_counts_distinct_persons(tmp_path):
    store = Store(tmp_path / "db")
    store.add_face_rows(rows_for("p1", "Alice", [1, 2]))
    store.add_face_rows(rows_for("p2", "Bob", [3]))
    assert store.people_count() == 2


def test_greeting_roundtrip(tmp_path):
    store = Store(tmp_path / "db")
    assert store.get_last_greeted("p1") is None
    store.set_last_greeted("p1", "2026-07-06T10:00:00+00:00")
    assert store.get_last_greeted("p1") == "2026-07-06T10:00:00+00:00"
    store.set_last_greeted("p1", "2026-07-06T12:00:00+00:00")  # upsert
    assert store.get_last_greeted("p1") == "2026-07-06T12:00:00+00:00"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_people_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'FACE_EMBEDDING_DIM'`.

- [x] **Step 3: Extend the schemas**

Append to `src/reachy_vec/store/schemas.py`:

```python
FACE_EMBEDDING_DIM = 512


class FaceRow(LanceModel):
    embedding_id: str
    person_id: str
    name: str
    vector: Vector(FACE_EMBEDDING_DIM)
    created_at: str  # ISO-8601 UTC


class GreetingRow(LanceModel):
    person_id: str
    last_greeted: str  # ISO-8601 UTC
```

(and update the module docstring: docs + people + greetings implemented; memories/messages remain future.)

- [x] **Step 4: Extend the store**

In `src/reachy_vec/store/db.py`, import `FaceRow, GreetingRow`, add table constants `PEOPLE_TABLE = "people"`, `GREETINGS_TABLE = "greetings"`, a generic `_table(name, schema)` helper mirroring `_docs()`, and:

```python
    def add_face_rows(self, rows: list[FaceRow]) -> None:
        if rows:
            self._table(PEOPLE_TABLE, FaceRow).add(rows)

    def match_face(self, vector: list[float], k: int = 5):
        table = self._table(PEOPLE_TABLE, FaceRow)
        if table.count_rows() == 0:
            return None
        hits = (
            table.search(vector).metric("cosine").limit(k).to_pydantic(FaceRow)
        )
        # majority vote over person_id; score = best cosine of the winner
        counts: dict[str, int] = {}
        for h in hits:
            counts[h.person_id] = counts.get(h.person_id, 0) + 1
        winner = max(counts, key=counts.get)
        # recompute best similarity for the winner via a fresh scored search
        scored = table.search(vector).metric("cosine").limit(k).to_list()
        best = min(
            (row["_distance"] for row in scored if row["person_id"] == winner),
        )
        name = next(h.name for h in hits if h.person_id == winner)
        return winner, name, 1.0 - best

    def people_count(self) -> int:
        table = self._table(PEOPLE_TABLE, FaceRow)
        return len({r["person_id"] for r in table.to_arrow().to_pylist()})

    def get_last_greeted(self, person_id: str) -> str | None:
        table = self._table(GREETINGS_TABLE, GreetingRow)
        rows = table.search().where(f"person_id = '{person_id}'").to_list()
        return rows[0]["last_greeted"] if rows else None

    def set_last_greeted(self, person_id: str, when_iso: str) -> None:
        table = self._table(GREETINGS_TABLE, GreetingRow)
        table.delete(f"person_id = '{person_id}'")
        table.add([GreetingRow(person_id=person_id, last_greeted=when_iso)])
```

Refactor `_docs()` to use `_table(DOCS_TABLE, DocChunk)`:

```python
    def _table(self, name: str, schema) -> lancedb.table.Table:
        if name not in self._db.list_tables().tables:
            self._db.create_table(name, schema=schema)
        return self._db.open_table(name)

    def _docs(self) -> lancedb.table.Table:
        return self._table(DOCS_TABLE, DocChunk)
```

- [x] **Step 5: Run tests, then commit**

Run: `uv run pytest -q` — expected: all PASS (adjust to `.to_list()`-based approaches if a lancedb API differs; keep the produced signatures identical).

```bash
git add src/reachy_vec/store/schemas.py src/reachy_vec/store/db.py tests/test_people_store.py
git commit -m "feat: people and greetings tables with cosine face matching"
```

---

### Task 2: Camera + FaceMatcher + enroll CLI

**Files:**
- Create: `src/reachy_vec/perception/camera.py`
- Modify: `src/reachy_vec/perception/face.py` (docstring stub)
- Modify: `src/reachy_vec/cli/enroll.py`
- Modify: `src/reachy_vec/config.py` (add `face_threshold: float = 0.45`, `camera_index: int = 0`)
- Modify: `tests/conftest.py` (add `FakeCamera`, `FakeFaceMatcher` for 1d reuse)
- Test: `tests/test_face.py`

**Interfaces:**
- Consumes: `Store.add_face_rows`, `Store.match_face` (Task 1).
- Produces: `Camera` protocol with `read() -> "np.ndarray | None"`; `WebcamCamera(index: int)`; `Observation` dataclass `(person_id: str | None, name: str | None, score: float)` where `person_id=None` means unknown-face-present; `FaceMatcher` protocol with `observe(frame) -> Observation | None` (None = no face at all) and `embed(frame) -> list[float] | None`; `InsightFaceMatcher(store, threshold)`; `enroll_person(name, camera, matcher, store, prompts, n_frames=5) -> str | None` (returns person_id, None if capture failed); working `reachy-vec enroll NAME`.

- [x] **Step 1: Add dependencies**

```bash
uv add insightface onnxruntime opencv-python
```

(Remove `insightface` from the `perception` extra; `speechbrain` stays there for Phase 2.)

- [x] **Step 2: Write the failing test**

Append to `tests/conftest.py`:

```python
class FakeCamera:
    """Serves scripted 'frames' (any object; matchers below don't inspect them)."""

    def __init__(self, frames: list):
        self._it = iter(frames)

    def read(self):
        return next(self._it, None)


class FakeFaceMatcher:
    """Scripted observations + constant embeddings."""

    def __init__(self, observations: list, embedding: list[float] | None = None):
        self._it = iter(observations)
        self._embedding = embedding

    def observe(self, frame):
        return next(self._it, None)

    def embed(self, frame):
        return self._embedding
```

`tests/test_face.py`:

```python
from reachy_vec.perception.face import Observation, enroll_person
from reachy_vec.store.db import Store

from tests.conftest import FakeCamera, FakeFaceMatcher


def test_enroll_person_stores_frames_and_is_matchable(tmp_path):
    store = Store(tmp_path / "db")
    vec = [1.0] + [0.0] * 511
    camera = FakeCamera(frames=["f"] * 5)
    matcher = FakeFaceMatcher(observations=[], embedding=vec)
    prompts: list[str] = []

    person_id = enroll_person("Alice", camera, matcher, store, prompts.append)

    assert person_id is not None
    assert store.people_count() == 1
    matched = store.match_face(vec)
    assert matched is not None and matched[1] == "Alice"
    assert len(prompts) == 5  # one guidance prompt per capture


def test_enroll_person_fails_gracefully_without_face(tmp_path):
    store = Store(tmp_path / "db")
    camera = FakeCamera(frames=["f"] * 5)
    matcher = FakeFaceMatcher(observations=[], embedding=None)  # never sees a face
    assert enroll_person("Alice", camera, matcher, store, lambda _: None) is None
    assert store.people_count() == 0


def test_observation_unknown_vs_known():
    unknown = Observation(person_id=None, name=None, score=0.2)
    known = Observation(person_id="p1", name="Alice", score=0.9)
    assert unknown.person_id is None and known.person_id == "p1"
```

- [x] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_face.py -v`
Expected: FAIL with `ImportError: cannot import name 'Observation'`.

- [x] **Step 4: Write camera and face modules**

`src/reachy_vec/perception/camera.py`:

```python
"""Frame sources. Phase 1 uses the Mac webcam; the robot camera slots in later."""

from typing import Protocol


class Camera(Protocol):
    def read(self): ...  # returns an ndarray frame or None


class WebcamCamera:
    def __init__(self, index: int = 0):
        import cv2

        self._cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None
```

Replace `src/reachy_vec/perception/face.py` with:

```python
"""Face recognition: detect -> 512-dim embedding -> store match.

Never guesses: below-threshold matches come back as unknown observations.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import FaceRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    person_id: str | None  # None = face present but unknown
    name: str | None
    score: float


class FaceMatcher(Protocol):
    def observe(self, frame) -> Observation | None: ...
    def embed(self, frame) -> list[float] | None: ...


class InsightFaceMatcher:
    """buffalo_s embeddings matched against the people table (lazy model load)."""

    def __init__(self, store: Store, threshold: float | None = None):
        self._store = store
        self._threshold = threshold if threshold is not None else settings.face_threshold
        self._app = None

    def _load(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name="buffalo_s")
            self._app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, frame) -> list[float] | None:
        self._load()
        faces = self._app.get(frame)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding.tolist()

    def observe(self, frame) -> Observation | None:
        vector = self.embed(frame)
        if vector is None:
            return None
        match = self._store.match_face(vector)
        if match is None:
            return Observation(person_id=None, name=None, score=0.0)
        person_id, name, score = match
        if score >= self._threshold:
            return Observation(person_id=person_id, name=name, score=score)
        if score >= self._threshold - 0.05:
            # Borderline (spec §5): probably a bad angle of a known person.
            # Report "no face" so the loop neither greets nor offers enrollment.
            return None
        return Observation(person_id=None, name=None, score=score)


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
) -> str | None:
    """Capture n_frames embeddings; returns person_id or None if no usable face."""
    person_id = f"person-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    rows: list[FaceRow] = []
    for i in range(n_frames):
        prompt(ENROLL_PROMPTS[i % len(ENROLL_PROMPTS)])
        frame = camera.read()
        vector = matcher.embed(frame) if frame is not None else None
        if vector is None:
            continue
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
```

- [x] **Step 5: Wire settings and the enroll CLI**

In `src/reachy_vec/config.py`, add to `Settings`:

```python
    # Perception
    face_threshold: float = 0.45  # cosine similarity; below = unknown
    camera_index: int = 0
```

Replace `src/reachy_vec/cli/enroll.py` with:

```python
import time

import typer

from reachy_vec.config import settings
from reachy_vec.perception.camera import WebcamCamera
from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
from reachy_vec.store.db import Store


def enroll(name: str) -> None:
    """Enroll a teammate's face from the webcam (keyboard-guided)."""
    store = Store(settings.lancedb_dir)
    camera = WebcamCamera(settings.camera_index)
    matcher = InsightFaceMatcher(store)

    def prompt(msg: str) -> None:
        typer.echo(f">> {msg}")
        time.sleep(1.5)  # give the person a beat to move

    person_id = enroll_person(name, camera, matcher, store, prompt)
    if person_id is None:
        typer.echo("No usable face captured - check lighting/camera and retry.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Enrolled {name} ({person_id}) with face data.")
```

- [x] **Step 6: Run tests, then commit**

Run: `uv run pytest -q` — expected: all PASS.

```bash
git add pyproject.toml uv.lock src/reachy_vec/perception/ src/reachy_vec/cli/enroll.py src/reachy_vec/config.py tests/conftest.py tests/test_face.py
git commit -m "feat: webcam face recognition, enrollment, and matcher protocol"
```

- [x] **Step 7: Manual smoke test (needs webcam permission)**

```bash
uv run reachy-vec enroll "YourName"          # first run downloads buffalo_s (~120 MB)
uv run python -c "
from reachy_vec.config import settings
from reachy_vec.perception.camera import WebcamCamera
from reachy_vec.perception.face import InsightFaceMatcher
from reachy_vec.store.db import Store
m = InsightFaceMatcher(Store(settings.lancedb_dir))
obs = m.observe(WebcamCamera(settings.camera_index).read())
print(obs)
"
```

Expected: enrollment prints five prompts and succeeds; the observe call prints your name with score ≥ 0.45. macOS will prompt for camera permission — accept it.

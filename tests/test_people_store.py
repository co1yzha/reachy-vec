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

from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow


def vec(seed: float) -> list[float]:
    return [seed] * VOICE_EMBEDDING_DIM


def row(i: int, person: str, name: str, seed: float, source: str = "enrolled") -> VoiceRow:
    return VoiceRow(
        voice_id=f"{person}:{i}",
        person_id=person,
        name=name,
        vector=vec(seed),
        created_at=f"2026-07-08T00:00:{i:02d}+00:00",
        source=source,
    )


def test_match_voice_empty_store_returns_none(tmp_path):
    assert Store(tmp_path / "db").match_voice(vec(0.5)) is None


def test_match_voice_majority_vote(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows(
        [row(0, "p1", "Alice", 0.9), row(1, "p1", "Alice", 0.9), row(2, "p2", "Bob", 0.1)]
    )
    person_id, name, score = store.match_voice(vec(0.9))
    assert (person_id, name) == ("p1", "Alice")
    assert score > 0.99


def test_passive_prune_keeps_newest(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows([row(i, "p1", "Alice", 0.5, source="passive") for i in range(4)])
    store.add_voice_rows([row(9, "p1", "Alice", 0.5)])  # enrolled row never pruned
    store.prune_passive_voices("p1", keep=2)
    assert store.passive_voice_count("p1") == 2
    remaining = {
        r["voice_id"] for r in store._table("voices", VoiceRow).to_arrow().to_pylist()
    }
    assert remaining == {"p1:9", "p1:2", "p1:3"}


def test_passive_count_ignores_other_people(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows([row(0, "p1", "Alice", 0.5, source="passive")])
    store.add_voice_rows([row(0, "p2", "Bob", 0.5, source="passive")])
    assert store.passive_voice_count("p1") == 1

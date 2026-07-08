from reachy_vec.store.db import Store
from reachy_vec.store.schemas import MemoryRow
from tests.conftest import FakeEmbedder


def memory(mid: str, person: str, text: str) -> MemoryRow:
    return MemoryRow(
        memory_id=mid,
        person_id=person,
        text=text,
        vector=FakeEmbedder().embed([text])[0],
        created_at="2026-07-07T00:00:00+00:00",
    )


def test_search_memories_is_person_scoped(tmp_path):
    store = Store(tmp_path / "db")
    store.add_memories(
        [
            memory("m1", "p1", "prefers short answers"),
            memory("m2", "p2", "prefers short answers"),  # same text, other person
        ]
    )
    hits = store.search_memories(
        FakeEmbedder().embed(["prefers short answers"])[0], person_id="p1", k=5
    )
    assert [h.memory_id for h in hits] == ["m1"]


def test_search_memories_empty(tmp_path):
    store = Store(tmp_path / "db")
    assert store.search_memories(FakeEmbedder().embed(["x"])[0], person_id="p1") == []

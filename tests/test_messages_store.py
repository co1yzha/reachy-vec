from reachy_vec.store.db import Store
from reachy_vec.store.schemas import FaceRow, MessageRow

from tests.conftest import FakeEmbedder


def enroll(store: Store, person_id: str, name: str) -> None:
    store.add_face_rows(
        [
            FaceRow(
                embedding_id=f"{person_id}:0",
                person_id=person_id,
                name=name,
                vector=[0.5] * 512,
                created_at="2026-07-07T00:00:00+00:00",
            )
        ]
    )


def message(mid: str, to_person: str) -> MessageRow:
    return MessageRow(
        message_id=mid,
        from_person="p1",
        from_name="Yang",
        to_person=to_person,
        to_name="Bob",
        text="the meeting moved to 3",
        created_at="2026-07-07T00:00:00+00:00",
        delivered_at="",
    )


def test_pending_and_mark_delivered(tmp_path):
    store = Store(tmp_path / "db")
    store.add_message(message("msg1", "p2"))
    store.add_message(message("msg2", "p3"))
    pending = store.pending_messages_for("p2")
    assert [m.message_id for m in pending] == ["msg1"]
    store.mark_delivered("msg1")
    assert store.pending_messages_for("p2") == []
    assert store.pending_messages_for("p3") != []  # untouched


def test_find_person_by_name_case_insensitive(tmp_path):
    store = Store(tmp_path / "db")
    enroll(store, "p2", "Bob")
    assert store.find_person_by_name("bob") == ("p2", "Bob")
    assert store.find_person_by_name("BOB") == ("p2", "Bob")
    assert store.find_person_by_name("Carol") is None


def test_pending_empty_table(tmp_path):
    assert Store(tmp_path / "db").pending_messages_for("p2") == []

from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, MessageRow, VoiceRow


def enroll(store: Store, person_id: str, name: str) -> None:
    store.add_voice_rows(
        [
            VoiceRow(
                voice_id=f"{person_id}:0",
                person_id=person_id,
                name=name,
                vector=[0.5] * VOICE_EMBEDDING_DIM,
                created_at="2026-07-07T00:00:00+00:00",
                source="enrolled",
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


def test_concurrent_add_message_is_thread_safe(tmp_path):
    import threading

    store = Store(tmp_path / "db")
    errors = []

    def writer(prefix):
        try:
            for i in range(20):
                store.add_message(message(f"{prefix}{i}", "p2"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(p,)) for p in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(store.pending_messages_for("p2")) == 40

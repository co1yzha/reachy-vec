from reachy_vec.store.db import Store


def test_greeting_roundtrip(tmp_path):
    store = Store(tmp_path / "db")
    assert store.get_last_greeted("p1") is None
    store.set_last_greeted("p1", "2026-07-06T10:00:00+00:00")
    assert store.get_last_greeted("p1") == "2026-07-06T10:00:00+00:00"
    store.set_last_greeted("p1", "2026-07-06T12:00:00+00:00")  # upsert
    assert store.get_last_greeted("p1") == "2026-07-06T12:00:00+00:00"

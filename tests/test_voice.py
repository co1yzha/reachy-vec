"""EcapaSpeakerIdentifier decision logic, with embed() faked (no model load)."""

import numpy as np

from reachy_vec.perception.voice import EcapaSpeakerIdentifier
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow

SR = 16000

ALICE_VEC = [1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 1)


def enrolled_store(tmp_path) -> Store:
    store = Store(tmp_path / "db")
    store.add_voice_rows(
        [
            VoiceRow(
                voice_id="p1:0",
                person_id="p1",
                name="Alice",
                vector=ALICE_VEC,
                created_at="2026-07-08T00:00:00+00:00",
                source="enrolled",
            )
        ]
    )
    return store


def identifier(store, vector):
    ident = EcapaSpeakerIdentifier(store, threshold=0.30)
    ident.embed = lambda audio: vector  # bypass the model
    return ident


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def test_confident_match(tmp_path):
    obs = identifier(enrolled_store(tmp_path), ALICE_VEC).identify(audio(2))
    assert (obs.person_id, obs.name) == ("p1", "Alice")


def test_below_threshold_is_unknown(tmp_path):
    # orthogonal vector -> cosine 0.0 < 0.30
    vec = [0.0, 1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 2)
    obs = identifier(enrolled_store(tmp_path), vec).identify(audio(2))
    assert obs is not None and obs.person_id is None


def test_borderline_is_cant_tell(tmp_path, monkeypatch):
    ident = EcapaSpeakerIdentifier(enrolled_store(tmp_path), threshold=0.30)
    ident.embed = lambda a: ALICE_VEC
    monkeypatch.setattr(ident._store, "match_voice", lambda v, k=5: ("p1", "Alice", 0.27))
    assert ident.identify(audio(2)) is None  # 0.27 within 0.05 under 0.30


def test_short_audio_is_cant_tell(tmp_path):
    real = EcapaSpeakerIdentifier(enrolled_store(tmp_path))
    assert real.embed(audio(0.5)) is None  # never loads the model
    assert real.identify(audio(0.5)) is None


def test_none_audio_is_cant_tell(tmp_path):
    real = EcapaSpeakerIdentifier(enrolled_store(tmp_path))
    assert real.identify(None) is None


def test_empty_voice_store_is_cant_tell(tmp_path):
    assert identifier(Store(tmp_path / "db"), ALICE_VEC).identify(audio(2)) is None


def test_broken_model_degrades_to_cant_tell(tmp_path, monkeypatch):
    ident = EcapaSpeakerIdentifier(enrolled_store(tmp_path))

    def boom(self):
        raise RuntimeError("no download")

    monkeypatch.setattr(
        "reachy_vec.perception.voice.EcapaSpeakerIdentifier._load_model", boom
    )
    assert ident.identify(audio(2)) is None
    assert ident.identify(audio(2)) is None  # stays disabled, no retry storm

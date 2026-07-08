from reachy_vec.store.embeddings import EMBEDDING_DIM, Embedder
from tests.conftest import FakeEmbedder


def test_embedding_dim_is_384():
    assert EMBEDDING_DIM == 384


def test_fake_embedder_is_deterministic_and_conforms():
    embedder: Embedder = FakeEmbedder()
    first = embedder.embed(["hello", "world"])
    second = embedder.embed(["hello", "world"])
    assert first == second
    assert len(first) == 2
    assert all(len(vec) == EMBEDDING_DIM for vec in first)
    assert first[0] != first[1]


def test_fake_embedder_embed_query_matches_embed():
    embedder = FakeEmbedder()
    assert embedder.embed_query("hello") == embedder.embed(["hello"])[0]


def test_bge_embed_query_prepends_prefix(monkeypatch):
    from reachy_vec.store.embeddings import BgeEmbedder

    embedder = BgeEmbedder("any-model", query_prefix="QP: ")
    captured = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return [[0.0] * EMBEDDING_DIM]

    monkeypatch.setattr(embedder, "embed", fake_embed)
    vector = embedder.embed_query("hello")
    assert captured["texts"] == ["QP: hello"]
    assert len(vector) == EMBEDDING_DIM


def test_bge_default_prefix_is_the_bge_instruction():
    from reachy_vec.store.embeddings import BGE_QUERY_PREFIX, BgeEmbedder

    assert BGE_QUERY_PREFIX == (
        "Represent this sentence for searching relevant passages: "
    )
    assert BgeEmbedder("any-model")._query_prefix == BGE_QUERY_PREFIX


def test_settings_default_query_prefix():
    from reachy_vec.config import Settings

    assert Settings().embedding_query_prefix == (
        "Represent this sentence for searching relevant passages: "
    )

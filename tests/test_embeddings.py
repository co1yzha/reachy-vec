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

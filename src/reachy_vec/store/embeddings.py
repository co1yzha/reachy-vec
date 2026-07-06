"""Text embeddings: protocol + sentence-transformers implementation.

BAAI/bge-small-en-v1.5 produces 384-dim normalized vectors. Anything that
embeds (ingestion, RAG search, later face/voice matching) depends on the
Embedder protocol, not on sentence-transformers directly, so tests can
substitute a deterministic fake.
"""

from typing import Protocol

EMBEDDING_DIM = 384


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class BgeEmbedder:
    """sentence-transformers embedder; loads the model lazily on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()

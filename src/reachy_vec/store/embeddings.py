"""Text embeddings: protocol + sentence-transformers implementation.

BAAI/bge-small-en-v1.5 produces 384-dim normalized vectors. Anything that
embeds (ingestion, RAG search, later face/voice matching) depends on the
Embedder protocol, not on sentence-transformers directly, so tests can
substitute a deterministic fake.
"""

from typing import Protocol

EMBEDDING_DIM = 384

# BGE models expect this instruction on *queries* (not documents); see the
# bge-small-en-v1.5 model card. Applied at query time only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    """sentence-transformers embedder; loads the model lazily on first use."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self._query_prefix + text])[0]

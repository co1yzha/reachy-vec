"""LanceDB table schemas.

Phase 0 uses only `docs`. The people/memories/messages tables arrive with
Phases 1-3 and will be defined here when implemented.
"""

from lancedb.pydantic import LanceModel, Vector

from reachy_vec.store.embeddings import EMBEDDING_DIM


class DocChunk(LanceModel):
    chunk_id: str
    text: str
    vector: Vector(EMBEDDING_DIM)
    source: str
    ingested_at: str  # ISO-8601 UTC timestamp

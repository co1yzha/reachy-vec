"""LanceDB table schemas.

Implemented: docs (Phase 0), people + greetings (Phase 1).
Future: memories (Phase 2), messages (Phase 3).
"""

from lancedb.pydantic import LanceModel, Vector

from reachy_vec.store.embeddings import EMBEDDING_DIM


class DocChunk(LanceModel):
    chunk_id: str
    text: str
    vector: Vector(EMBEDDING_DIM)
    source: str
    ingested_at: str  # ISO-8601 UTC timestamp


FACE_EMBEDDING_DIM = 512


class FaceRow(LanceModel):
    embedding_id: str
    person_id: str
    name: str
    vector: Vector(FACE_EMBEDDING_DIM)
    created_at: str  # ISO-8601 UTC


class GreetingRow(LanceModel):
    person_id: str
    last_greeted: str  # ISO-8601 UTC

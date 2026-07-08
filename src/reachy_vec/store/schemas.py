"""LanceDB table schemas.

Implemented: docs (Phase 0), people + greetings (Phase 1), memories (Phase 2a),
voices (Phase 2b), messages (Phase 3).
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


class MemoryRow(LanceModel):
    memory_id: str
    person_id: str
    text: str
    vector: Vector(EMBEDDING_DIM)
    created_at: str  # ISO-8601 UTC


VOICE_EMBEDDING_DIM = 192  # speechbrain ECAPA-TDNN


class VoiceRow(LanceModel):
    voice_id: str
    person_id: str
    name: str
    vector: Vector(VOICE_EMBEDDING_DIM)
    created_at: str  # ISO-8601 UTC
    source: str  # "enrolled" | "passive"


class MessageRow(LanceModel):
    message_id: str
    from_person: str
    from_name: str
    to_person: str
    to_name: str
    text: str
    created_at: str    # ISO-8601 UTC
    delivered_at: str  # ISO-8601 UTC; empty string = pending

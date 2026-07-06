"""LanceDB table schemas.

Tables:
- people:   person_id, name, face_embeddings, voice_embeddings, preferences
- docs:     chunk_id, text, embedding, source, ingested_at
- memories: memory_id, person_id, text, embedding, timestamp
- messages: message_id, from_person, to_person, text, created_at, delivered_at
"""

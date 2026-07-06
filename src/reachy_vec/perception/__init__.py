"""Identity layer: face recognition + speaker identification, fused.

Modules:
- face:   insightface embeddings matched against the people table
- voice:  speechbrain ECAPA speaker embeddings
- fusion: combine face + voice; agreement -> identified, otherwise ask, never guess
"""

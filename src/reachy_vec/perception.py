"""Identity: face recognition + speaker identification, fused.

Face embeddings (insightface) and voice embeddings (speechbrain ECAPA) are
matched against the people table. Fusion rule: agreement -> identified;
disagreement or unknown -> ask, never guess.
"""

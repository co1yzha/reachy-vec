"""Identity layer: who is talking, from their voice alone.

Modules:
- voice:  speechbrain ECAPA speaker embeddings matched against the voices table
- fusion: per-turn identity from the voice observation; never guess
- camera, vision, occlusion: camera tools, currently not wired into `run`
"""

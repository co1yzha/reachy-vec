"""Shared test fakes: deterministic embedder and canned LLM client."""

import hashlib

from reachy_vec.store.embeddings import EMBEDDING_DIM


class FakeEmbedder:
    """Deterministic embeddings: same text -> same vector, no model download."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Repeat the 32-byte digest to fill EMBEDDING_DIM floats in [0, 1).
            raw = (digest * (EMBEDDING_DIM // len(digest) + 1))[:EMBEDDING_DIM]
            vectors.append([b / 256 for b in raw])
        return vectors


class FakeChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeChoiceMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(self._reply)


class FakeChat:
    def __init__(self, reply: str):
        self.completions = FakeCompletions(reply)


class FakeLLMClient:
    """Mimics the openai client surface used by brain.rag: client.chat.completions.create()."""

    def __init__(self, reply: str = "canned answer"):
        self.chat = FakeChat(reply)


class FakeSpeaker:
    """Records spoken lines."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeTranscriber:
    """Returns scripted utterances, then None (silence)."""

    def __init__(self, utterances: list[str]):
        self._it = iter(utterances)

    def listen_once(self, timeout_s: float) -> str | None:
        return next(self._it, None)


class FakeCamera:
    """Serves scripted 'frames' (any object; fakes below don't inspect them)."""

    def __init__(self, frames: list):
        self._it = iter(frames)

    def read(self):
        return next(self._it, None)


class FakeBody:
    def __init__(self):
        self.motions: list[str] = []

    def perform(self, motion: str) -> None:
        self.motions.append(motion)


class FakeFaceMatcher:
    """Scripted observations + constant embeddings."""

    def __init__(self, observations: list, embedding: list[float] | None = None):
        self._it = iter(observations)
        self._embedding = embedding

    def observe(self, frame):
        return next(self._it, None)

    def embed(self, frame):
        return self._embedding

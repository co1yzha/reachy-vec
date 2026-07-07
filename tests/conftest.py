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
    def __init__(self, content: str | None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeToolCall:
    def __init__(self, name: str, arguments: str, call_id: str = "call_1"):
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeChoice:
    def __init__(self, message: FakeChoiceMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeChoiceMessage):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    """Serves scripted messages in order (repeating the last one), records calls."""

    def __init__(self, messages: list[FakeChoiceMessage]):
        self._messages = messages
        self.calls: list[dict] = []

    @property
    def last_kwargs(self) -> dict | None:
        return self.calls[-1] if self.calls else None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._messages) - 1)
        return FakeResponse(self._messages[index])


class FakeChat:
    def __init__(self, messages: list[FakeChoiceMessage]):
        self.completions = FakeCompletions(messages)


class FakeLLMClient:
    """Mimics the openai client surface: client.chat.completions.create().

    FakeLLMClient(reply="x") answers "x" to every call;
    FakeLLMClient(messages=[...]) serves a scripted sequence (for tool calls).
    """

    def __init__(self, reply: str = "canned answer", messages=None):
        self.chat = FakeChat(messages or [FakeChoiceMessage(reply)])


class FakeBrain:
    """Scripted ChatBrain stand-in: echoes questions, records lifecycle."""

    def __init__(self, fail: bool = False):
        self.begun: list[tuple[str | None, str | None]] = []
        self.ended = 0
        self.asked: list[tuple[str, str | None]] = []
        self._fail = fail

    def begin_conversation(self, person_id, name) -> None:
        self.begun.append((person_id, name))

    def end_conversation(self) -> None:
        self.ended += 1

    def respond(self, question: str, speaker_name: str | None = None) -> str:
        if self._fail:
            raise RuntimeError("api down")
        self.asked.append((question, speaker_name))
        return f"answer to {question}"


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

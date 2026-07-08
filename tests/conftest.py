"""Shared test fakes: deterministic embedder and canned LLM client."""

import hashlib

from reachy_vec.audio.listen import Utterance
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


class FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeToolCallDelta:
    def __init__(self, index: int, call_id, name, arguments):
        self.index = index
        self.id = call_id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeStreamChunk:
    def __init__(self, delta: FakeDelta):
        self.choices = [type("C", (), {"delta": delta})()]


def _message_to_stream(message: FakeChoiceMessage):
    """Split a scripted message into streaming chunks (content in small deltas)."""
    chunks = []
    if message.tool_calls:
        for i, tc in enumerate(message.tool_calls):
            args = tc.function.arguments
            half = max(1, len(args) // 2)
            chunks.append(
                FakeStreamChunk(FakeDelta(tool_calls=[
                    FakeToolCallDelta(i, tc.id, tc.function.name, args[:half])
                ]))
            )
            chunks.append(
                FakeStreamChunk(FakeDelta(tool_calls=[
                    FakeToolCallDelta(i, None, None, args[half:])
                ]))
            )
    if message.content:
        text = message.content
        for start in range(0, len(text), 8):
            chunks.append(FakeStreamChunk(FakeDelta(content=text[start:start + 8])))
    return iter(chunks)


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
        message = self._messages[index]
        if kwargs.get("stream"):
            return _message_to_stream(message)
        return FakeResponse(message)


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

    def respond(self, question: str, speaker_name: str | None = None, on_sentence=None) -> str:
        if self._fail:
            raise RuntimeError("api down")
        self.asked.append((question, speaker_name))
        reply = f"answer to {question}"
        if on_sentence is not None:
            on_sentence(reply)
        return reply


class FakeSpeaker:
    """Records spoken lines."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeTranscriber:
    """Returns scripted utterances, then None (silence). Accepts plain strings
    or Utterance objects (for tests that script audio)."""

    def __init__(self, utterances: list):
        self._it = iter(utterances)

    def listen_once(self, timeout_s: float) -> Utterance | None:
        nxt = next(self._it, None)
        if nxt is None or isinstance(nxt, Utterance):
            return nxt
        return Utterance(text=nxt)


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

import json

from reachy_vec.brain.chat import ChatBrain
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import (
    FakeChoiceMessage,
    FakeEmbedder,
    FakeLLMClient,
    FakeToolCall,
)


def seeded_store(tmp_path) -> Store:
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder()
    text = "Demo: Food Mapping\nURL: https://example.org/food\nfights food insecurity"
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id="d1",
                text=text,
                vector=embedder.embed([text])[0],
                source="demo: Food Mapping",
                ingested_at="2026-07-07T00:00:00+00:00",
            )
        ]
    )
    return store


def make_brain(tmp_path, client, opener=None):
    return ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        opener=opener or (lambda url: None),
    )


def test_context_and_speaker_injected_each_turn(tmp_path):
    client = FakeLLMClient(reply="It's the Food Mapping demo.")
    brain = make_brain(tmp_path, client)
    reply = brain.respond("any demos about food?", speaker_name="Yang")
    assert reply == "It's the Food Mapping demo."
    sent = client.chat.completions.last_kwargs["messages"]
    assert sent[0]["role"] == "system" and "Reachy" in sent[0]["content"]
    assert "food insecurity" in sent[-1]["content"]      # retrieved context
    assert "Yang: any demos about food?" in sent[-1]["content"]


def test_history_carries_across_turns_and_reset_clears(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    brain.respond("first question", speaker_name="Yang")
    brain.respond("tell me more", speaker_name="Yang")
    sent = client.chat.completions.last_kwargs["messages"]
    joined = json.dumps(sent)
    assert "first question" in joined and "tell me more" in joined
    brain.reset()
    brain.respond("fresh start", speaker_name="Yang")
    sent = client.chat.completions.last_kwargs["messages"]
    assert "first question" not in json.dumps(sent)


def test_open_url_tool_executes_and_confirms(tmp_path):
    opened: list[str] = []
    tool_call = FakeToolCall(
        "open_url", json.dumps({"url": "https://example.org/food", "title": "Food Mapping"})
    )
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Opening the Food Mapping demo now!"),
        ]
    )
    brain = make_brain(tmp_path, client, opener=opened.append)
    reply = brain.respond("open the food mapping demo", speaker_name="Yang")
    assert opened == ["https://example.org/food"]
    assert reply == "Opening the Food Mapping demo now!"
    # second call carried the tool result back to the model
    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert any(
        m.get("role") == "tool" and "opened https://example.org/food" in m["content"]
        for m in second_call_messages
    )


def test_open_url_refuses_non_http(tmp_path):
    opened: list[str] = []
    tool_call = FakeToolCall("open_url", json.dumps({"url": "file:///etc/passwd"}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Sorry, I can't open that."),
        ]
    )
    brain = make_brain(tmp_path, client, opener=opened.append)
    brain.respond("open something weird")
    assert opened == []
    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert any(
        m.get("role") == "tool" and "refused" in m["content"] for m in second_call_messages
    )


def test_history_trimmed(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    for i in range(30):
        brain.respond(f"question {i}")
    sent = client.chat.completions.last_kwargs["messages"]
    assert len(sent) <= 22  # system + trimmed history

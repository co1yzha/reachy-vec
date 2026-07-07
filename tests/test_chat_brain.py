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


def test_begin_conversation_recalls_person_memories(tmp_path):
    from reachy_vec.store.schemas import MemoryRow

    client = FakeLLMClient(reply="Welcome back!")
    brain = make_brain(tmp_path, client)
    text = "prefers espresso over tea"
    brain._store.add_memories(
        [
            MemoryRow(
                memory_id="m1",
                person_id="p1",
                text=text,
                vector=FakeEmbedder().embed([text])[0],
                created_at="2026-07-07T00:00:00+00:00",
            )
        ]
    )
    brain.begin_conversation("p1", "Yang")
    brain.respond("any coffee tips?", speaker_name="Yang")
    sent = client.chat.completions.last_kwargs["messages"]
    assert "prefers espresso over tea" in sent[-1]["content"]


def test_save_note_tool_stores_attributed_memory(tmp_path):
    tool_call = FakeToolCall("save_note", json.dumps({"note": "Yang prefers short answers"}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Noted!"),
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Yang")
    assert brain.respond("remember I prefer short answers", speaker_name="Yang") == "Noted!"
    hits = brain._store.search_memories(
        FakeEmbedder().embed(["Yang prefers short answers"])[0], person_id="p1"
    )
    assert len(hits) == 1 and "short answers" in hits[0].text


def test_save_note_without_person_is_refused(tmp_path):
    tool_call = FakeToolCall("save_note", json.dumps({"note": "whatever"}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Sorry, I don't know who you are."),
        ]
    )
    brain = make_brain(tmp_path, client)  # no begin_conversation -> anonymous
    brain.respond("remember this")
    tool_result = client.chat.completions.calls[1]["messages"][-1]
    assert tool_result["role"] == "tool" and "don't know who" in tool_result["content"]


def test_end_conversation_stores_summary(tmp_path):
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage("nice chat"),                       # respond
            FakeChoiceMessage("- Yang is planning a demo day"),   # summary
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Yang")
    brain.respond("we're planning a demo day", speaker_name="Yang")
    brain.end_conversation()
    hits = brain._store.search_memories(
        FakeEmbedder().embed(["demo day"])[0], person_id="p1", k=5
    )
    assert any("demo day" in h.text for h in hits)
    assert brain._history == []  # reset after ending


def test_end_conversation_skips_summary_when_no_exchange(tmp_path):
    client = FakeLLMClient(reply="unused")
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Yang")
    brain.end_conversation()  # no respond() happened
    assert client.chat.completions.calls == []  # no summary LLM call


def enroll_bob(store):
    from reachy_vec.store.schemas import FaceRow

    store.add_face_rows(
        [
            FaceRow(
                embedding_id="p2:0",
                person_id="p2",
                name="Bob",
                vector=[0.5] * 512,
                created_at="2026-07-07T00:00:00+00:00",
            )
        ]
    )


def test_send_message_tool_queues_for_enrolled_recipient(tmp_path):
    tool_call = FakeToolCall(
        "send_message", json.dumps({"to_name": "bob", "message": "meeting moved to 3"})
    )
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Will do - I'll tell Bob when I see him."),
        ]
    )
    brain = make_brain(tmp_path, client)
    enroll_bob(brain._store)
    brain.begin_conversation("p1", "Yang")
    reply = brain.respond("tell bob the meeting moved to 3", speaker_name="Yang")
    assert "Bob" in reply
    pending = brain._store.pending_messages_for("p2")
    assert len(pending) == 1
    assert pending[0].text == "meeting moved to 3"
    assert pending[0].from_name == "Yang"


def test_send_message_to_unknown_recipient_refused(tmp_path):
    tool_call = FakeToolCall(
        "send_message", json.dumps({"to_name": "Carol", "message": "hi"})
    )
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Sorry, I haven't met Carol."),
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Yang")
    brain.respond("tell carol hi", speaker_name="Yang")
    tool_result = client.chat.completions.calls[1]["messages"][-1]
    assert "don't know anyone called Carol" in tool_result["content"]


def test_get_weather_tool_reports_conditions(tmp_path):
    tool_call = FakeToolCall("get_weather", "{}")
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("It's 18 degrees and partly cloudy out there!"),
        ]
    )
    brain = make_brain(tmp_path, client)
    brain._weather_fetch = lambda: {
        "temperature_2m": 18.2,
        "apparent_temperature": 17.1,
        "weather_code": 2,
        "wind_speed_10m": 14.0,
    }
    reply = brain.respond("what's the weather like?", speaker_name="Yang")
    assert "18" in reply
    tool_result = client.chat.completions.calls[1]["messages"][-1]
    assert "18.2" in tool_result["content"]
    assert "partly cloudy" in tool_result["content"]


def test_get_weather_failure_is_friendly(tmp_path):
    tool_call = FakeToolCall("get_weather", "{}")
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Couldn't check, sorry!"),
        ]
    )
    brain = make_brain(tmp_path, client)

    def broken():
        raise OSError("network down")

    brain._weather_fetch = broken
    brain.respond("weather?")
    tool_result = client.chat.completions.calls[1]["messages"][-1]
    assert "couldn't reach the weather service" in tool_result["content"]


def test_near_duplicate_memories_are_skipped(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Yang")
    brain._store_memories(["Yang prefers short answers"])
    brain._store_memories(["Yang prefers short answers"])   # exact dup -> skip
    brain._store_memories(["Yang is planning a demo day"])  # different -> stored
    vec = FakeEmbedder().embed(["Yang prefers short answers"])[0]
    hits = brain._store.search_memories(vec, person_id="p1", k=10)
    texts = [h.text for h in hits]
    assert texts.count("Yang prefers short answers") == 1
    all_vec = FakeEmbedder().embed(["Yang is planning a demo day"])[0]
    assert any(
        "demo day" in h.text
        for h in brain._store.search_memories(all_vec, person_id="p1", k=10)
    )


def test_history_trimmed(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    for i in range(30):
        brain.respond(f"question {i}")
    sent = client.chat.completions.last_kwargs["messages"]
    assert len(sent) <= 22  # system + trimmed history

import json

import pytest

from reachy_vec.brain.chat import ChatBrain, SpeechInterrupted, ThinkFilter
from reachy_vec.perception.fusion import TurnIdentity
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk
from tests.conftest import (
    FakeChoiceMessage,
    FakeEmbedder,
    FakeLLMClient,
    FakeToolCall,
)

YANG = TurnIdentity("p1", "Yang")


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


def make_brain(tmp_path, client, opener=None, web_search_fetch=None,
               look_around_fn=None, disable_thinking=False, hermes=None, **kw):
    return ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        opener=opener or (lambda url: None),
        web_search_fetch=web_search_fetch,
        look_around_fn=look_around_fn,
        disable_thinking=disable_thinking,
        hermes=hermes,
        **kw,
    )


def _http_error(code):
    import io
    import urllib.error

    return urllib.error.HTTPError(
        "https://api.tavily.com/search", code, "err", {}, io.BytesIO(b"")
    )


def test_look_around_tool_offered_only_when_fn_present(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = make_brain(tmp_path, FakeLLMClient(), look_around_fn=lambda: "a room")
    assert not any(t["function"]["name"] == "look_around" for t in off._active_tools())
    assert any(t["function"]["name"] == "look_around" for t in on._active_tools())


def test_look_around_tool_dispatches_to_fn(tmp_path):
    calls = []

    def fake_look_around():
        calls.append(True)
        return "a whiteboard on my left, two desks on my right"

    tool_call = FakeToolCall("look_around", json.dumps({}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Here's what I saw."),
        ]
    )
    brain = make_brain(tmp_path, client, look_around_fn=fake_look_around)
    reply = brain.respond("have a look around", identity=YANG)
    assert calls == [True]
    assert reply == "Here's what I saw."


def test_web_search_returns_answer(tmp_path):
    brain = make_brain(
        tmp_path, FakeLLMClient(), web_search_fetch=lambda q: "Paris is the capital."
    )
    assert brain._tool_web_search({"query": "capital of France"}) == "Paris is the capital."


def test_web_search_out_of_credits_tells_user(tmp_path):
    def boom(q):
        raise _http_error(432)

    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=boom)
    msg = brain._tool_web_search({"query": "latest news"}).lower()
    assert "credit" in msg or "allowance" in msg


def test_web_search_paygo_limit_also_out_of_credits(tmp_path):
    def boom(q):
        raise _http_error(433)

    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=boom)
    assert "allowance" in brain._tool_web_search({"query": "x"}).lower()


def test_web_search_rate_limited_is_distinct(tmp_path):
    def boom(q):
        raise _http_error(429)

    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=boom)
    msg = brain._tool_web_search({"query": "x"}).lower()
    assert "moment" in msg or "rate" in msg
    assert "allowance" not in msg  # not confused with credit exhaustion


def test_web_search_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=lambda q: "ok")
    assert not any(t["function"]["name"] == "web_search" for t in off._active_tools())
    assert any(t["function"]["name"] == "web_search" for t in on._active_tools())


def test_get_time_tool_returns_local_time(tmp_path):
    import re

    brain = make_brain(tmp_path, FakeLLMClient())
    result = brain._tool_get_time({})
    assert "the time is" in result.lower()
    assert re.search(r"\d{1,2}:\d{2}", result)  # HH:MM clock value


def test_interrupt_keeps_partial_reply_and_skips_tools(tmp_path):
    client = FakeLLMClient(reply="One sentence. Two sentence. Three sentence.")
    brain = make_brain(tmp_path, client)
    spoken = []

    def on_sentence(text):
        spoken.append(text)
        if len(spoken) == 1:  # interrupt right after the first sentence
            raise SpeechInterrupted()

    text = brain.respond("hi", on_sentence=on_sentence)
    assert spoken == ["One sentence."]           # only the first was spoken
    assert text.startswith("One sentence.")      # partial returned
    assert brain._history[-1]["content"].endswith(" -- (interrupted)")
    assert brain._exchanges == 1


def test_context_and_speaker_injected_each_turn(tmp_path):
    client = FakeLLMClient(reply="It's the Food Mapping demo.")
    brain = make_brain(tmp_path, client)
    reply = brain.respond("any demos about food?", identity=YANG)
    assert reply == "It's the Food Mapping demo."
    sent = client.chat.completions.last_kwargs["messages"]
    assert sent[0]["role"] == "system" and "Reachy" in sent[0]["content"]
    assert "food insecurity" in sent[-1]["content"]      # retrieved context
    assert "Yang: any demos about food?" in sent[-1]["content"]


def test_history_carries_across_turns_and_reset_clears(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    brain.respond("first question", identity=YANG)
    brain.respond("tell me more", identity=YANG)
    sent = client.chat.completions.last_kwargs["messages"]
    joined = json.dumps(sent)
    assert "first question" in joined and "tell me more" in joined
    brain.reset()
    brain.respond("fresh start", identity=YANG)
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
    reply = brain.respond("open the food mapping demo", identity=YANG)
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
    brain.respond("any coffee tips?", identity=YANG)
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
    assert brain.respond("remember I prefer short answers", identity=YANG) == "Noted!"
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
    brain.respond("we're planning a demo day", identity=YANG)
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
    from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow

    store.add_voice_rows(
        [
            VoiceRow(
                voice_id="p2:0",
                person_id="p2",
                name="Bob",
                vector=[0.5] * VOICE_EMBEDDING_DIM,
                created_at="2026-07-07T00:00:00+00:00",
                source="enrolled",
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
    reply = brain.respond("tell bob the meeting moved to 3", identity=YANG)
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
    brain.respond("tell carol hi", identity=YANG)
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
    reply = brain.respond("what's the weather like?", identity=YANG)
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
    brain._store_memories(["Yang prefers short answers"], person_id="p1")
    brain._store_memories(["Yang prefers short answers"], person_id="p1")  # dup -> skip
    brain._store_memories(["Yang is planning a demo day"], person_id="p1")  # stored
    vec = FakeEmbedder().embed(["Yang prefers short answers"])[0]
    hits = brain._store.search_memories(vec, person_id="p1", k=10)
    texts = [h.text for h in hits]
    assert texts.count("Yang prefers short answers") == 1
    all_vec = FakeEmbedder().embed(["Yang is planning a demo day"])[0]
    assert any(
        "demo day" in h.text
        for h in brain._store.search_memories(all_vec, person_id="p1", k=10)
    )


def test_streaming_emits_sentences_progressively(tmp_path):
    client = FakeLLMClient(reply="First sentence here. Second one arrives! And a third?")
    brain = make_brain(tmp_path, client)
    sentences: list[str] = []
    reply = brain.respond("chat to me", identity=YANG, on_sentence=sentences.append)
    assert sentences == [
        "First sentence here.",
        "Second one arrives!",
        "And a third?",
    ]
    assert reply == "First sentence here. Second one arrives! And a third?"
    assert client.chat.completions.last_kwargs["stream"] is True


def test_streaming_with_tool_call_speaks_only_final_answer(tmp_path):
    opened: list[str] = []
    tool_call = FakeToolCall(
        "open_url", json.dumps({"url": "https://example.org/food"})
    )
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Opening it now. Have a look!"),
        ]
    )
    brain = make_brain(tmp_path, client, opener=opened.append)
    sentences: list[str] = []
    brain.respond("open the food demo", on_sentence=sentences.append)
    assert opened == ["https://example.org/food"]
    assert sentences == ["Opening it now.", "Have a look!"]


def test_history_trimmed(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = make_brain(tmp_path, client)
    for i in range(30):
        brain.respond(f"question {i}")
    sent = client.chat.completions.last_kwargs["messages"]
    assert len(sent) <= 22  # system + trimmed history


def test_turn_identity_switches_attribution(tmp_path):
    """Bob chimes into Alice's conversation; his note goes to Bob."""
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(
                None, tool_calls=[FakeToolCall("save_note", '{"note": "Bob prefers tea"}')]
            ),
            FakeChoiceMessage("Noted, Bob!"),
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("remember I prefer tea", identity=TurnIdentity("p2", "Bob"))
    vec = FakeEmbedder().embed(["Bob prefers tea"])[0]
    assert any("tea" in h.text for h in brain._store.search_memories(vec, person_id="p2", k=3))
    assert brain._store.search_memories(vec, person_id="p1", k=3) == []


def test_anonymous_turn_refuses_notes_despite_owner(tmp_path):
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[FakeToolCall("save_note", '{"note": "x"}')]),
            FakeChoiceMessage("Sorry, I don't know who's asking."),
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("remember this", identity=None)  # anonymous despite owner
    vec = FakeEmbedder().embed(["x"])[0]
    assert brain._store.search_memories(vec, person_id="p1", k=3) == []
    tool_result = client.chat.completions.calls[1]["messages"][-1]
    assert "don't know who's speaking" in tool_result["content"]


def test_distillation_covers_every_speaker(tmp_path):
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage("hi alice"),
            FakeChoiceMessage("hi bob"),
            FakeChoiceMessage("- Alice likes charts"),  # summary for Alice
            FakeChoiceMessage("- Bob is visiting"),     # summary for Bob
        ]
    )
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("hello", identity=TurnIdentity("p1", "Alice"))
    brain.respond("hello from bob", identity=TurnIdentity("p2", "Bob"))
    brain.end_conversation()
    emb = FakeEmbedder()
    assert brain._store.search_memories(emb.embed(["charts"])[0], person_id="p1", k=3)
    assert brain._store.search_memories(emb.embed(["visiting"])[0], person_id="p2", k=3)


def test_reasoning_effort_sent_for_gpt5_models(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-5-mini",
        reasoning_effort="minimal",
        opener=lambda url: None,
    )
    brain.respond("hello", identity=YANG)
    assert client.chat.completions.last_kwargs["reasoning_effort"] == "minimal"


def test_reasoning_effort_omitted_for_non_gpt5_models(tmp_path):
    client = FakeLLMClient(reply="ok")
    brain = ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        reasoning_effort="minimal",
        opener=lambda url: None,
    )
    brain.respond("hello", identity=YANG)
    assert "reasoning_effort" not in client.chat.completions.last_kwargs


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.query_texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return super().embed_query(text)


def test_question_is_embedded_as_query(tmp_path):
    embedder = RecordingEmbedder()
    brain = ChatBrain(
        store=seeded_store(tmp_path),
        embedder=embedder,
        client=FakeLLMClient(reply="ok"),
        model="gpt-4o",
        opener=lambda url: None,
    )
    brain.respond("any demos about food?", identity=YANG)
    assert embedder.query_texts == ["any demos about food?"]


def test_keyword_only_match_ranks_first_in_context(tmp_path):
    # The question's fake vector is unrelated to both chunks; only BM25 on
    # "Jane Smith" can put the Robot Arm chunk first in the injected context.
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder()
    texts = {
        "arm": "Demo: Robot Arm Teleop\nAuthors: Jane Smith",
        "food": "Demo: Food Mapping\nAuthors: Bob",
    }
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id=key,
                text=text,
                vector=embedder.embed([text])[0],
                source=f"demo: {key}",
                ingested_at="2026-07-08T00:00:00+00:00",
            )
            for key, text in texts.items()
        ]
    )
    client = FakeLLMClient(reply="ok")
    brain = ChatBrain(
        store=store,
        embedder=embedder,
        client=client,
        model="gpt-4o",
        opener=lambda url: None,
    )
    brain.respond("who is Jane Smith?", identity=YANG)
    context = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert context.index("Robot Arm Teleop") < context.index("Food Mapping")


def test_look_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=lambda q: "a desk",
    )
    assert not any(t["function"]["name"] == "look" for t in off._active_tools())
    assert any(t["function"]["name"] == "look" for t in on._active_tools())
    assert "camera" in on._system_prompt().lower()


def test_look_tool_passes_question_and_returns_answer(tmp_path):
    seen = []
    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=lambda q: seen.append(q) or "two monitors",
    )
    assert brain._tool_look({"question": "what's on my desk?"}) == "two monitors"
    assert seen == ["what's on my desk?"]


def test_look_tool_failure_is_friendly(tmp_path):
    def boom(q):
        raise RuntimeError("cam died")

    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=boom,
    )
    assert "trouble" in brain._tool_look({"question": "x"}).lower()


def test_look_tool_handles_null_question(tmp_path):
    seen = []
    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        look_fn=lambda q: seen.append(q) or "a wall",
    )
    assert brain._tool_look({"question": None}) == "a wall"
    assert seen == [""]  # null coerced to empty string, passed through


def test_selfie_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=lambda: "took a photo and popped it up",
    )
    assert not any(t["function"]["name"] == "selfie" for t in off._active_tools())
    assert any(t["function"]["name"] == "selfie" for t in on._active_tools())


def test_selfie_tool_invokes_closure(tmp_path):
    calls = []
    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=lambda: calls.append(1) or "took a photo and popped it up",
    )
    assert brain._tool_selfie({}) == "took a photo and popped it up"
    assert calls == [1]


def test_selfie_tool_failure_is_friendly(tmp_path):
    def boom():
        raise RuntimeError("cam died")

    brain = ChatBrain(
        store=seeded_store(tmp_path), embedder=FakeEmbedder(),
        client=FakeLLMClient(), model="gpt-4o", opener=lambda url: None,
        selfie_fn=boom,
    )
    assert "couldn't take" in brain._tool_selfie({}).lower()


# -- open_url only on an explicit request ---------------------------------------


def _open_url_brain(tmp_path, opened):
    tool_call = FakeToolCall("open_url", json.dumps({"url": "https://example.org/food"}))
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Food Mapping maps food deserts."),
        ]
    )
    return make_brain(tmp_path, client, opener=opened.append), client


def test_open_url_not_executed_without_explicit_request(tmp_path):
    opened: list[str] = []
    brain, client = _open_url_brain(tmp_path, opened)
    reply = brain.respond("any demos about food?", identity=YANG)
    assert opened == []                                # model tried; guard refused
    assert reply == "Food Mapping maps food deserts."  # still answers
    tool_results = [m for m in client.chat.completions.calls[1]["messages"]
                    if m.get("role") == "tool"]
    assert "not requested" in tool_results[0]["content"]


@pytest.mark.parametrize(
    "question",
    ["can you show me the food demo", "pull up food mapping", "Open it please",
     "bring up the food one", "launch the food mapping demo"],
)
def test_open_url_executed_on_explicit_request(tmp_path, question):
    opened: list[str] = []
    brain, _ = _open_url_brain(tmp_path, opened)
    brain.respond(question, identity=YANG)
    assert opened == ["https://example.org/food"]


def test_open_url_trigger_needs_whole_word(tmp_path):
    opened: list[str] = []
    brain, _ = _open_url_brain(tmp_path, opened)
    brain.respond("is the lab openness policy on the wiki?", identity=YANG)
    assert opened == []


def test_system_prompt_does_not_push_demos():
    from reachy_vec.brain.chat import PERSONALITY, TOOLS

    assert "inside out" not in PERSONALITY
    open_url = next(t for t in TOOLS if t["function"]["name"] == "open_url")
    assert "explicitly" in open_url["function"]["description"]


def test_disable_thinking_reaches_every_create_call(tmp_path):
    tool_call = FakeToolCall(
        "open_url", json.dumps({"url": "https://example.org/food", "title": "Food"})
    )
    client = FakeLLMClient(
        messages=[
            FakeChoiceMessage(None, tool_calls=[tool_call]),
            FakeChoiceMessage("Opening it."),
            FakeChoiceMessage("[]"),
        ]
    )
    brain = make_brain(tmp_path, client, disable_thinking=True)
    brain.begin_conversation(YANG.person_id, YANG.name)
    brain.respond("open the food demo", identity=YANG, on_sentence=lambda s: None)
    brain.end_conversation()
    calls = client.chat.completions.calls
    assert len(calls) == 3  # tool-call turn, follow-up, summary
    for call in calls:
        assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_no_extra_body_by_default(tmp_path):
    client = FakeLLMClient(reply="hi")
    brain = make_brain(tmp_path, client)
    brain.respond("hello")
    assert "extra_body" not in client.chat.completions.last_kwargs


def test_think_filter_drops_reasoning():
    f = ThinkFilter()
    assert f.feed("<think>plan</think>Hello.") + f.flush() == "Hello."


def test_think_filter_tag_split_across_chunks():
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in ["<thi", "nk>secret</th", "ink>Hi there."]) + f.flush()
    assert out == "Hi there."


def test_think_filter_unclosed_think_emits_nothing():
    f = ThinkFilter()
    assert f.feed("<think>still reasoning") + f.flush() == ""


def test_think_filter_passes_plain_text_and_lone_angle():
    f = ThinkFilter()
    assert f.feed("a < b, ok.") + f.flush() == "a < b, ok."


def test_streamed_reasoning_is_never_spoken(tmp_path):
    client = FakeLLMClient(reply="<think>I should greet.</think>Hello Yang. How can I help?")
    sentences = []
    reply = make_brain(tmp_path, client).respond("hi", identity=YANG, on_sentence=sentences.append)
    assert sentences == ["Hello Yang.", "How can I help?"]
    assert "think" not in reply


def test_reasoning_only_reply_speaks_nothing(tmp_path):
    sentences = []
    reply = make_brain(tmp_path, FakeLLMClient(reply="<think>hmm</think>")).respond(
        "hi", on_sentence=sentences.append
    )
    assert sentences == [] and reply == ""


def test_non_streaming_reasoning_is_stripped(tmp_path):
    client = FakeLLMClient(reply="<think>plan</think>Hello Yang.")
    reply = make_brain(tmp_path, client).respond("hi", identity=YANG)
    assert reply == "Hello Yang."


def test_warm_up_sends_real_system_prompt_and_tools(tmp_path):
    client = FakeLLMClient(reply="x")
    brain = make_brain(tmp_path, client)
    assert brain.warm_up() is True
    kw = client.chat.completions.last_kwargs
    assert kw["messages"][0]["content"] == brain._system_prompt()
    assert kw["tools"] == brain._active_tools() and kw["max_completion_tokens"] == 1


def test_warm_up_failure_is_non_fatal(tmp_path):
    class _Completions:
        def create(self, **kwargs):
            raise ConnectionError("down")

    class _Chat:
        completions = _Completions()

    class Down:
        chat = _Chat()

    assert make_brain(tmp_path, Down()).warm_up() is False


ALICE, BOB = TurnIdentity("p1", "Alice"), TurnIdentity("p2", "Bob")


def _last_user(client):
    return client.chat.completions.last_kwargs["messages"][-1]["content"]


def test_no_change_note_on_first_identified_turn(tmp_path):
    client = FakeLLMClient()
    brain = make_brain(tmp_path, client)
    brain.respond("q1", identity=ALICE)
    assert "Speaker changed" not in _last_user(client)


def test_change_note_when_recognised_speaker_changes(tmp_path):
    client = FakeLLMClient()
    brain = make_brain(tmp_path, client)
    brain.respond("q1", identity=ALICE)
    brain.respond("q2", identity=BOB)
    assert _last_user(client).startswith("(Speaker changed: now Bob.)\n")


def test_anonymous_turn_is_not_a_change_and_has_no_name(tmp_path):
    client = FakeLLMClient()
    brain = make_brain(tmp_path, client)
    brain.respond("q1", identity=ALICE)
    brain.respond("q2", identity=None)
    assert "Speaker changed" not in _last_user(client)
    assert "Alice" not in _last_user(client)
    brain.respond("q3", identity=ALICE)
    assert "Speaker changed" not in _last_user(client)


def test_change_across_anonymous_turn(tmp_path):
    client = FakeLLMClient()
    brain = make_brain(tmp_path, client)
    brain.respond("q1", identity=ALICE)
    brain.respond("q2", identity=None)
    brain.respond("q3", identity=BOB)
    assert _last_user(client).startswith("(Speaker changed: now Bob.)\n")


def test_reset_forgets_last_speaker(tmp_path):
    client = FakeLLMClient()
    brain = make_brain(tmp_path, client)
    brain.respond("q1", identity=ALICE)
    brain.reset()
    brain.respond("q2", identity=BOB)
    assert "Speaker changed" not in _last_user(client)


def test_personality_has_name_rule():
    from reachy_vec.brain.chat import PERSONALITY

    assert "use their first name occasionally" in PERSONALITY


def _hermes_brain(tmp_path, agent, tool_args=None):
    from reachy_vec.brain.hermes import HermesJobs

    store = seeded_store(tmp_path)
    jobs = HermesJobs(agent, store, wait_s=1.0, task_timeout_s=5)
    call = FakeToolCall("ask_hermes", json.dumps(tool_args or {"task": "research X"}))
    client = FakeLLMClient(
        messages=[FakeChoiceMessage(None, tool_calls=[call]), FakeChoiceMessage("Done: X shipped.")]
    )
    brain = ChatBrain(
        store=store, embedder=FakeEmbedder(), client=client, model="gpt-4o", hermes=jobs
    )
    return brain, client


def test_ask_hermes_absent_by_default(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient())
    assert not any(t["function"]["name"] == "ask_hermes" for t in brain._active_tools())
    assert "ask_hermes" not in brain._system_prompt()


def test_ask_hermes_offered_when_configured(tmp_path):
    from tests.conftest import FakeAgent

    brain, _ = _hermes_brain(tmp_path, FakeAgent(["ok"]))
    assert any(t["function"]["name"] == "ask_hermes" for t in brain._active_tools())
    assert "ask_hermes" in brain._system_prompt()


def test_ask_hermes_passes_identity_and_speaks_holding_line(tmp_path):
    from tests.conftest import FakeAgent

    agent = FakeAgent(["X shipped."])
    brain, _ = _hermes_brain(tmp_path, agent)
    sentences = []
    brain.respond("research X", identity=YANG, on_sentence=sentences.append)
    assert sentences[0] == "Let me look into that."
    assert agent.calls == [("research X", "Yang")]


def test_ask_hermes_anonymous_passes_none(tmp_path):
    from tests.conftest import FakeAgent

    agent = FakeAgent(["X shipped."])
    brain, _ = _hermes_brain(tmp_path, agent)
    brain.respond("research X", on_sentence=lambda s: None)
    assert agent.calls[0][1] is None


def test_ask_hermes_empty_task(tmp_path):
    from tests.conftest import FakeAgent

    agent = FakeAgent([])
    brain, _ = _hermes_brain(tmp_path, agent, tool_args={"task": "  "})
    brain.respond("hmm", identity=YANG, on_sentence=lambda s: None)
    assert agent.calls == []
    tool_msgs = [m for m in brain._history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "can't ask: empty task"


def test_barge_in_on_holding_line_still_waits(tmp_path):
    from tests.conftest import FakeAgent

    agent = FakeAgent(["X shipped."])
    brain, _ = _hermes_brain(tmp_path, agent)

    def interrupt(sentence):
        if sentence == "Let me look into that.":
            raise SpeechInterrupted

    brain.respond("research X", identity=YANG, on_sentence=interrupt)
    assert agent.calls == [("research X", "Yang")]
    tool_msgs = [m for m in brain._history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "X shipped."


def test_ask_hermes_busy_skips_holding_line(tmp_path):
    from reachy_vec.brain.hermes import BUSY
    from tests.conftest import FakeAgent

    agent = FakeAgent([])
    brain, _ = _hermes_brain(tmp_path, agent)
    brain._hermes._future = type("F", (), {"done": lambda self: False})()
    sentences = []
    brain.respond("research X", identity=YANG, on_sentence=sentences.append)
    assert "Let me look into that." not in sentences
    assert agent.calls == []
    tool_msgs = [m for m in brain._history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == BUSY


# -- retrieval relevance gate ---------------------------------------------------


def _scored_brain(tmp_path, client, scores, **kw):
    from reachy_vec.store.schemas import DocChunk

    brain = make_brain(tmp_path, client, **kw)
    hits = [
        (DocChunk(chunk_id=f"c{i}", text=f"snippet {i}", vector=[0.0] * 384,
                  source=f"demo: D{i}", ingested_at="2026-10-20T00:00:00+00:00"), s)
        for i, s in enumerate(scores)
    ]
    brain._store.search_docs_scored = lambda v, k, query_text=None: hits[:k]
    return brain


def test_low_scoring_docs_are_not_sent(tmp_path):
    client = FakeLLMClient(reply="I'm well.")
    brain = _scored_brain(tmp_path, client, [0.61, 0.55, 0.50], rag_min_score=0.62)
    brain.respond("how are you today", identity=YANG)
    user = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert "snippet" not in user and "(nothing relevant found)" in user


def test_relevant_docs_are_sent_and_capped(tmp_path):
    client = FakeLLMClient(reply="Food Mapping.")
    brain = _scored_brain(tmp_path, client, [0.78, 0.66, 0.64, 0.63, 0.40],
                          rag_min_score=0.62, k=3)
    brain.respond("tell me about the food mapping demo", identity=YANG)
    user = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert "snippet 0" in user and "snippet 2" in user
    assert "snippet 3" not in user and "snippet 4" not in user   # k=3 cap + gate


def test_rag_settings_defaults():
    from reachy_vec.config import Settings

    assert Settings.model_fields["rag_min_score"].default == 0.62
    assert Settings.model_fields["rag_k"].default == 3


def test_list_demos_tool_returns_newest_first(tmp_path):
    tool_call = FakeToolCall("list_demos", json.dumps({"limit": 2}))
    client = FakeLLMClient(messages=[FakeChoiceMessage(None, tool_calls=[tool_call]),
                                     FakeChoiceMessage("The newest is New.")])
    brain = make_brain(tmp_path, client)
    brain._store.demo_catalog = lambda: [("New", "2026-10-13"), ("Mid", "2024-01-01"),
                                         ("Old", "2022-05-19")]
    brain.respond("what's the latest demo?", identity=YANG)
    result = next(m["content"] for m in client.chat.completions.calls[1]["messages"]
                  if m.get("role") == "tool")
    assert result == "newest first: New (added 2026-10-13); Mid (added 2024-01-01)"


def test_list_demos_tool_defaults_and_caps_limit(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient(reply="x"))
    brain._store.demo_catalog = lambda: [(f"D{i}", f"2026-01-{i + 1:02d}") for i in range(20)]
    assert brain._tool_list_demos({}).count("added") == 5          # default
    assert brain._tool_list_demos({"limit": 50}).count("added") == 10  # cap
    brain._store.demo_catalog = lambda: [("Undated", None)]
    assert brain._tool_list_demos({}) == "newest first: Undated (date unknown)"
    brain._store.demo_catalog = lambda: []
    assert brain._tool_list_demos({}) == "no demos in the knowledge base"


def test_list_demos_tool_is_offered():
    from reachy_vec.brain.chat import TOOLS

    assert "list_demos" in [t["function"]["name"] for t in TOOLS]


def test_trimmed_persona_keeps_the_rules():
    from reachy_vec.brain.chat import PERSONALITY

    assert len(PERSONALITY) < 2000  # was 3195; every char is re-read each turn
    for phrase in ("upbeat", "emoji", "spoken aloud", "leading with a brief one", "Never invent",
                   "off the top of my head",
                   "offer to find out", "use their first name occasionally",
                   "explicitly asks to open"):
        assert phrase in PERSONALITY


def test_markdown_emphasis_is_not_spoken(tmp_path):
    client = FakeLLMClient(reply="The latest is **Metaliverpool**. It's __new__.")
    sentences = []
    make_brain(tmp_path, client).respond("latest?", on_sentence=sentences.append)
    assert sentences == ["The latest is Metaliverpool.", "It's new."]


def test_persona_says_when_to_use_tools():
    from reachy_vec.brain.chat import PERSONALITY

    for tool in ("get_time", "save_note", "send_message", "list_demos", "open_url"):
        assert tool in PERSONALITY


class _StatusHermes:
    """HermesJobs stand-in exposing only what ChatBrain reads per turn."""

    def __init__(self, status):
        self._status = status
        self.busy = status is not None

    def status(self):
        return self._status

    def ask(self, task, *, requester_id, requester_name, on_progress=None):
        return "ok"


def test_turn_context_says_what_hermes_is_working_on(tmp_path):
    client = FakeLLMClient(reply="Still on it.")
    brain = make_brain(tmp_path, client, hermes=_StatusHermes(("research X", 45.2)))
    brain.respond("is Hermes done yet?", identity=YANG)
    user = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert '[Hermes is working on: "research X" - 45 s so far]' in user


def test_turn_context_says_hermes_is_idle(tmp_path):
    client = FakeLLMClient(reply="Nothing running.")
    brain = make_brain(tmp_path, client, hermes=_StatusHermes(None))
    brain.respond("is Hermes doing anything?", identity=YANG)
    user = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert "[Hermes is idle - no task running]" in user


def test_no_hermes_line_without_hermes(tmp_path):
    client = FakeLLMClient(reply="hi")
    make_brain(tmp_path, client).respond("hello", identity=YANG)
    assert "Hermes" not in client.chat.completions.last_kwargs["messages"][-1]["content"]


def test_hermes_hint_says_it_is_an_agent_not_a_teammate():
    from reachy_vec.brain.chat import ASK_HERMES_HINT

    assert "not a person" in ASK_HERMES_HINT and "send_message" in ASK_HERMES_HINT


class _RecordingHermes(_StatusHermes):
    def __init__(self):
        super().__init__(None)
        self.asked = []

    def ask(self, task, *, requester_id, requester_name, on_progress=None):
        self.asked.append((task, requester_name))
        return "Canberra."


def test_hermes_hint_covers_current_information():
    from reachy_vec.brain.chat import ASK_HERMES_HINT

    assert "news" in ASK_HERMES_HINT and "current" in ASK_HERMES_HINT


def test_fix_body_tool_offered_only_with_a_body(tmp_path):
    def names(client):
        return [t["function"]["name"] for t in client.chat.completions.last_kwargs["tools"]]

    c1 = FakeLLMClient(reply="ok")
    make_brain(tmp_path, c1).respond("hi")
    c2 = FakeLLMClient(reply="ok")
    make_brain(tmp_path, c2, fix_body_fn=lambda: "x").respond("hi")
    assert "fix_body" not in names(c1) and "fix_body" in names(c2)


@pytest.mark.parametrize(("llm_says", "expected"), [
    ("Yang", "Yang"), ("  yang. ", "Yang"), ("NONE", None), ("hannah smith", "Hannah Smith"),
    ("<think>hm</think>Yang", "Yang"),
    ("I think the name is probably Yang but not sure", None),   # rambling -> rejected
])
def test_extract_name_uses_the_llm(tmp_path, llm_says, expected):
    client = FakeLLMClient(reply=llm_says)
    brain = make_brain(tmp_path, client)
    assert brain.extract_name("whatever was heard") == expected
    sent = client.chat.completions.last_kwargs["messages"]
    assert "whatever was heard" in sent[-1]["content"]


def test_extract_name_falls_back_to_rules_when_llm_fails(tmp_path):
    class Down:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise ConnectionError("oMLX down")

    brain = make_brain(tmp_path, Down())
    assert brain.extract_name("My name is Yang.") == "Yang"
    assert brain.extract_name("Y, A, N, G.") == "Yang"


# -- turn router: what Reachy does with each route ------------------------------
# (understanding phrasings is the Router's job - see scripts/route_eval.py)


def _routed_brain(tmp_path, client, route, **kw):
    from tests.conftest import FakeRouter

    return make_brain(tmp_path, client, router=FakeRouter({"q": route}), **kw)


def test_hermes_route_sends_the_task_and_cancels_the_normal_answer(tmp_path):
    from reachy_vec.brain.router import Route

    client = FakeLLMClient(reply="Hermes says it is Canberra.")
    hermes = _RecordingHermes()
    sentences = []
    reply = _routed_brain(tmp_path, client, Route("hermes", "find the capital"),
                          hermes=hermes).respond("q", identity=YANG,
                                                 on_sentence=sentences.append)
    assert hermes.asked == [("find the capital", "Yang")]
    assert sentences == ["Let me look into that.", "Hermes says it is Canberra."]
    assert reply == "Hermes says it is Canberra."


def test_fix_body_route_repairs_and_reports(tmp_path):
    fixed = []
    client = FakeLLMClient(reply="Done - I'm up!")
    _routed_brain(tmp_path, client, "fix_body",
                  fix_body_fn=lambda: fixed.append(1) or "motors were off - fixed"
                  ).respond("q", identity=YANG, on_sentence=lambda s: None)
    assert fixed == [1]
    tool = [m for m in client.chat.completions.last_kwargs["messages"] if m.get("role") == "tool"]
    assert tool[0]["content"] == "motors were off - fixed"


def test_latest_demos_route_supplies_the_date_sorted_catalogue(tmp_path):
    client = FakeLLMClient(reply="Metaliverpool - Site Assessment.")
    brain = _routed_brain(tmp_path, client, "latest_demos")
    brain._store.demo_catalog = lambda: [("Metaliverpool - Site Assessment", "2026-10-13"),
                                         ("Wundrr Agentic AI", "2026-10-06")]
    brain.respond("q", identity=YANG, on_sentence=lambda s: None)
    user = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert user.startswith("[Demos, newest first]")
    assert user.index("Metaliverpool") < user.index("Wundrr")


def test_remember_me_route_steps_aside_for_the_oracle(tmp_path):
    client = FakeLLMClient(reply="I'll remember you!")       # must never be spoken
    sentences = []
    brain = _routed_brain(tmp_path, client, "remember_me")
    assert brain.respond("q", identity=YANG, on_sentence=sentences.append) == ""
    assert sentences == [] and brain.last_route.name == "remember_me"


def test_chat_route_answers_normally(tmp_path):
    client = FakeLLMClient(reply="Hello Yang.")
    sentences = []
    brain = _routed_brain(tmp_path, client, "chat", hermes=_RecordingHermes())
    brain.respond("q", identity=YANG, on_sentence=sentences.append)
    assert sentences == ["Hello Yang."]


def test_route_without_the_capability_falls_back_to_a_normal_answer(tmp_path):
    client = FakeLLMClient(reply="I can't reach Hermes, but here's what I know.")
    sentences = []
    _routed_brain(tmp_path, client, "hermes").respond(      # no hermes configured
        "q", identity=YANG, on_sentence=sentences.append)
    assert sentences == ["I can't reach Hermes, but here's what I know."]


def test_no_router_means_no_routing(tmp_path):
    client = FakeLLMClient(reply="ok")
    hermes = _RecordingHermes()
    make_brain(tmp_path, client, hermes=hermes).respond("ask Hermes to research X",
                                                        identity=YANG)
    assert hermes.asked == []



def test_condense_for_speech_uses_the_llm(tmp_path):
    client = FakeLLMClient(reply="**Two** short sentences. Done!")
    out = make_brain(tmp_path, client).condense_for_speech("- long item.\n- markdown.\n" * 20)
    assert out == "Two short sentences. Done!"
    assert "spoken" in client.chat.completions.last_kwargs["messages"][0]["content"]


def test_condense_for_speech_short_text_skips_the_llm(tmp_path):
    client = FakeLLMClient(reply="x")
    assert make_brain(tmp_path, client).condense_for_speech("Meeting moved to 3.") == \
        "Meeting moved to 3."
    assert client.chat.completions.calls == []


def test_condense_for_speech_falls_back_to_first_sentences(tmp_path):
    class Down:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise ConnectionError("down")

    text = "**One.** Two is here. Three is long. " + "More. " * 50
    out = make_brain(tmp_path, Down()).condense_for_speech(text, max_sentences=2)
    assert out == "One. Two is here." and "**" not in out


def test_open_url_guard_note_does_not_invite_an_explanation(tmp_path):
    tool_call = FakeToolCall("open_url", json.dumps({"url": "https://example.org/x"}))
    client = FakeLLMClient(messages=[FakeChoiceMessage(None, tool_calls=[tool_call]),
                                     FakeChoiceMessage("It maps food access.")])
    make_brain(tmp_path, client).respond("tell me about the food demo", identity=YANG)
    note = [m for m in client.chat.completions.calls[1]["messages"] if m.get("role") == "tool"]
    assert "do not mention" in note[0]["content"].lower()


def test_condense_respects_the_sentence_budget(tmp_path):
    client = FakeLLMClient(reply="One. Two. Three.")
    make_brain(tmp_path, client).condense_for_speech("x. " * 200, max_sentences=4)
    assert "4 short sentences" in client.chat.completions.last_kwargs["messages"][0]["content"]


def test_add_context_lets_follow_ups_see_the_full_message(tmp_path):
    client = FakeLLMClient(reply="David builds the demos.")
    brain = make_brain(tmp_path, client)
    brain.add_context("FULL: David Boardman built the AI news feed.", "Short: team update.")
    brain.respond("tell me more about David", identity=YANG)
    sent = json.dumps(client.chat.completions.last_kwargs["messages"])
    assert "David Boardman built the AI news feed" in sent


def test_persona_allows_an_emoji_as_a_gesture():
    from reachy_vec.brain.chat import PERSONALITY

    assert "emoji" in PERSONALITY


def test_hermes_wait_speaks_progress_updates(tmp_path):
    from reachy_vec.brain.chat import HERMES_PROGRESS_LINES

    class SlowHermes(_RecordingHermes):
        def ask(self, task, *, requester_id, requester_name, on_progress=None):
            on_progress()
            on_progress()
            return "Found it."

    tool_call = FakeToolCall("ask_hermes", json.dumps({"task": "research X"}))
    client = FakeLLMClient(messages=[FakeChoiceMessage(None, tool_calls=[tool_call]),
                                     FakeChoiceMessage("Here it is.")])
    sentences = []
    make_brain(tmp_path, client, hermes=SlowHermes()).respond(
        "research X", identity=YANG, on_sentence=sentences.append)
    assert sentences[0] == "Let me look into that."
    assert sum(s in HERMES_PROGRESS_LINES for s in sentences) == 2
    assert sentences[-1] == "Here it is."


def test_interrupting_a_progress_update_stops_waiting(tmp_path):
    from reachy_vec.brain.chat import SpeechInterrupted

    results = []

    class SlowHermes(_RecordingHermes):
        def ask(self, task, *, requester_id, requester_name, on_progress=None):
            results.append(on_progress())        # the spoken update gets interrupted
            return "still working; tell the user you'll let them know when it's ready"

    def on_sentence(s):
        if s != "Let me look into that.":
            raise SpeechInterrupted()

    tool_call = FakeToolCall("ask_hermes", json.dumps({"task": "research X"}))
    client = FakeLLMClient(messages=[FakeChoiceMessage(None, tool_calls=[tool_call]),
                                     FakeChoiceMessage("ok")])
    make_brain(tmp_path, client, hermes=SlowHermes()).respond(
        "research X", identity=YANG, on_sentence=on_sentence)
    assert results == [False]


# -- opening pages: the open route and searches -----------------------------------


def test_open_route_opens_the_page_without_the_model_choosing_the_tool(tmp_path):
    from reachy_vec.brain.router import Route
    from tests.conftest import FakeRouter

    opened = []
    # the chat model would only *claim* it opened YouTube - the route acts instead
    client = FakeLLMClient(reply="Opened YouTube for you.")
    brain = make_brain(tmp_path, client, opener=opened.append, router=FakeRouter(
        {"can you open YouTube": Route("open", "https://www.youtube.com")}))
    sentences = []
    reply = brain.respond("can you open YouTube", identity=YANG, on_sentence=sentences.append)
    assert opened == ["https://www.youtube.com"]
    assert reply == "Opened YouTube for you."
    tool = [m for m in client.chat.completions.last_kwargs["messages"] if m.get("role") == "tool"]
    assert tool[0]["content"] == "opened https://www.youtube.com in the browser"


def test_open_route_still_needs_an_explicit_ask(tmp_path):
    from reachy_vec.brain.router import Route
    from tests.conftest import FakeRouter

    opened = []
    client = FakeLLMClient(reply="It's a mapping demo.")
    brain = make_brain(tmp_path, client, opener=opened.append, router=FakeRouter(
        {"tell me about food mapping": Route("open", "https://example.org/food")}))
    brain.respond("tell me about food mapping", identity=YANG, on_sentence=lambda s: None)
    assert opened == []


@pytest.mark.parametrize(("target", "url"), [
    ("https://example.org/x", "https://example.org/x"),
    ("youtube.com", "https://youtube.com"),
    ("www.bbc.co.uk/news", "https://www.bbc.co.uk/news"),
    ("lofi beats", "https://www.google.com/search?q=lofi+beats"),
])
def test_open_url_accepts_sites_and_searches(tmp_path, target, url):
    opened = []
    client = FakeLLMClient(messages=[
        FakeChoiceMessage(None, tool_calls=[FakeToolCall("open_url", json.dumps({"url": target}))]),
        FakeChoiceMessage("Done."),
    ])
    make_brain(tmp_path, client, opener=opened.append).respond("open it please")
    assert opened == [url]


def test_play_counts_as_an_explicit_ask(tmp_path):
    opened = []
    target = "https://www.youtube.com/results?search_query=jazz"
    client = FakeLLMClient(messages=[
        FakeChoiceMessage(None, tool_calls=[FakeToolCall("open_url", json.dumps({"url": target}))]),
        FakeChoiceMessage("Here's some jazz."),
    ])
    make_brain(tmp_path, client, opener=opened.append).respond("play some jazz")
    assert opened == [target]


def test_a_rerouted_answer_closes_its_stream(tmp_path):
    # an abandoned stream is closed later by the GC - inside httpcore's pool
    # lock on the next request, which deadlocked the whole loop
    from reachy_vec.brain.router import Route
    from tests.conftest import FakeRouter

    client = FakeLLMClient(reply="Sure, opening it now! Enjoy the tunes.")
    brain = make_brain(tmp_path, client, opener=lambda url: None, router=FakeRouter(
        {"play some jazz": Route("open", "https://www.youtube.com/results?search_query=jazz")}))
    brain.respond("play some jazz", identity=YANG, on_sentence=lambda s: None)
    assert all(stream.closed for stream in client.chat.completions.streams[:-1])
    assert client.chat.completions.streams[0].closed

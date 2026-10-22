import pytest

from reachy_vec.brain.router import CHAT, Route, Router, parse_route
from tests.conftest import FakeLLMClient


@pytest.mark.parametrize(("raw", "expected"), [
    ('{"route": "hermes", "task": "find AI news"}', Route("hermes", "find AI news")),
    ('Sure! {"route": "fix_body", "task": "x"}', Route("fix_body", "")),   # task only for hermes
    ('<think>hmm</think>{"route": "latest_demos"}', Route("latest_demos", "")),
    ('{"route": "REMEMBER_ME", "task": ""}', Route("remember_me", "")),
    ('{"route": "dance", "task": ""}', CHAT),          # unknown route
    ("hermes: find AI news", Route("hermes", "find AI news")),
    ("chat", CHAT), ("fix_body", Route("fix_body", "")), ("latest_demos.", Route("latest_demos")),
    ("I think it's chat", CHAT),                       # rambling
    ('{"route": ', CHAT),                              # truncated
    ("", CHAT),
])
def test_parse_route(raw, expected):
    assert parse_route(raw) == expected


def test_router_sends_the_utterance_with_thinking_off():
    client = FakeLLMClient(reply='{"route": "hermes", "task": "news"}')
    kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    route = Router(client, "M", llm_kwargs=kwargs).route("what's the latest AI news?")
    assert route == Route("hermes", "news")
    sent = client.chat.completions.last_kwargs
    assert sent["messages"][-1]["content"] == "what's the latest AI news?"
    assert sent["extra_body"] == kwargs["extra_body"] and "tools" not in sent


def test_router_failure_means_chat():
    class Down:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise ConnectionError("oMLX down")

    assert Router(Down(), "M").route("anything") == CHAT


def test_blank_text_skips_the_call():
    client = FakeLLMClient(reply='{"route": "hermes"}')
    assert Router(client, "M").route("  ") == CHAT
    assert client.chat.completions.calls == []

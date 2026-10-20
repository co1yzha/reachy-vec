from reachy_vec.brain.llm import make_llm_client, uses_local_llm
from reachy_vec.config import Settings


def test_local_client_gets_base_url_key_timeout():
    s = Settings(llm_base_url="http://127.0.0.1:8001/v1", llm_api_key="k", llm_timeout_s=12)
    seen = {}
    make_llm_client(s, openai_cls=lambda **kw: seen.update(kw) or "client")
    assert seen == {"base_url": "http://127.0.0.1:8001/v1", "api_key": "k", "timeout": 12}


def test_cloud_client_when_no_base_url():
    seen = {}
    make_llm_client(Settings(llm_base_url=None), openai_cls=lambda **kw: seen.update(kw))
    assert seen == {}


def test_local_without_key_sends_placeholder():  # openai SDK requires a non-empty key
    seen = {}
    make_llm_client(
        Settings(llm_base_url="http://x/v1", llm_api_key=None),
        openai_cls=lambda **kw: seen.update(kw),
    )
    assert seen["api_key"] == "local"


def test_uses_local_llm():
    assert uses_local_llm(Settings(llm_base_url="http://x/v1"))
    assert not uses_local_llm(Settings(llm_base_url=None))

"""LLM client factory: OpenAI cloud by default, or a local OpenAI-compatible server (oMLX)."""

from reachy_vec.config import Settings
from reachy_vec.config import settings as _settings


def uses_local_llm(settings: Settings = _settings) -> bool:
    return bool(settings.llm_base_url)


def make_llm_client(settings: Settings = _settings, openai_cls=None):
    if openai_cls is None:
        from openai import OpenAI as openai_cls
    if not uses_local_llm(settings):
        return openai_cls()
    return openai_cls(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "local",  # the SDK requires a non-empty key
        timeout=settings.llm_timeout_s,
    )

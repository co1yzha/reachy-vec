from reachy_vec.brain.rag import Answer, answer
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import FakeEmbedder, FakeLLMClient


def seeded_store(tmp_path) -> Store:
    store = Store(tmp_path / "lancedb")
    embedder = FakeEmbedder()
    texts = {"c1": "the pipeline runs nightly", "c2": "lunch is at noon"}
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id=cid,
                text=text,
                vector=embedder.embed([text])[0],
                source=f"{cid}.md",
                ingested_at="2026-07-06T00:00:00+00:00",
            )
            for cid, text in texts.items()
        ]
    )
    return store


def test_answer_returns_llm_text_and_sources(tmp_path):
    store = seeded_store(tmp_path)
    client = FakeLLMClient(reply="It runs nightly.")
    # FakeEmbedder is hash-based (no semantics), so query with the exact chunk
    # text to make c1 the guaranteed nearest neighbour.
    result = answer(
        "the pipeline runs nightly",
        store=store,
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        k=1,
    )
    assert isinstance(result, Answer)
    assert result.text == "It runs nightly."
    assert result.sources == ["c1.md"]


def test_answer_puts_retrieved_context_and_question_in_prompt(tmp_path):
    store = seeded_store(tmp_path)
    client = FakeLLMClient()
    answer(
        "when does the pipeline run?",
        store=store,
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        k=5,  # retrieve everything - hash embeddings carry no semantic ranking
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    user_message = kwargs["messages"][-1]["content"]
    assert "the pipeline runs nightly" in user_message
    assert "lunch is at noon" in user_message
    assert "when does the pipeline run?" in user_message


def test_answer_with_empty_store_says_no_context(tmp_path):
    store = Store(tmp_path / "lancedb")
    result = answer(
        "anything?",
        store=store,
        embedder=FakeEmbedder(),
        client=FakeLLMClient(),
        model="gpt-4o",
    )
    assert result.sources == []
    assert "knowledge base is empty" in result.text

from reachy_vec.brain.loop import chat_loop
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import DocChunk

from tests.conftest import FakeEmbedder, FakeLLMClient


def make_store(tmp_path) -> Store:
    store = Store(tmp_path / "lancedb")
    text = "the pipeline runs nightly"
    store.add_doc_chunks(
        [
            DocChunk(
                chunk_id="c1",
                text=text,
                vector=FakeEmbedder().embed([text])[0],
                source="notes.md",
                ingested_at="2026-07-06T00:00:00+00:00",
            )
        ]
    )
    return store


def run_loop(store, inputs: list[str]) -> list[str]:
    inputs_iter = iter(inputs)
    printed: list[str] = []
    chat_loop(
        store=store,
        embedder=FakeEmbedder(),
        client=FakeLLMClient(reply="Nightly."),
        model="gpt-4o",
        input_fn=lambda prompt="": next(inputs_iter),
        print_fn=printed.append,
    )
    return printed


def test_loop_answers_then_exits(tmp_path):
    printed = run_loop(make_store(tmp_path), ["when does it run?", "exit"])
    joined = "\n".join(printed)
    assert "Nightly." in joined
    assert "notes.md" in joined


def test_loop_exits_on_eof(tmp_path):
    inputs_iter = iter(["one question"])

    def input_fn(prompt=""):
        try:
            return next(inputs_iter)
        except StopIteration:
            raise EOFError

    printed: list[str] = []
    chat_loop(
        store=make_store(tmp_path),
        embedder=FakeEmbedder(),
        client=FakeLLMClient(),
        model="gpt-4o",
        input_fn=input_fn,
        print_fn=printed.append,
    )  # must return instead of raising
    assert printed  # answered the one question before EOF


def test_loop_skips_blank_lines(tmp_path):
    printed = run_loop(make_store(tmp_path), ["", "  ", "quit"])
    assert all("canned answer" not in line for line in printed)

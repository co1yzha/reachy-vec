# Phase 0c: RAG Chat Loop + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reachy-vec chat` answers questions about ingested team docs in a terminal REPL via OpenAI RAG — the Phase 0 milestone.

**Architecture:** `brain/rag.py` embeds a question, vector-searches the `docs` table, and prompts the OpenAI chat API with the retrieved context, returning an `Answer(text, sources)`. `brain/loop.py` is a REPL with injectable `input_fn`/`print_fn` (Phase 1 swaps these for robot audio). The typer `chat` command wires real dependencies.

**Tech Stack:** Python 3.12, uv, openai SDK, typer, pytest.

**Depends on:** plan 0a (`Embedder`, `Store`, `DocChunk`, conftest fakes). Plan 0b is needed only for the final manual smoke test (Task 3 Step 4).

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- LLM: OpenAI chat completions, model from `settings.llm_model` (default `"gpt-4o"`); API key from `OPENAI_API_KEY` env var, read implicitly by the openai SDK.
- Settings come from `reachy_vec.config.settings` (pydantic-settings, `REACHY_VEC_` env prefix).
- Tests must not hit the network: use `FakeEmbedder` and `FakeLLMClient` from `tests/conftest.py`; never instantiate `BgeEmbedder` or `openai.OpenAI` in tests.
- Commit after every green test cycle; conventional-commit messages (`feat:`, `test:`, `chore:`).

---

### Task 1: RAG answer

**Files:**
- Modify: `src/reachy_vec/brain/rag.py` (currently a docstring stub)
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `Store.search_docs`, `Embedder`, `FakeLLMClient` (plan 0a).
- Produces: `Answer` dataclass with `text: str, sources: list[str]`; `answer(question: str, *, store: Store, embedder: Embedder, client, model: str, k: int = 5) -> Answer`. `client` is any object with `chat.completions.create(model=..., messages=...)` returning an openai-shaped response.

- [ ] **Step 1: Write the failing test**

`tests/test_rag.py`:

```python
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
    result = answer(
        "when does the pipeline run?",
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
        k=1,
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    user_message = kwargs["messages"][-1]["content"]
    assert "the pipeline runs nightly" in user_message
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL with `ImportError: cannot import name 'Answer'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/reachy_vec/brain/rag.py` with:

```python
"""Retrieval-augmented generation: search docs, prompt the LLM."""

from dataclasses import dataclass

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

SYSTEM_PROMPT = (
    "You are Reachy, a friendly team assistant. Answer using ONLY the provided "
    "team-knowledge context. If the context does not contain the answer, say you "
    "don't know. Keep answers short and conversational - one or two sentences."
)

USER_TEMPLATE = """Context from the team knowledge base:

{context}

Question: {question}"""


@dataclass
class Answer:
    text: str
    sources: list[str]


def answer(
    question: str,
    *,
    store: Store,
    embedder: Embedder,
    client,
    model: str,
    k: int = 5,
) -> Answer:
    query_vector = embedder.embed([question])[0]
    hits = store.search_docs(query_vector, k=k)
    if not hits:
        return Answer(
            text="My knowledge base is empty - ingest some documents first.",
            sources=[],
        )

    context = "\n\n".join(f"[{hit.source}]\n{hit.text}" for hit in hits)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
    )
    return Answer(
        text=response.choices[0].message.content,
        sources=sorted({hit.source for hit in hits}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/rag.py tests/test_rag.py
git commit -m "feat: RAG answer over the docs table via OpenAI chat API"
```

---

### Task 2: Terminal chat loop

**Files:**
- Modify: `src/reachy_vec/brain/loop.py` (currently a docstring stub)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `answer` / `Answer` (Task 1), `Store`, `Embedder` (plan 0a).
- Produces: `chat_loop(*, store, embedder, client, model, input_fn=input, print_fn=print) -> None` — REPL that exits on `exit`, `quit`, or EOF, and prints each answer followed by its sources.

- [ ] **Step 1: Write the failing test**

`tests/test_loop.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'chat_loop'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/reachy_vec/brain/loop.py` with:

```python
"""Terminal conversation loop (Phase 0). Phase 1 swaps input/output for robot audio."""

from reachy_vec.brain.rag import answer
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

EXIT_COMMANDS = {"exit", "quit"}


def chat_loop(
    *,
    store: Store,
    embedder: Embedder,
    client,
    model: str,
    input_fn=input,
    print_fn=print,
) -> None:
    print_fn("Reachy KB chat - ask about your team docs ('exit' to leave).")
    while True:
        try:
            question = input_fn("you> ").strip()
        except EOFError:
            return
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return
        result = answer(question, store=store, embedder=embedder, client=client, model=model)
        print_fn(f"reachy> {result.text}")
        if result.sources:
            print_fn(f"        (sources: {', '.join(result.sources)})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loop.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/loop.py tests/test_loop.py
git commit -m "feat: terminal chat loop over the RAG layer"
```

---

### Task 3: Wire the `chat` CLI command and smoke-test end to end

**Files:**
- Modify: `src/reachy_vec/cli/chat.py`

**Interfaces:**
- Consumes: `chat_loop` (Task 2), `Store`, `BgeEmbedder` (plan 0a), `settings` from `reachy_vec.config`.
- Produces: working `reachy-vec chat` command — the Phase 0 milestone.

- [ ] **Step 1: Implement the chat command**

Replace `src/reachy_vec/cli/chat.py` with:

```python
import typer

from reachy_vec.config import settings
from reachy_vec.brain.loop import chat_loop
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder


def chat() -> None:
    """Chat with the team knowledge base in the terminal (no robot needed)."""
    from openai import OpenAI

    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)
    chat_loop(
        store=store,
        embedder=BgeEmbedder(settings.embedding_model),
        client=OpenAI(),
        model=settings.llm_model,
    )
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS (the command is covered by the existing `--help` smoke test; its body is real-dependency wiring exercised in Step 4).

- [ ] **Step 3: Verify empty-KB guard**

Run: `REACHY_VEC_DATA_DIR=/tmp/reachy-empty uv run reachy-vec chat`
Expected: prints `Knowledge base is empty - run 'reachy-vec ingest <path>' first.` and exits with code 1.

- [ ] **Step 4: Manual end-to-end smoke test (real models, needs OPENAI_API_KEY; requires plan 0b done)**

```bash
mkdir -p /tmp/team-docs
printf 'The nightly data pipeline kicks off at 02:00 UTC and takes about 40 minutes.\n' > /tmp/team-docs/pipeline.md
uv run reachy-vec ingest /tmp/team-docs
uv run reachy-vec chat
# you> when does the pipeline run?
# expect an answer mentioning 02:00 UTC with (sources: /tmp/team-docs/pipeline.md), then: exit
```

Expected: first run downloads the BGE model (~130 MB), then answers correctly.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/cli/chat.py
git commit -m "feat: wire the chat CLI command - Phase 0 milestone"
```

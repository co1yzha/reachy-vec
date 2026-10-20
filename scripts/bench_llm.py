"""Speed + tool-accuracy check for the local LLM (oMLX). Never prints the API key.

Run: uv run --no-sync python scripts/bench_llm.py
"""

import time
from collections.abc import Callable

# (utterance, expected tool name or None)
TOOL_CASES: list[tuple[str, str | None]] = [
    ("tell Bob the review moved to 3pm", "send_message"),
    ("remember that I prefer short answers", "save_note"),
    ("what's the weather like outside", "get_weather"),
    ("what time is it", "get_time"),
    ("open the food mapping demo", "open_url"),
    ("research the latest Reachy Mini SDK release and summarise it", "ask_hermes"),
    ("how are you today", None),
    ("what's a p-value", None),
    ("any demos about food?", None),
    ("thanks, that's all", None),
]

SMALL_TALK = [
    "how's your day going",
    "tell me something fun",
    "what do you like about robots",
    "good morning",
    "are you tired",
    "what's your favourite colour",
    "say something nice",
    "how do you feel today",
    "what can you do",
    "do you like music",
]

MAX_FIRST_SENTENCE_S = 1.5


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def leaked_reasoning(sentences: list[str]) -> bool:
    """True if any spoken sentence contains a think tag (not the word 'think')."""
    markers = ("<think", "</think", "think>")
    return any(m in s.lower() for s in sentences for m in markers)


def score_tool_calls(
    cases: list[tuple[str, str | None]],
    first_tool_of: Callable[[str], str | None],
    available_tools: set[str] | None = None,
) -> tuple[int, int]:
    """Return (correct, scored). Cases expecting a tool that is not in
    `available_tools` (when given) are skipped, not counted."""
    correct = scored = 0
    for utterance, expected in cases:
        if expected is not None and available_tools is not None and expected not in available_tools:
            continue
        scored += 1
        if first_tool_of(utterance) == expected:
            correct += 1
    return correct, scored


def main() -> int:
    from dotenv import load_dotenv

    from reachy_vec.brain.chat import ChatBrain
    from reachy_vec.brain.llm import make_llm_client, uses_local_llm
    from reachy_vec.config import settings
    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder

    load_dotenv()
    client = make_llm_client()
    brain = ChatBrain(
        k=settings.rag_k,
        rag_min_score=settings.rag_min_score,
        store=Store(settings.lancedb_dir),
        embedder=BgeEmbedder(
            settings.embedding_model, query_prefix=settings.embedding_query_prefix
        ),
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
        disable_thinking=uses_local_llm() and settings.llm_disable_thinking,
    )
    print(f"model={settings.llm_model} local={uses_local_llm()} base_url={settings.llm_base_url}")
    brain.warm_up()

    # 1. time to first spoken sentence
    times: list[float] = []
    spoken: list[str] = []
    for question in SMALL_TALK:
        brain.reset()
        start = time.perf_counter()
        first: list[float] = []

        def on_sentence(text: str, first=first, start=start) -> None:
            spoken.append(text)
            if not first:
                first.append(time.perf_counter() - start)

        brain.respond(question, on_sentence=on_sentence)
        if first:
            times.append(first[0])
    med = median(times) if times else float("inf")
    print(
        f"median time to first sentence: {med:.2f}s over {len(times)} turns "
        f"(limit {MAX_FIRST_SENTENCE_S}s)"
    )

    # 2. tool selection
    tools = brain._active_tools()
    names = {t["function"]["name"] for t in tools}

    def first_tool_of(utterance: str) -> str | None:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": brain._system_prompt()},
                {"role": "user", "content": utterance},
            ],
            tools=tools,
            **brain._llm_kwargs,
        )
        calls = response.choices[0].message.tool_calls
        return calls[0].function.name if calls else None

    correct, total = score_tool_calls(TOOL_CASES, first_tool_of, available_tools=names)
    if "ask_hermes" not in names:
        print("note: ask_hermes not registered; its case is skipped")
    print(f"tool score: {correct}/{total}")
    min_correct = total - 1  # one-miss tolerance: 9/10 or 8/9

    # 3. reasoning leak
    leaked = leaked_reasoning(spoken)
    print(f"'think' leaked into speech: {leaked}")

    ok = med < MAX_FIRST_SENTENCE_S and correct >= min_correct and not leaked
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

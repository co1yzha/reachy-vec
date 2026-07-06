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

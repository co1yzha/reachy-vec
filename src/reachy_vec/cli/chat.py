import typer

from reachy_vec.brain.chat import ChatBrain
from reachy_vec.brain.loop import chat_loop
from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder


def chat() -> None:
    """Chat with Reachy in the terminal (history + tools, no robot needed)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()  # expose OPENAI_API_KEY from .env to the openai SDK

    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)
    brain = ChatBrain(
        store=store,
        embedder=BgeEmbedder(
            settings.embedding_model, query_prefix=settings.embedding_query_prefix
        ),
        client=OpenAI(),
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
    )
    chat_loop(brain=brain)

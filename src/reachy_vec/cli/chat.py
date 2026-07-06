import typer

from reachy_vec.config import settings
from reachy_vec.brain.loop import chat_loop
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder


def chat() -> None:
    """Chat with the team knowledge base in the terminal (no robot needed)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()  # expose OPENAI_API_KEY from .env to the openai SDK

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

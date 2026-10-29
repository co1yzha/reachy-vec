import typer

from reachy_vec.brain.chat import ChatBrain
from reachy_vec.brain.loop import chat_loop
from reachy_vec.brain.persona import Persona
from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder


def chat() -> None:
    """Chat with Reachy in the terminal (history + tools, no robot needed)."""
    from dotenv import load_dotenv

    from reachy_vec.brain.llm import make_llm_client, uses_local_llm
    from reachy_vec.cli.run import make_hermes

    load_dotenv()  # expose OPENAI_API_KEY from .env to the openai SDK

    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)
    from reachy_vec.brain.router import Router

    client = make_llm_client()
    thinking_off = uses_local_llm() and settings.llm_disable_thinking
    hermes = make_hermes(store)  # None unless REACHY_VEC_HERMES_ENABLED and hermes is installed
    brain = ChatBrain(
        k=settings.rag_k,
        rag_min_score=settings.rag_min_score,
        store=store,
        embedder=BgeEmbedder(
            settings.embedding_model, query_prefix=settings.embedding_query_prefix
        ),
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
        disable_thinking=thinking_off,
        hermes=hermes,
        router=Router(client, settings.llm_model,
                      llm_kwargs={"extra_body": {"chat_template_kwargs":
                                                 {"enable_thinking": False}}}
                      if thinking_off else None),
        persona=Persona(settings.personality),
    )
    try:
        chat_loop(brain=brain)
    finally:
        if hermes is not None:
            hermes.close()

import os

import typer

from reachy_vec.config import settings


def sync_mongo() -> None:
    """Sync the aixlab.demos MongoDB collection into the knowledge base."""
    from dotenv import load_dotenv

    load_dotenv()
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        typer.echo("MONGODB_URI is not set - add it to .env first.", err=True)
        raise typer.Exit(code=1)

    from pymongo import MongoClient

    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder
    from reachy_vec.store.mongo_sync import sync_demos

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        demos = list(client["aixlab"]["demos"].find())
    except Exception as exc:
        typer.echo(f"Could not read from MongoDB: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    store = Store(settings.lancedb_dir)
    count = sync_demos(demos, store, BgeEmbedder(settings.embedding_model))
    typer.echo(f"Synced {count} chunks from {len(demos)} demos in aixlab.demos.")

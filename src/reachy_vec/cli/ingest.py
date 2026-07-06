from pathlib import Path

import typer

from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import BgeEmbedder, Embedder
from reachy_vec.store.ingestion import ingest_path


def make_embedder() -> Embedder:
    return BgeEmbedder(settings.embedding_model)


def ingest(path: Path) -> None:
    """Ingest team documents at PATH (.md/.txt file or directory) into the knowledge base."""
    if not path.exists():
        typer.echo(f"error: {path} does not exist", err=True)
        raise typer.Exit(code=1)
    store = Store(settings.lancedb_dir)
    count = ingest_path(path, store, make_embedder())
    suffix = "" if count == 1 else "s"
    typer.echo(f"Ingested {count} chunk{suffix} into {settings.lancedb_dir}.")

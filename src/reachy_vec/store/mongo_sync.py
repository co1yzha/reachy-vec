"""Sync the MongoDB demo library (aixlab.demos) into the LanceDB docs table.

MongoDB is the source of truth; this replaces all demo-sourced rows on each
run (idempotent). The pre-existing 1536-dim `embedding` field in Mongo is
ignored - everything is re-embedded locally into the single BGE space.
"""

from datetime import UTC, datetime

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.ingestion import chunk_text
from reachy_vec.store.schemas import DocChunk

DEMO_SOURCE_PREFIX = "demo: "


def format_demo(doc: dict) -> str:
    """Render one demo record as searchable text; tolerates missing fields."""
    title = doc.get("title", "Untitled demo")
    lines = [f"Demo: {title}"]
    if doc.get("project"):
        lines[0] += f" ({doc['project']})"
    if doc.get("authors"):
        lines.append("Authors: " + ", ".join(doc["authors"]))
    if doc.get("tags"):
        lines.append("Tags: " + ", ".join(doc["tags"]))
    if doc.get("url"):
        lines.append(f"URL: {doc['url']}")
    if doc.get("note"):
        lines.append(str(doc["note"]))
    return "\n".join(lines)


def sync_demos(demos: list[dict], store: Store, embedder: Embedder) -> int:
    """Replace all demo-sourced chunks with fresh ones. Returns chunk count.

    Embeddings are computed before the old rows are deleted, so a failure
    mid-way never leaves the store partially emptied.
    """
    now = datetime.now(UTC).isoformat()
    rows: list[DocChunk] = []
    for doc in demos:
        title = doc.get("title", "Untitled demo")
        texts = chunk_text(format_demo(doc))
        vectors = embedder.embed(texts) if texts else []
        rows.extend(
            DocChunk(
                chunk_id=f"mongo:demos/{doc.get('_id')}:{i}",
                text=text,
                vector=vector,
                source=f"{DEMO_SOURCE_PREFIX}{title}",
                ingested_at=now,
            )
            for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        )
    store.delete_docs_by_source_prefix(DEMO_SOURCE_PREFIX)
    store.add_doc_chunks(rows)
    return len(rows)

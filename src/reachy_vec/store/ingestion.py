"""Ingestion pipeline: read .md/.txt files, chunk, embed, write to the store."""

from datetime import UTC, datetime
from pathlib import Path

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.schemas import DocChunk

SUPPORTED_SUFFIXES = {".md", ".txt"}


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars.

    Oversized single paragraphs are hard-split at max_chars boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        while len(para) > max_chars:
            pieces.append(para[:max_chars])
            para = para[max_chars:]
        if para:
            pieces.append(para)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def ingest_path(path: Path, store: Store, embedder: Embedder) -> int:
    """Ingest one file or every supported file under a directory tree."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*") if p.suffix in SUPPORTED_SUFFIXES and p.is_file()
        )

    now = datetime.now(UTC).isoformat()
    written = 0
    for file in files:
        texts = chunk_text(file.read_text())
        if not texts:
            continue
        vectors = embedder.embed(texts)
        store.add_doc_chunks(
            [
                DocChunk(
                    chunk_id=f"{file}:{i}",
                    text=text,
                    vector=vector,
                    source=str(file),
                    ingested_at=now,
                )
                for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
            ]
        )
        written += len(texts)
    return written

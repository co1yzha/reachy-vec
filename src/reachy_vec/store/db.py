"""LanceDB connection and vector-search helpers."""

from pathlib import Path

import lancedb

from reachy_vec.store.schemas import DocChunk

DOCS_TABLE = "docs"


class Store:
    """One embedded LanceDB database holding all reachy-vec tables."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(db_path)

    def _docs(self) -> lancedb.table.Table:
        if DOCS_TABLE not in self._db.list_tables().tables:
            self._db.create_table(DOCS_TABLE, schema=DocChunk)
        return self._db.open_table(DOCS_TABLE)

    def add_doc_chunks(self, chunks: list[DocChunk]) -> None:
        if chunks:
            self._docs().add(chunks)

    def search_docs(self, query_vector: list[float], k: int = 5) -> list[DocChunk]:
        return self._docs().search(query_vector).limit(k).to_pydantic(DocChunk)

    def doc_count(self) -> int:
        return self._docs().count_rows()

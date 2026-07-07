"""LanceDB connection and vector-search helpers."""

from pathlib import Path

import lancedb

from reachy_vec.store.schemas import DocChunk, FaceRow, GreetingRow, MemoryRow, MessageRow

DOCS_TABLE = "docs"
PEOPLE_TABLE = "people"
GREETINGS_TABLE = "greetings"
MEMORIES_TABLE = "memories"
MESSAGES_TABLE = "messages"


class Store:
    """One embedded LanceDB database holding all reachy-vec tables."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(db_path)

    def _table(self, name: str, schema) -> lancedb.table.Table:
        if name not in self._db.list_tables().tables:
            self._db.create_table(name, schema=schema)
        return self._db.open_table(name)

    # -- docs (Phase 0) ---------------------------------------------------

    def _docs(self) -> lancedb.table.Table:
        return self._table(DOCS_TABLE, DocChunk)

    def add_doc_chunks(self, chunks: list[DocChunk]) -> None:
        if chunks:
            self._docs().add(chunks)

    def search_docs(self, query_vector: list[float], k: int = 5) -> list[DocChunk]:
        return self._docs().search(query_vector).limit(k).to_pydantic(DocChunk)

    def doc_count(self) -> int:
        return self._docs().count_rows()

    def search_docs_scored(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[DocChunk, float]]:
        """Like search_docs, with cosine similarity scores (1.0 = identical)."""
        if self.doc_count() == 0:
            return []
        rows = self._docs().search(query_vector).metric("cosine").limit(k).to_list()
        return [
            (
                DocChunk(**{k_: row[k_] for k_ in DocChunk.model_fields}),
                1.0 - row["_distance"],
            )
            for row in rows
        ]

    def delete_docs_by_source_prefix(self, prefix: str) -> None:
        escaped = prefix.replace("'", "''")
        self._docs().delete(f"source LIKE '{escaped}%'")

    def demo_titles(self, limit: int = 30) -> list[str]:
        """Distinct demo titles (sources with the 'demo: ' prefix stripped)."""
        sources = {
            r["source"]
            for r in self._docs().to_arrow().to_pylist()
            if r["source"].startswith("demo: ")
        }
        return sorted(s.removeprefix("demo: ") for s in sources)[:limit]

    # -- people + greetings (Phase 1) --------------------------------------

    def add_face_rows(self, rows: list[FaceRow]) -> None:
        if rows:
            self._table(PEOPLE_TABLE, FaceRow).add(rows)

    def match_face(self, vector: list[float], k: int = 5) -> tuple[str, str, float] | None:
        """k-NN majority vote over people rows.

        Returns (person_id, name, best cosine similarity of the winning
        person) or None if nobody is enrolled.
        """
        table = self._table(PEOPLE_TABLE, FaceRow)
        if table.count_rows() == 0:
            return None
        hits = table.search(vector).metric("cosine").limit(k).to_list()
        counts: dict[str, int] = {}
        for row in hits:
            counts[row["person_id"]] = counts.get(row["person_id"], 0) + 1
        winner = max(counts, key=counts.get)
        best_distance = min(r["_distance"] for r in hits if r["person_id"] == winner)
        name = next(r["name"] for r in hits if r["person_id"] == winner)
        return winner, name, 1.0 - best_distance

    def people_count(self) -> int:
        table = self._table(PEOPLE_TABLE, FaceRow)
        return len({r["person_id"] for r in table.to_arrow().to_pylist()})

    # -- memories (Phase 2a) ------------------------------------------------

    def add_memories(self, rows: list[MemoryRow]) -> None:
        if rows:
            self._table(MEMORIES_TABLE, MemoryRow).add(rows)

    def search_memories(
        self, query_vector: list[float], *, person_id: str, k: int = 3
    ) -> list[MemoryRow]:
        table = self._table(MEMORIES_TABLE, MemoryRow)
        if table.count_rows() == 0:
            return []
        escaped = person_id.replace("'", "''")
        return (
            table.search(query_vector)
            .metric("cosine")
            .where(f"person_id = '{escaped}'")
            .limit(k)
            .to_pydantic(MemoryRow)
        )

    # -- messages (Phase 3) --------------------------------------------------

    def add_message(self, row: MessageRow) -> None:
        self._table(MESSAGES_TABLE, MessageRow).add([row])

    def pending_messages_for(self, person_id: str) -> list[MessageRow]:
        table = self._table(MESSAGES_TABLE, MessageRow)
        rows = [
            r
            for r in table.to_arrow().to_pylist()
            if r["to_person"] == person_id and not r["delivered_at"]
        ]
        rows.sort(key=lambda r: r["created_at"])
        return [MessageRow(**{k: r[k] for k in MessageRow.model_fields}) for r in rows]

    def mark_delivered(self, message_id: str) -> None:
        from datetime import datetime, timezone

        escaped = message_id.replace("'", "''")
        self._table(MESSAGES_TABLE, MessageRow).update(
            where=f"message_id = '{escaped}'",
            values={"delivered_at": datetime.now(timezone.utc).isoformat()},
        )

    def find_person_by_name(self, name: str) -> tuple[str, str] | None:
        """Case-insensitive lookup among enrolled people."""
        wanted = name.strip().lower()
        for r in self._table(PEOPLE_TABLE, FaceRow).to_arrow().to_pylist():
            if r["name"].lower() == wanted:
                return r["person_id"], r["name"]
        return None

    def get_last_greeted(self, person_id: str) -> str | None:
        table = self._table(GREETINGS_TABLE, GreetingRow)
        rows = [
            r
            for r in table.to_arrow().to_pylist()
            if r["person_id"] == person_id
        ]
        return rows[0]["last_greeted"] if rows else None

    def set_last_greeted(self, person_id: str, when_iso: str) -> None:
        table = self._table(GREETINGS_TABLE, GreetingRow)
        table.delete(f"person_id = '{person_id}'")
        table.add([GreetingRow(person_id=person_id, last_greeted=when_iso)])

"""Retrieval-augmented generation: search docs, prompt the LLM."""

from dataclasses import dataclass

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

SYSTEM_PROMPT = (
    "You are Reachy, a friendly team assistant. Answer using ONLY the provided "
    "team-knowledge context. If the context does not contain the answer, say you "
    "don't know. Keep answers short and conversational - one or two sentences."
)

USER_TEMPLATE = """Context from the team knowledge base:

{context}

Question: {question}"""


@dataclass
class Answer:
    text: str
    sources: list[str]


def answer(
    question: str,
    *,
    store: Store,
    embedder: Embedder,
    client,
    model: str,
    k: int = 5,
) -> Answer:
    query_vector = embedder.embed([question])[0]
    hits = store.search_docs(query_vector, k=k)
    if not hits:
        return Answer(
            text="My knowledge base is empty - ingest some documents first.",
            sources=[],
        )

    context = "\n\n".join(f"[{hit.source}]\n{hit.text}" for hit in hits)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
    )
    return Answer(
        text=response.choices[0].message.content,
        sources=sorted({hit.source for hit in hits}),
    )

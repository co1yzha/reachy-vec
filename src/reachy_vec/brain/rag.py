"""Retrieval-augmented generation: search docs, prompt the LLM.

Score-gated: strong retrieval -> grounded answer with sources; weak
retrieval -> general-knowledge fallback explicitly labeled as such.
"""

from dataclasses import dataclass

from reachy_vec.config import settings
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

SYSTEM_PROMPT = (
    "You are Reachy, a friendly team assistant. Answer using ONLY the provided "
    "team-knowledge context. If the context does not contain the answer, say you "
    "don't know. Keep answers short and conversational - one or two sentences."
)

FALLBACK_SYSTEM_PROMPT = (
    "You are Reachy, a friendly team assistant. The team knowledge base has "
    "nothing relevant to this question, so answer from general knowledge. "
    "Start your answer with exactly: 'Not from our team docs, but'. Keep it "
    "short and conversational - one or two sentences."
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
    scored = store.search_docs_scored(query_vector, k=k)

    if not scored:
        return Answer(
            text="My knowledge base is empty - ingest some documents first.",
            sources=[],
        )

    best_score = scored[0][1]
    if best_score < settings.rag_min_score:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return Answer(text=response.choices[0].message.content, sources=[])

    hits = [chunk for chunk, _score in scored]
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

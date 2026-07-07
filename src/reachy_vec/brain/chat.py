"""ChatBrain: personable, conversational, tool-using brain with memory.

One LLM call per turn: docs and the current speaker's memories are retrieved
locally every turn and injected as context. Tool calls (open a demo, save a
note) cost a second call - acceptable for side effects. Conversations are
bracketed by begin_conversation/end_conversation; ending one distills up to
three memories about the person into the store.
"""

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Callable

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.schemas import MemoryRow

logger = logging.getLogger(__name__)

PERSONALITY = (
    "You are Reachy, the team's desk robot - a small, expressive robot from "
    "the north of England who works as the lab's resident data scientist and "
    "knows the team's demo library inside out. You're energetic and upbeat "
    "but professional: friendly northern directness, no fluff, plain words "
    "over jargon, and the odd 'brilliant', 'proper', or 'dead easy' where it "
    "fits - never laying the dialect on thick. You get genuinely excited "
    "about good data: a tidy pipeline or a well-made chart delights you, and "
    "you'll happily nerd out about what the numbers actually show. You have "
    "warmth and a dry sense of humour; you're happy to chat about anything, "
    "offer an honest opinion when asked, and occasionally ask a short "
    "question back. "
    "Your words are SPOKEN ALOUD, so answer in one or two short conversational "
    "sentences; never use lists, markdown, or URLs in speech.\n\n"
    "Each user turn includes retrieved context: team knowledge with relevance "
    "scores, and things you remember about this person from earlier visits. "
    "Use what's relevant; mention demos by name. When your answer doesn't come "
    "from the team library, slip in a light, natural signal (for example 'off "
    "the top of my head...') - vary the phrasing, never a stock disclaimer, "
    "and skip it for casual chit-chat. Weave in remembered details naturally "
    "when they fit; don't recite them.\n\n"
    "Tools: open_url opens a demo in the browser (take the URL from context); "
    "save_note stores something worth remembering about this person when they "
    "ask you to remember, or share a clear preference."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a web page (e.g. a team demo) in the browser on the lab PC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to open"},
                    "title": {"type": "string", "description": "what is being opened"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Remember something about the current person for future visits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "short third-person fact to remember"},
                },
                "required": ["note"],
            },
        },
    },
]

SUMMARY_PROMPT = (
    "Review the conversation you just had with {name}. Write up to 3 short "
    "third-person notes worth remembering about them for future visits (facts, "
    "preferences, plans), one per line starting with '- '. If it was only "
    "trivial chit-chat, reply with exactly: NONE"
)

MAX_HISTORY_MESSAGES = 20

CONTEXT_TEMPLATE = """[Team knowledge context - relevance scores 0..1]
{context}
[/context]
{memories}
{speaker}: {question}"""

MEMORIES_TEMPLATE = """[What I remember about {name}]
{memories}
[/memories]
"""


def default_opener(url: str) -> None:
    subprocess.run(["open", url], check=False)


class ChatBrain:
    def __init__(
        self,
        *,
        store: Store,
        embedder: Embedder,
        client,
        model: str,
        opener: Callable[[str], None] = default_opener,
        k: int = 5,
    ):
        self._store = store
        self._embedder = embedder
        self._client = client
        self._model = model
        self._opener = opener
        self._k = k
        self._history: list[dict] = []
        self._person_id: str | None = None
        self._person_name: str | None = None
        self._exchanges = 0

    # -- conversation lifecycle --------------------------------------------

    def begin_conversation(self, person_id: str | None, name: str | None) -> None:
        self.reset()
        self._person_id = person_id
        self._person_name = name

    def end_conversation(self) -> None:
        """Distill the visit into stored memories, then reset."""
        try:
            if self._person_id and self._exchanges > 0:
                self._summarize_and_store()
        except Exception:
            logger.exception("conversation summary failed; skipping")
        finally:
            self.reset()

    def reset(self) -> None:
        self._history = []
        self._person_id = None
        self._person_name = None
        self._exchanges = 0

    # -- responding ---------------------------------------------------------

    def respond(self, question: str, speaker_name: str | None = None) -> str:
        vector = self._embedder.embed([question])[0]
        self._history.append(
            {
                "role": "user",
                "content": CONTEXT_TEMPLATE.format(
                    context=self._retrieve_docs(vector) or "(nothing relevant found)",
                    memories=self._retrieve_memories(vector),
                    speaker=speaker_name or self._person_name or "User",
                    question=question,
                ),
            }
        )
        message = self._complete()
        if getattr(message, "tool_calls", None):
            self._history.append(_assistant_tool_message(message))
            for call in message.tool_calls:
                self._history.append(self._execute_tool(call))
            message = self._complete()
        text = (message.content or "").strip()
        self._history.append({"role": "assistant", "content": text})
        self._exchanges += 1
        self._trim()
        logger.info("reply to %s: %r", speaker_name or "user", text)
        return text

    # -- internals ------------------------------------------------------------

    def _retrieve_docs(self, vector: list[float]) -> str:
        scored = self._store.search_docs_scored(vector, k=self._k)
        return "\n\n".join(
            f"[{chunk.source} | score {score:.2f}]\n{chunk.text}"
            for chunk, score in scored
        )

    def _retrieve_memories(self, vector: list[float]) -> str:
        if not self._person_id:
            return ""
        hits = self._store.search_memories(vector, person_id=self._person_id, k=3)
        if not hits:
            return ""
        return MEMORIES_TEMPLATE.format(
            name=self._person_name or "them",
            memories="\n".join(f"- {h.text}" for h in hits),
        )

    def _complete(self):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": PERSONALITY}, *self._history],
            tools=TOOLS,
        )
        return response.choices[0].message

    def _execute_tool(self, call) -> dict:
        try:
            args = json.loads(call.function.arguments)
        except Exception:
            args = {}
        handlers = {"open_url": self._tool_open_url, "save_note": self._tool_save_note}
        handler = handlers.get(call.function.name)
        result = handler(args) if handler else "unknown tool"
        return {"role": "tool", "tool_call_id": call.id, "content": result}

    def _tool_open_url(self, args: dict) -> str:
        url = args.get("url", "")
        if not url.startswith(("http://", "https://")):
            return f"refused: {url!r} is not an http(s) URL"
        try:
            self._opener(url)
        except Exception as exc:
            logger.exception("open_url failed")
            return f"failed to open: {exc}"
        logger.info("open_url: %s", url)
        return f"opened {url} in the browser"

    def _tool_save_note(self, args: dict) -> str:
        if not self._person_id:
            return "can't save: I don't know who I'm talking to (not a recognized visit)"
        note = args.get("note", "").strip()
        if not note:
            return "can't save an empty note"
        self._store_memories([note])
        logger.info("save_note for %s: %r", self._person_name, note)
        return f"noted: {note}"

    def _summarize_and_store(self) -> None:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": PERSONALITY},
                *self._history,
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(name=self._person_name or "them"),
                },
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if text.upper() == "NONE":
            return
        notes = [line.lstrip("- ").strip() for line in text.splitlines() if line.strip()]
        self._store_memories([n for n in notes if n][:3])

    def _store_memories(self, notes: list[str]) -> None:
        if not notes:
            return
        now = datetime.now(timezone.utc).isoformat()
        vectors = self._embedder.embed(notes)
        rows = []
        for note, vector in zip(notes, vectors):
            if self._is_duplicate_memory(vector):
                logger.info("skipping near-duplicate memory: %r", note)
                continue
            rows.append(
                MemoryRow(
                    memory_id=f"mem-{uuid.uuid4().hex[:10]}",
                    person_id=self._person_id,
                    text=note,
                    vector=vector,
                    created_at=now,
                )
            )
        self._store.add_memories(rows)

    DUPLICATE_SIMILARITY = 0.97

    def _is_duplicate_memory(self, vector: list[float]) -> bool:
        hits = self._store.search_memories(vector, person_id=self._person_id, k=1)
        if not hits:
            return False
        existing = hits[0].vector
        dot = sum(a * b for a, b in zip(vector, existing))
        norms = (sum(a * a for a in vector) ** 0.5) * (sum(b * b for b in existing) ** 0.5)
        return norms > 0 and dot / norms >= self.DUPLICATE_SIMILARITY

    def _trim(self) -> None:
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
            # never let the window start mid-tool-exchange
            while self._history and self._history[0]["role"] == "tool":
                self._history.pop(0)


def _assistant_tool_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in message.tool_calls
        ],
    }

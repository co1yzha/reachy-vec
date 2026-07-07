"""ChatBrain: personable, conversational, tool-using brain.

One LLM call per turn: retrieval runs locally every turn and is injected as
context (pushed, not pulled), so answering stays fast. Tool calls (actions
like opening a demo in the browser) cost a second call - acceptable for
side effects. History lives here; the Oracle resets it per conversation.
"""

import json
import logging
import subprocess
from typing import Callable

from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder

logger = logging.getLogger(__name__)

PERSONALITY = (
    "You are Reachy, the team's desk robot and resident familiar - a small "
    "expressive robot who knows the team's demo library inside out. Warm, a "
    "little playful, always professional. Your words are SPOKEN ALOUD, so "
    "answer in one or two short conversational sentences; never use lists, "
    "markdown, or URLs in speech.\n\n"
    "Each user turn includes retrieved context from the team knowledge base "
    "with relevance scores. When the context answers the question, use it and "
    "mention the demo or document by name. When it doesn't, answer from "
    "general knowledge but start with: 'Not from our team docs, but'.\n\n"
    "When asked to open, show, or launch a demo, call the open_url tool with "
    "the demo's URL from the context, then confirm briefly in speech."
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
    }
]

MAX_HISTORY_MESSAGES = 20

CONTEXT_TEMPLATE = """[Team knowledge context - relevance scores 0..1]
{context}
[/context]

{speaker}: {question}"""


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

    def reset(self) -> None:
        self._history = []

    def respond(self, question: str, speaker_name: str | None = None) -> str:
        context = self._retrieve(question)
        self._history.append(
            {
                "role": "user",
                "content": CONTEXT_TEMPLATE.format(
                    context=context or "(nothing relevant found)",
                    speaker=speaker_name or "User",
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
        self._trim()
        return text

    # -- internals --------------------------------------------------------

    def _retrieve(self, question: str) -> str:
        vector = self._embedder.embed([question])[0]
        scored = self._store.search_docs_scored(vector, k=self._k)
        return "\n\n".join(
            f"[{chunk.source} | score {score:.2f}]\n{chunk.text}"
            for chunk, score in scored
        )

    def _complete(self):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": PERSONALITY}, *self._history],
            tools=TOOLS,
        )
        return response.choices[0].message

    def _execute_tool(self, call) -> dict:
        result = "unknown tool"
        if call.function.name == "open_url":
            try:
                args = json.loads(call.function.arguments)
                url = args.get("url", "")
                if url.startswith(("http://", "https://")):
                    self._opener(url)
                    result = f"opened {url} in the browser"
                    logger.info("open_url: %s", url)
                else:
                    result = f"refused: {url!r} is not an http(s) URL"
            except Exception as exc:
                logger.exception("open_url failed")
                result = f"failed to open: {exc}"
        return {"role": "tool", "tool_call_id": call.id, "content": result}

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

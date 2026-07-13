"""ChatBrain: personable, conversational, tool-using brain with memory.

One LLM call per turn: docs and the current speaker's memories are retrieved
locally every turn and injected as context. Tool calls (open a demo, save a
note) cost a second call - acceptable for side effects. Conversations are
bracketed by begin_conversation/end_conversation; ending one distills up to
three memories about the person into the store.
"""

import json
import logging
import os
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.schemas import MemoryRow

logger = logging.getLogger(__name__)


class SpeechInterrupted(Exception):
    """Raised from on_sentence when the user barges in; caught in ChatBrain."""


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
    "ask you to remember, or share a clear preference; send_message relays a "
    "spoken message to an enrolled teammate next time you see them; "
    "get_weather checks the live weather outside the lab; "
    "get_time tells the current local date and time."
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
                    "note": {
                        "type": "string",
                        "description": "short third-person fact to remember",
                    },
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current live weather outside the lab.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Relay a message to an enrolled teammate; it is spoken to them "
                "the next time the robot sees their face."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_name": {"type": "string", "description": "recipient's name"},
                    "message": {"type": "string", "description": "what to tell them"},
                },
                "required": ["to_name", "message"],
            },
        },
    },
]

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current, real-time, or very recent information "
            "that isn't in the team library and isn't stable general knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "what to search for"}},
            "required": ["query"],
        },
    },
}

WEB_SEARCH_HINT = (
    " web_search looks things up on the live web - use it only when the user wants "
    "current or up-to-the-minute info you can't give from the team library or your "
    "own knowledge (it costs a limited search budget, so don't use it for chit-chat)."
)

LOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "look",
        "description": (
            "Look through the robot's camera and answer about what's physically "
            "in view right now (surroundings, objects, how many people, or to "
            "read visible text)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "what to look for or answer about the scene; "
                        "omit to describe the view"
                    ),
                },
            },
        },
    },
}

LOOK_HINT = (
    " look lets you actually see through your camera - use it when the user asks "
    "about your physical surroundings, to read something in view, or to count "
    "people; describe only what you can see."
)

SELFIE_TOOL = {
    "type": "function",
    "function": {
        "name": "selfie",
        "description": (
            "Take a photo of the person(s) in front of you and show it to them. "
            "Use when asked for a photo, selfie, or picture."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

SELFIE_HINT = (
    " selfie snaps a photo through your camera and pops it up on screen - use it "
    "when the user asks you to take a photo, selfie, or picture of them."
)


def fetch_tavily(query: str, api_key: str, max_results: int = 3, timeout: float = 5.0) -> str:
    """Tavily search; returns the spoken-ready `answer`. Raises urllib errors."""
    import urllib.request

    body = json.dumps(
        {"query": query, "include_answer": True, "max_results": max_results}
    ).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return (data.get("answer") or "").strip()


SUMMARY_PROMPT = (
    "Review the conversation you just had with {name}. Write up to 3 short "
    "third-person notes worth remembering about them for future visits (facts, "
    "preferences, plans), one per line starting with '- '. If it was only "
    "trivial chit-chat, reply with exactly: NONE"
)

MAX_HISTORY_MESSAGES = 20

# WMO weather interpretation codes (Open-Meteo), condensed
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Current conditions from Open-Meteo (no API key)."""
    import urllib.request

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
    )
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())["current"]

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
        reasoning_effort: str | None = None,
        opener: Callable[[str], None] = default_opener,
        k: int = 5,
        web_search: bool = False,
        web_search_fetch=None,
        look_fn: Callable[[str], str] | None = None,
        selfie_fn: Callable[[], str] | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._client = client
        self._model = model
        # gpt-4o and older reject the reasoning_effort param
        self._llm_kwargs = (
            {"reasoning_effort": reasoning_effort}
            if reasoning_effort and model.startswith("gpt-5")
            else {}
        )
        self._opener = opener
        self._k = k
        if web_search_fetch is None and web_search:
            key = os.getenv("TAVILY_API_KEY")
            if key:
                web_search_fetch = lambda q: fetch_tavily(q, key)  # noqa: E731
            else:
                logger.warning("web_search enabled but TAVILY_API_KEY missing; disabling")
        self._web_search_fetch = web_search_fetch
        self._look_fn = look_fn
        self._selfie_fn = selfie_fn
        self._history: list[dict] = []
        self._person_id: str | None = None
        self._person_name: str | None = None
        self._turn: TurnIdentity = ANONYMOUS
        self._participants: dict[str, str | None] = {}
        self._exchanges = 0
        from reachy_vec.config import settings

        self._weather_fetch = lambda: fetch_open_meteo(
            settings.weather_lat, settings.weather_lon
        )

    # -- conversation lifecycle --------------------------------------------

    def begin_conversation(self, person_id: str | None, name: str | None) -> None:
        self.reset()
        self._person_id = person_id
        self._person_name = name
        if person_id:
            self._participants[person_id] = name

    def end_conversation(self) -> None:
        """Distill the visit into stored memories, then reset."""
        try:
            if self._participants and self._exchanges > 0:
                self._summarize_and_store()
        except Exception:
            logger.exception("conversation summary failed; skipping")
        finally:
            self.reset()

    def reset(self) -> None:
        self._history = []
        self._person_id = None
        self._person_name = None
        self._turn = ANONYMOUS
        self._participants = {}
        self._exchanges = 0

    # -- responding ---------------------------------------------------------

    def respond(
        self,
        question: str,
        identity: TurnIdentity | None = None,
        on_sentence: Callable[[str], None] | None = None,
    ) -> str:
        """Answer one utterance, attributed to `identity` (the fused per-turn
        speaker; None = anonymous). With on_sentence, the LLM response streams
        and each completed sentence is emitted as it arrives (speech starts
        after the first sentence instead of the full reply)."""
        self._turn = identity or ANONYMOUS
        if self._turn.person_id:
            self._participants[self._turn.person_id] = self._turn.name
        vector = self._embedder.embed_query(question)
        self._history.append(
            {
                "role": "user",
                "content": CONTEXT_TEMPLATE.format(
                    context=self._retrieve_docs(vector, question)
                    or "(nothing relevant found)",
                    memories=self._retrieve_memories(vector),
                    speaker=self._turn.name or "User",
                    question=question,
                ),
            }
        )
        message = self._complete(on_sentence)
        interrupted = getattr(message, "interrupted", False)
        if not interrupted and getattr(message, "tool_calls", None):
            self._history.append(_assistant_tool_message(message))
            for call in message.tool_calls:
                self._history.append(self._execute_tool(call))
            message = self._complete(on_sentence)
            interrupted = getattr(message, "interrupted", False)
        text = (message.content or "").strip()
        self._history.append(
            {
                "role": "assistant",
                "content": f"{text} -- (interrupted)" if interrupted else text,
            }
        )
        self._exchanges += 1
        self._trim()
        logger.info(
            "reply to %s: %r%s",
            self._turn.name or "user",
            text,
            " (interrupted)" if interrupted else "",
        )
        return text

    # -- internals ------------------------------------------------------------

    def _retrieve_docs(self, vector: list[float], question: str) -> str:
        scored = self._store.search_docs_scored(vector, k=self._k, query_text=question)
        return "\n\n".join(
            f"[{chunk.source} | score {score:.2f}]\n{chunk.text}"
            for chunk, score in scored
        )

    def _retrieve_memories(self, vector: list[float]) -> str:
        if not self._turn.person_id:
            return ""
        hits = self._store.search_memories(vector, person_id=self._turn.person_id, k=3)
        if not hits:
            return ""
        return MEMORIES_TEMPLATE.format(
            name=self._turn.name or "them",
            memories="\n".join(f"- {h.text}" for h in hits),
        )

    def _active_tools(self) -> list:
        tools = list(TOOLS)
        if self._web_search_fetch:
            tools.append(WEB_SEARCH_TOOL)
        if self._look_fn:
            tools.append(LOOK_TOOL)
        if self._selfie_fn:
            tools.append(SELFIE_TOOL)
        return tools

    def _system_prompt(self) -> str:
        prompt = PERSONALITY
        if self._web_search_fetch:
            prompt += WEB_SEARCH_HINT
        if self._look_fn:
            prompt += LOOK_HINT
        if self._selfie_fn:
            prompt += SELFIE_HINT
        return prompt

    def _complete(self, on_sentence: Callable[[str], None] | None = None):
        if on_sentence is None:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": self._system_prompt()}, *self._history],
                tools=self._active_tools(),
                **self._llm_kwargs,
            )
            return response.choices[0].message
        return self._complete_streaming(on_sentence)

    def _complete_streaming(self, on_sentence: Callable[[str], None]):
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": self._system_prompt()}, *self._history],
            tools=self._active_tools(),
            stream=True,
            **self._llm_kwargs,
        )
        content, buffer = "", ""
        tool_calls: dict[int, dict] = {}
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                for tc in getattr(delta, "tool_calls", None) or []:
                    entry = tool_calls.setdefault(
                        tc.index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if getattr(tc.function, "name", None):
                        entry["name"] = tc.function.name
                    if getattr(tc.function, "arguments", None):
                        entry["arguments"] += tc.function.arguments
                if getattr(delta, "content", None):
                    content += delta.content
                    if not tool_calls:  # don't speak alongside pending tool calls
                        buffer += delta.content
                        buffer = _flush_sentences(buffer, on_sentence)
            if not tool_calls and buffer.strip():
                on_sentence(buffer.strip())
        except SpeechInterrupted:
            if hasattr(stream, "close"):
                stream.close()
            return _StreamedMessage(content, {}, interrupted=True)
        return _StreamedMessage(content, tool_calls)

    def _execute_tool(self, call) -> dict:
        try:
            args = json.loads(call.function.arguments)
        except Exception:
            args = {}
        handlers = {
            "open_url": self._tool_open_url,
            "save_note": self._tool_save_note,
            "send_message": self._tool_send_message,
            "get_weather": self._tool_get_weather,
            "get_time": self._tool_get_time,
            "web_search": self._tool_web_search,
            "look": self._tool_look,
            "selfie": self._tool_selfie,
        }
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
        if not self._turn.person_id:
            return "can't save: I don't know who's speaking (not a recognized voice/face)"
        note = args.get("note", "").strip()
        if not note:
            return "can't save an empty note"
        self._store_memories([note], person_id=self._turn.person_id)
        logger.info("save_note for %s: %r", self._turn.name, note)
        return f"noted: {note}"

    def _tool_get_time(self, args: dict) -> str:
        now = datetime.now().astimezone()
        return f"the time is {now:%A %d %B %Y, %H:%M} ({now.tzname()})"

    def _tool_web_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "can't search: empty query"
        if self._web_search_fetch is None:
            return "web search isn't set up right now"
        import urllib.error

        try:
            answer = self._web_search_fetch(query)
        except urllib.error.HTTPError as exc:
            if exc.code in (432, 433):
                logger.warning("Tavily credits exhausted (HTTP %s)", exc.code)
                return (
                    "I've used up my web-search allowance for now, so I can't look "
                    "that up - tell the user plainly."
                )
            if exc.code == 429:
                return "web search is rate-limited right now; suggest trying again in a moment"
            if exc.code == 401:
                logger.error("Tavily auth failed (HTTP 401) - check TAVILY_API_KEY")
                return "web search isn't set up properly right now"
            logger.exception("Tavily HTTP error %s", exc.code)
            return "couldn't reach web search just now"
        except Exception:
            logger.exception("Tavily request failed")
            return "couldn't reach web search just now"
        return answer or "I searched but didn't find a clear answer"

    def _tool_look(self, args: dict) -> str:
        if self._look_fn is None:
            return "I can't see right now."
        question = (args.get("question") or "").strip()
        try:
            return self._look_fn(question)
        except Exception:
            logger.exception("look tool failed")
            return "I had trouble seeing just now."

    def _tool_selfie(self, args: dict) -> str:
        if self._selfie_fn is None:
            return "I can't take a photo right now."
        try:
            return self._selfie_fn()
        except Exception:
            logger.exception("selfie tool failed")
            return "I couldn't take the photo just now."

    def _tool_get_weather(self, args: dict) -> str:
        try:
            current = self._weather_fetch()
        except Exception:
            logger.exception("weather fetch failed")
            return "couldn't reach the weather service right now"
        condition = WEATHER_CODES.get(current.get("weather_code"), "unknown conditions")
        return (
            f"current weather: {current.get('temperature_2m')}C "
            f"(feels like {current.get('apparent_temperature')}C), {condition}, "
            f"wind {current.get('wind_speed_10m')} km/h"
        )

    def _tool_send_message(self, args: dict) -> str:
        if not self._turn.person_id:
            return "can't send: I don't know who's asking (not a recognized voice/face)"
        to_name = args.get("to_name", "").strip()
        text = args.get("message", "").strip()
        if not to_name or not text:
            return "can't send: need both a recipient and a message"
        recipient = self._store.find_person_by_name(to_name)
        if recipient is None:
            return (
                f"can't send: I don't know anyone called {to_name} - "
                "I can only relay messages to people I've met (enrolled)"
            )
        to_person, resolved_name = recipient
        from reachy_vec.store.schemas import MessageRow

        self._store.add_message(
            MessageRow(
                message_id=f"msg-{uuid.uuid4().hex[:10]}",
                from_person=self._turn.person_id,
                from_name=self._turn.name or "someone",
                to_person=to_person,
                to_name=resolved_name,
                text=text,
                created_at=datetime.now(UTC).isoformat(),
                delivered_at="",
            )
        )
        logger.info("message queued for %s from %s: %r", resolved_name, self._turn.name, text)
        return f"message queued for {resolved_name}; I'll pass it on next time I see them"

    def _summarize_and_store(self) -> None:
        """One distillation pass per enrolled person who spoke this visit."""
        for person_id, name in self._participants.items():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": PERSONALITY},
                    *self._history,
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(name=name or "them"),
                    },
                ],
                **self._llm_kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            if text.upper() == "NONE":
                continue
            notes = [line.lstrip("- ").strip() for line in text.splitlines() if line.strip()]
            self._store_memories([n for n in notes if n][:3], person_id=person_id)

    def _store_memories(self, notes: list[str], *, person_id: str) -> None:
        if not notes:
            return
        now = datetime.now(UTC).isoformat()
        vectors = self._embedder.embed(notes)
        rows = []
        for note, vector in zip(notes, vectors, strict=True):
            if self._is_duplicate_memory(vector, person_id=person_id):
                logger.info("skipping near-duplicate memory: %r", note)
                continue
            rows.append(
                MemoryRow(
                    memory_id=f"mem-{uuid.uuid4().hex[:10]}",
                    person_id=person_id,
                    text=note,
                    vector=vector,
                    created_at=now,
                )
            )
        self._store.add_memories(rows)

    DUPLICATE_SIMILARITY = 0.97

    def _is_duplicate_memory(self, vector: list[float], *, person_id: str) -> bool:
        hits = self._store.search_memories(vector, person_id=person_id, k=1)
        if not hits:
            return False
        existing = hits[0].vector
        dot = sum(a * b for a, b in zip(vector, existing, strict=True))
        norms = (sum(a * a for a in vector) ** 0.5) * (sum(b * b for b in existing) ** 0.5)
        return norms > 0 and dot / norms >= self.DUPLICATE_SIMILARITY

    def _trim(self) -> None:
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
            # never let the window start mid-tool-exchange
            while self._history and self._history[0]["role"] == "tool":
                self._history.pop(0)


SENTENCE_END = (". ", "! ", "? ")


def _flush_sentences(buffer: str, on_sentence: Callable[[str], None]) -> str:
    """Emit each complete sentence in buffer; return the unfinished tail."""
    while True:
        cut = -1
        for mark in SENTENCE_END:
            found = buffer.find(mark)
            if found != -1 and (cut == -1 or found < cut):
                cut = found
        if cut == -1:
            return buffer
        sentence, buffer = buffer[: cut + 1].strip(), buffer[cut + 2 :]
        if sentence:
            on_sentence(sentence)


class _ToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _ToolFunction(name, arguments)


class _StreamedMessage:
    def __init__(self, content: str, tool_calls: dict[int, dict], interrupted: bool = False):
        self.content = content or None
        self.interrupted = interrupted
        self.tool_calls = [
            _ToolCall(entry["id"] or f"call_{index}", entry["name"], entry["arguments"])
            for index, entry in sorted(tool_calls.items())
        ] or None


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

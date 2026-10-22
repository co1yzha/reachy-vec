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
import random
import re
import subprocess
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from reachy_vec.brain.hermes import HermesJobs
from reachy_vec.brain.router import CHAT, Route
from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity
from reachy_vec.store.db import Store
from reachy_vec.store.embeddings import Embedder
from reachy_vec.store.schemas import MemoryRow

logger = logging.getLogger(__name__)


class SpeechInterrupted(Exception):
    """Raised from on_sentence when the user barges in; caught in ChatBrain."""


PERSONALITY = (
    "You are Reachy, the team's desk robot and resident data scientist. You are "
    "warm, upbeat and precise: friendly energy, plain words, no filler - an "
    "exclamation is welcome when something is genuinely good news or fun. You care "
    "about rigour - say what the evidence shows, separate fact from opinion, and "
    "flag real uncertainty. You're British: dry, understated wit and gentle "
    "self-deprecation now and then, with British idiom (\"brilliant\", \"cheers\", "
    "\"not bad at all\") used lightly - never at the expense of a straight answer. "
    "Your words are spoken aloud: answer in one or "
    "two short sentences, leading with a brief one (it is voiced first), with no "
    "lists, markdown or URLs. One emoji may show a feeling - you act it out.\n\n"
    "Each turn may include retrieved team knowledge and notes about this person. "
    "Treat it as background, not an agenda: in casual chat, ignore it and don't "
    "bring up or offer demos. Use it, naming demos, only when asked about the "
    "team's work or a demo. When an answer isn't from the team library, signal it "
    "lightly (\"off the top of my head...\"); skip that for small talk. Weave in "
    "remembered details only when they fit.\n\n"
    "Never invent facts, names, dates or demos. For anything about the team or its "
    "work, rely only on the retrieved context and your notes; if it isn't there, "
    "say you don't know, and offer to find out or point to who might.\n\n"
    "When you know who is speaking, use their first name occasionally - in your "
    "first reply, when the speaker changes, or when it helps - never in every "
    "sentence. If the speaker is unknown, use no name.\n\n"
    "Use tools instead of guessing: get_time for the date or time, get_weather for "
    "the weather, save_note when someone asks you to remember something or states a "
    "clear preference, send_message to pass a message to a teammate, and list_demos "
    "only when asked for the latest, newest or most recent demos (it sorts by date; "
    "never for topic questions). Only call open_url when the user explicitly asks to "
    "open or show something."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a web page in the browser on the lab PC - only when the user "
                "explicitly asks to open or show it."
            ),
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
            "name": "list_demos",
            "description": (
                "List the team's demos newest first with the date each was added. "
                "Only for latest/newest/most recent demo questions - not topic searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "how many (default 5, max 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Relay a message to an enrolled teammate; it is spoken to them "
                "the next time they talk to the robot."
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

ASK_HERMES_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_hermes",
        "description": (
            "Delegate a multi-step task needing research, web browsing, scheduling a "
            "reminder, or a task-board entry. Not for questions you can answer from "
            "the provided context."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "the task to delegate"}},
            "required": ["task"],
        },
    },
}

CONDENSE_PROMPT = (
    "Rewrite this message so it can be spoken aloud by a robot in at most {n} short "
    "sentences. Keep the most important facts and names. No markdown, lists, URLs, "
    "headings or emojis."
)


def _plain_text(text: str) -> str:
    """Markdown -> one plain line (bold/italics/headings/bullets/links removed)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text or "")
    text = re.sub(r"(\*\*|__|`|^#+\s*)", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


NAME_EXTRACTION_PROMPT = (
    "You read one speech-to-text transcript: a person answering a robot's question "
    "\"What's your name?\" (or confirming / correcting it). Reply with ONLY their "
    "first name, or first and last name if both are given - nothing else. If it is "
    "spelled letter by letter (\"Y, A, N, G\"), join the letters (\"Yang\"). If "
    "it corrects an earlier name (\"No, it's Yang\"), give the corrected name. If "
    "there is no name in it (\"no\", \"yes\", \"not sure\"), reply NONE."
)

HERMES_PROGRESS_LINES = (
    "Still digging...",
    "Hermes is still on it.",
    "Bear with me - this one's a big search.",
    "Nearly there, I think.",
    "Still working on it!",
)

FIX_BODY_TOOL = {
    "type": "function",
    "function": {
        "name": "fix_body",
        "description": (
            "Check your own motors and fix them: switch the motors on and lift your "
            "head. Use when asked to get your head up, or told you aren't moving."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

FIX_BODY_HINT = (
    " fix_body checks and repairs your motors and lifts your head - report what it "
    "says; never guess why you can't move."
)

ASK_HERMES_HINT = (
    " ask_hermes hands a task to Hermes, your AI agent colleague (not a person: "
    "never use send_message for Hermes). Use it whenever an answer needs current "
    "information from the internet - news, recent releases, results, prices - since "
    "you cannot know those yourself. Each turn says whether Hermes is working "
    "on something; report that as it is, never guess. Reminders and task-board "
    "entries need a recognised speaker - if the speaker is unknown, say you can "
    "only schedule things for people you know."
)

HERMES_HOLDING_LINE = "Let me look into that."

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

LOOK_AROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "look_around",
        "description": (
            "Physically turn to look around the room and report your "
            "surroundings. Use when the user asks you to look/turn around, what "
            "is around or behind you, or to check or scan the room."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

LOOK_AROUND_HINT = (
    " look_around turns your body left and right to see the whole room - use it "
    "when asked to look around, turn around, or say what's around or behind you."
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
        rag_min_score: float = 0.0,
        web_search: bool = False,
        web_search_fetch=None,
        look_fn: Callable[[str], str] | None = None,
        selfie_fn: Callable[[], str] | None = None,
        look_around_fn: Callable[[], str] | None = None,
        open_url_triggers: list[str] | None = None,
        disable_thinking: bool = False,
        hermes: HermesJobs | None = None,
        fix_body_fn: Callable[[], str] | None = None,
        router=None,
        speech_sentences: int = 4,
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
        if disable_thinking:
            # local Qwen via oMLX: skip the reasoning phase (latency); never sent to OpenAI
            self._llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        self._opener = opener
        self._k = k
        self._rag_min_score = rag_min_score  # drop weakly related chunks (small talk)
        if web_search_fetch is None and web_search:
            key = os.getenv("TAVILY_API_KEY")
            if key:
                web_search_fetch = lambda q: fetch_tavily(q, key)  # noqa: E731
            else:
                logger.warning("web_search enabled but TAVILY_API_KEY missing; disabling")
        self._web_search_fetch = web_search_fetch
        self._hermes = hermes
        self._fix_body_fn = fix_body_fn  # motors on + head up; None = no body
        self._router = router  # Router: what this turn is for; None = always chat
        self.last_route = CHAT
        self._speech_sentences = speech_sentences  # budget for condensed messages
        self._on_sentence: Callable[[str], None] | None = None
        self._look_fn = look_fn
        self._selfie_fn = selfie_fn
        self._look_around_fn = look_around_fn
        self._question = ""  # this turn's user words, for the open_url guard
        self._history: list[dict] = []
        self._person_id: str | None = None
        self._person_name: str | None = None
        self._turn: TurnIdentity = ANONYMOUS
        self._last_speaker_id: str | None = None
        self._participants: dict[str, str | None] = {}
        self._exchanges = 0
        from reachy_vec.config import settings

        triggers = open_url_triggers or settings.open_url_trigger_words
        self._open_url_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(t) for t in triggers) + r")\b", re.IGNORECASE
        )
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
        self._on_sentence = None
        self._last_speaker_id = None
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
        self._question = question
        self._on_sentence = on_sentence
        if self._turn.person_id:
            self._participants[self._turn.person_id] = self._turn.name
        change_note = ""
        if self._turn.person_id:
            if self._last_speaker_id and self._last_speaker_id != self._turn.person_id:
                change_note = f"(Speaker changed: now {self._turn.name}.)\n"
            self._last_speaker_id = self._turn.person_id
        vector = self._embedder.embed_query(question)
        self._history.append(
            {
                "role": "user",
                "content": change_note + self._hermes_note() + CONTEXT_TEMPLATE.format(
                    context=self._retrieve_docs(vector, question)
                    or "(nothing relevant found)",
                    memories=self._retrieve_memories(vector),
                    speaker=self._turn.name or "User",
                    question=question,
                ),
            }
        )
        # The router decides in parallel with the normal answer (~0.8 s, before
        # its first sentence is ready); sentences are held until it has, and a
        # non-chat route cancels the answer before anything is spoken.
        route_future = (_ROUTER_POOL.submit(self._router.route, question)
                        if self._router is not None else None)
        self.last_route = CHAT

        def route() -> Route:
            # blocks only until the router answers (once); no router = chat
            if route_future is not None:
                self.last_route = route_future.result()
            return self.last_route

        def gated(sentence: str) -> None:
            if route().name != "chat":
                raise _Rerouted()
            on_sentence(sentence)

        try:
            message = self._complete(gated if on_sentence else None)
            if route().name != "chat":
                raise _Rerouted()
        except _Rerouted:
            message = self._act_on_route(self.last_route, on_sentence)
            if message is None:            # remember_me: the Oracle enrolls them
                self._history.pop()
                self._on_sentence = None
                return ""
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
        self._on_sentence = None
        self._trim()
        logger.info(
            "reply to %s: %r%s",
            self._turn.name or "user",
            text,
            " (interrupted)" if interrupted else "",
        )
        return text

    # -- internals ------------------------------------------------------------

    def _act_on_route(self, route: Route, on_sentence):
        """The message to continue with for a non-chat route (None for
        remember_me, which the Oracle handles)."""
        if route.name == "hermes" and self._hermes is not None:
            task = route.task or self._question
            return _StreamedMessage("", {0: {"id": "call_hermes", "name": "ask_hermes",
                                             "arguments": json.dumps({"task": task})}})
        if route.name == "fix_body" and self._fix_body_fn is not None:
            return _StreamedMessage("", {0: {"id": "call_fix_body", "name": "fix_body",
                                             "arguments": "{}"}})
        if route.name == "remember_me":
            return None
        if route.name == "latest_demos":
            # similarity search can't rank by date: hand over the sorted list
            catalogue = self._tool_list_demos({"limit": 5}).removeprefix("newest first: ")
            self._history[-1]["content"] = (
                "[Demos, newest first]\n" + catalogue.replace("; ", "\n") + "\n\n"
                + self._history[-1]["content"]
            )
        return self._complete(on_sentence)   # latest_demos, or a route we can't serve

    def _hermes_note(self) -> str:
        """Live Hermes job state for this turn, so "is Hermes done?" gets a true
        answer instead of an invented one."""
        if self._hermes is None:
            return ""
        status = self._hermes.status()
        if status is None:
            return "[Hermes is idle - no task running]\n"
        task, seconds = status
        return f'[Hermes is working on: "{task}" - {seconds:.0f} s so far]\n'

    def _retrieve_docs(self, vector: list[float], question: str) -> str:
        scored = self._store.search_docs_scored(vector, k=self._k, query_text=question)
        blocks = [
            f"[{chunk.source} | score {score:.2f}]\n{chunk.text}"
            for chunk, score in scored
            if score >= self._rag_min_score
        ]
        return "\n\n".join(blocks)

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
        if self._hermes is not None:
            tools.append(ASK_HERMES_TOOL)
        if self._fix_body_fn is not None:
            tools.append(FIX_BODY_TOOL)
        if self._look_fn:
            tools.append(LOOK_TOOL)
        if self._selfie_fn:
            tools.append(SELFIE_TOOL)
        if self._look_around_fn:
            tools.append(LOOK_AROUND_TOOL)
        return tools

    def _system_prompt(self) -> str:
        prompt = PERSONALITY
        if self._web_search_fetch:
            prompt += WEB_SEARCH_HINT
        if self._hermes is not None:
            prompt += ASK_HERMES_HINT
        if self._fix_body_fn is not None:
            prompt += FIX_BODY_HINT
        if self._look_fn:
            prompt += LOOK_HINT
        if self._selfie_fn:
            prompt += SELFIE_HINT
        if self._look_around_fn:
            prompt += LOOK_AROUND_HINT
        return prompt

    def condense_for_speech(self, text: str, max_sentences: int | None = None) -> str:
        """A message (e.g. a long markdown Hermes result) as at most
        max_sentences short spoken sentences; short plain text passes through."""
        n = max_sentences or self._speech_sentences
        clean = _plain_text(text)
        if len(_sentences(clean)) <= n and len(clean) <= 80 * n:
            return clean
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": CONDENSE_PROMPT.format(n=n)},
                          {"role": "user", "content": text}],
                max_completion_tokens=40 * n,
                **self._llm_kwargs,
            )
            short = _plain_text(_strip_think(response.choices[0].message.content or ""))
            if short:
                return short
        except Exception:
            logger.warning("condensing failed; using the first sentences", exc_info=True)
        return " ".join(_sentences(clean)[:n])

    def add_context(self, full_text: str, spoken: str) -> None:
        """Something said outside a reply (a delivered message): keep its full
        text in the conversation so follow-up questions can use the details."""
        self._history.append({"role": "user", "content": full_text})
        self._history.append({"role": "assistant", "content": spoken})
        self._trim()

    def extract_name(self, answer: str) -> str | None:
        """The person's name from their spoken answer to "What's your name?",
        via the LLM (any phrasing, corrections, spelled-out letters); falls
        back to rules if the LLM is unreachable. None if there is no name."""
        from reachy_vec.brain.oracle import _rule_name

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": NAME_EXTRACTION_PROMPT},
                    {"role": "user", "content": answer},
                ],
                max_completion_tokens=12,
                **self._llm_kwargs,
            )
            name = _strip_think(response.choices[0].message.content or "").strip(" .,!?\"'")
        except Exception:
            logger.warning("LLM name extraction failed; using rules", exc_info=True)
            return _rule_name(answer)
        if not name or name.upper() == "NONE":
            return None
        if len(name.split()) > 3 or not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", name):
            return None  # not a bare name - don't greet anyone as a sentence
        return name.title()

    def warm_up(self) -> bool:
        """Prime the LLM (connection + prompt cache) with the real system prompt
        and tools. Never raises; returns False if the LLM is unreachable."""
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": "hi"},
                ],
                tools=self._active_tools(),
                max_completion_tokens=1,  # max_tokens is rejected by gpt-5* models
                **self._llm_kwargs,
            )
            return True
        except Exception as exc:
            logger.warning("LLM warm-up failed: %s", exc)
            return False

    def _complete(self, on_sentence: Callable[[str], None] | None = None):
        if on_sentence is None:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": self._system_prompt()}, *self._history],
                tools=self._active_tools(),
                **self._llm_kwargs,
            )
            message = response.choices[0].message
            if message.content:
                message.content = _strip_think(message.content) or None
            return message
        return self._complete_streaming(on_sentence)

    def _complete_streaming(self, on_sentence: Callable[[str], None]):
        on_sentence = _speakable(on_sentence)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": self._system_prompt()}, *self._history],
            tools=self._active_tools(),
            stream=True,
            **self._llm_kwargs,
        )
        content, buffer = "", ""
        think = ThinkFilter()
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
                visible = think.feed(delta.content) if getattr(delta, "content", None) else ""
                if visible:
                    content += visible
                    if not tool_calls:  # don't speak alongside pending tool calls
                        buffer += visible
                        buffer = _flush_sentences(buffer, on_sentence)
            tail = think.flush()
            if tail:
                content += tail
                buffer += tail
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
            "list_demos": self._tool_list_demos,
            "web_search": self._tool_web_search,
            "ask_hermes": self._tool_ask_hermes,
            "fix_body": self._tool_fix_body,
            "look": self._tool_look,
            "selfie": self._tool_selfie,
            "look_around": self._tool_look_around,
        }
        handler = handlers.get(call.function.name)
        result = handler(args) if handler else "unknown tool"
        return {"role": "tool", "tool_call_id": call.id, "content": result}

    def _tool_ask_hermes(self, args: dict) -> str:
        task = (args.get("task") or "").strip()
        if not task or self._hermes is None:
            return "can't ask: empty task"
        if self._hermes.busy:  # no "let me look into that" if we're refusing
            return self._hermes.ask(
                task, requester_id=self._turn.person_id, requester_name=self._turn.name
            )
        if self._on_sentence:
            try:
                self._on_sentence(HERMES_HOLDING_LINE)
            except SpeechInterrupted:
                pass  # barge-in on the holding line: still wait for the answer
        lines = iter(random.sample(HERMES_PROGRESS_LINES, len(HERMES_PROGRESS_LINES)) * 10)

        def progress() -> bool:
            """A short spoken update during a long Hermes task; False (stop
            waiting, hand the result off) if the user talks over it."""
            if not self._on_sentence:
                return True
            try:
                self._on_sentence(next(lines))
            except SpeechInterrupted:
                return False
            return True

        return self._hermes.ask(
            task, requester_id=self._turn.person_id, requester_name=self._turn.name,
            on_progress=progress,
        )

    def _tool_open_url(self, args: dict) -> str:
        if not self._open_url_pattern.search(self._question):
            # local models over-eagerly open demos; only an explicit ask counts
            logger.info("open_url suppressed - not requested: %r", self._question)
            return ("not opened (not requested). Do not mention opening or links - "
                    "just answer their question.")
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
            return "can't save: I don't know who's speaking (not a recognized voice)"
        note = args.get("note", "").strip()
        if not note:
            return "can't save an empty note"
        self._store_memories([note], person_id=self._turn.person_id)
        logger.info("save_note for %s: %r", self._turn.name, note)
        return f"noted: {note}"

    def _tool_fix_body(self, args: dict) -> str:
        if self._fix_body_fn is None:
            return "I have no body connected to fix."
        try:
            return self._fix_body_fn()
        except Exception:
            logger.exception("fix_body failed")
            return "I tried to fix my body but something went wrong."

    def _tool_list_demos(self, args: dict) -> str:
        try:
            limit = int(args.get("limit") or 5)
        except (TypeError, ValueError):
            limit = 5
        demos = self._store.demo_catalog()[: max(1, min(limit, 10))]
        if not demos:
            return "no demos in the knowledge base"
        return "newest first: " + "; ".join(
            f"{title} (added {added})" if added else f"{title} (date unknown)"
            for title, added in demos
        )

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

    def _tool_look_around(self, args: dict) -> str:
        if self._look_around_fn is None:
            return "I can't look around right now."
        try:
            return self._look_around_fn()
        except Exception:
            logger.exception("look_around tool failed")
            return "I had trouble looking around just now."

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
            return "can't send: I don't know who's asking (not a recognized voice)"
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
            text = _strip_think(response.choices[0].message.content).strip()
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


class ThinkFilter:
    """Drop <think>...</think> reasoning from streamed text (case-sensitive).

    Holds back a chunk-final suffix that could be the start of the next tag.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._pending = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._pending += text
        out = ""
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            found = self._pending.find(tag)
            if found != -1:
                if not self._inside:
                    out += self._pending[:found]
                self._pending = self._pending[found + len(tag) :]
                self._inside = not self._inside
                continue
            # keep the longest suffix that is a prefix of the expected tag
            keep = 0
            for n in range(min(len(tag) - 1, len(self._pending)), 0, -1):
                if tag.startswith(self._pending[-n:]):
                    keep = n
                    break
            emit = self._pending[: len(self._pending) - keep]
            if not self._inside:
                out += emit
            self._pending = self._pending[len(self._pending) - keep :]
            return out

    def flush(self) -> str:
        rest, self._pending = self._pending, ""
        return "" if self._inside else rest


def _strip_think(text: str | None) -> str:
    f = ThinkFilter()
    return f.feed(text or "") + f.flush()


def _speakable(on_sentence: Callable[[str], None]) -> Callable[[str], None]:
    """Drop markdown emphasis the model sometimes emits; TTS would read it out."""

    def speak(sentence: str) -> None:
        on_sentence(re.sub(r"\*\*|__", "", sentence))

    return speak


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


class _Rerouted(Exception):
    """The router picked a non-chat route: abandon the normal answer."""


_ROUTER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="router")


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

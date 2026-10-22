"""Turn router: one short LLM call decides what a spoken turn is for.

Replaces keyword rules (English-only, blind to phrasing) for the decisions the
local model made badly as tool choices inside the answer: delegating to Hermes
(incl. anything needing live information), listing the newest demos, repairing
the body, and voice enrollment. Anything else - or any failure - is "chat",
the normal answer with all the usual tools.
"""

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ROUTES = ("chat", "hermes", "latest_demos", "fix_body", "remember_me")


@dataclass(frozen=True)
class Route:
    name: str = "chat"
    task: str = ""  # for "hermes": the task, phrased for Hermes


CHAT = Route()

ROUTER_PROMPT = """You route what someone said to Reachy, a desk robot assistant, to one action.
Reply with ONLY the route name. For hermes, add a colon and the task: "hermes: <task>".

Routes:
- hermes: a multi-step task for Hermes, the AI agent colleague (research, browsing,
  summarising web pages, reminders, scheduling, task-board entries), OR anything that
  needs current information from the internet: news, today's events, recent releases,
  results, scores, prices. Also whenever they explicitly ask Hermes to do something.
  Write the task without the words "ask Hermes".
- latest_demos: which team demos are newest / most recent / latest added.
- fix_body: the robot's own body - lift its head, its motors, it is not moving or stuck.
- remember_me: they want the robot to remember / learn their voice or name.
- chat: everything else - conversation, questions about the team's demos or work,
  general knowledge, weather, time, notes, messages to teammates, opening a page,
  looking through the camera, taking a photo.

Examples:
"what's the latest AI news" -> hermes: find today's top AI news
"ask Hermes to research the SDK" -> hermes: research the SDK
"what's the newest demo" -> latest_demos
"can you get your head up" -> fix_body
"I was reaching for my coffee" -> chat
"remember my voice" -> remember_me
"tell Bob the meeting moved" -> chat
"any demos about food" -> chat
"what's the weather" -> chat"""


class Router:
    """route(text) -> Route via one non-streaming completion (thinking off,
    a few tokens). `client` is the OpenAI-compatible LLM client."""

    def __init__(self, client, model: str, llm_kwargs: dict | None = None):
        self._client = client
        self._model = model
        self._llm_kwargs = llm_kwargs or {}

    def route(self, text: str) -> Route:
        if not text.strip():
            return CHAT
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": ROUTER_PROMPT},
                          {"role": "user", "content": text}],
                max_completion_tokens=40,  # "chat" is 1 token; a hermes task a few more
                **self._llm_kwargs,
            )
            raw = response.choices[0].message.content or ""
        except Exception:
            logger.warning("router call failed; answering normally", exc_info=True)
            return CHAT
        route = parse_route(raw)
        logger.info("route %s%s for %r", route.name,
                    f" ({route.task!r})" if route.task else "", text)
        return route


def parse_route(raw: str) -> Route:
    """"chat" / "fix_body" / "hermes: <task>" (JSON from older prompts is also
    accepted); unknown or malformed -> chat."""
    from reachy_vec.brain.chat import _strip_think

    text = _strip_think(raw).strip().strip("`").strip()
    if "{" in text:
        try:
            data = json.loads(re.search(r"\{.*?\}", text, re.DOTALL).group(0))
            text = f"{data.get('route', '')}: {data.get('task') or ''}"
        except (ValueError, AttributeError):
            return CHAT
    head, _, task = text.partition(":")
    name = head.strip().strip("\"'.").lower()
    if name not in ROUTES:
        return CHAT
    return Route(name, task.strip().strip("\"'") if name == "hermes" else "")

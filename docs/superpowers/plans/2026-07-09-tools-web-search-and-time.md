# Tools: `web_search` (Tavily) + `get_time` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Reachy two new tools — `get_time` (always on, no cost) and `web_search` (Tavily, opt-in, credit-aware) — so it can answer "what time is it?" and ground "current info" questions on the live web, while degrading gracefully and audibly when Tavily credits run out.

**Architecture:** Both follow the existing tool pattern in `brain/chat.py` (schema in `TOOLS`, a `_tool_x` handler, a line in the `_execute_tool` dispatch, a mention in the system prompt). `get_time` is pure `datetime`. `web_search` POSTs to `https://api.tavily.com/search` (raw `urllib`, no new dependency), uses the `answer` field for a spoken-ready reply, and maps Tavily's HTTP error codes to distinct spoken messages — **432/433 (out of credits) → an explicit "I've used up my web-search allowance" message** the model relays to the user. `web_search` is gated: offered only when `REACHY_VEC_WEB_SEARCH=true` **and** `TAVILY_API_KEY` is set, so it never burns credits unless deliberately enabled. The tool result string is fed back to the LLM, which speaks it — so "let the user know" happens through the normal reply path (same mechanism as the weather tool's error strings).

**Tech Stack:** Python 3.12, `urllib` (stdlib), Tavily Search API, pytest, ruff.

## Global Constraints

- Python **3.12+**; **no new third-party dependency** — use `urllib`, not the `tavily-python` SDK (mirrors `fetch_open_meteo`).
- `TAVILY_API_KEY` is a **secret with no prefix** (read via `os.getenv`, same convention as `OPENAI_API_KEY`/`MONGODB_URI`) — never a `REACHY_VEC_`-prefixed setting, never logged.
- **Credit-thrift is a requirement:** `web_search` off by default; the system prompt tells the model to use it *only* for current/live info it can't answer from the team library or general knowledge.
- **Every network call has a 5 s timeout** and never raises out of the handler — all failures return a spoken-friendly string.
- Tool output is **spoken aloud**: return one short sentence; never dump the `results` list.
- Tests run **offline**: inject a fake `web_search_fetch`; never hit Tavily or read a real key.
- After changes: `uv run ruff check src tests` and `uv run pytest -q` must both pass.

---

### Task 1: `get_time` tool

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Produces: a `get_time` tool that returns the current local date/time as a spoken sentence. No config, no network.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_chat_brain.py
def test_get_time_tool_returns_local_time(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient())
    result = brain._tool_get_time({})
    assert "the time is" in result.lower()
    # contains a HH:MM-ish clock value
    import re
    assert re.search(r"\d{1,2}:\d{2}", result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_brain.py -k get_time -v`
Expected: FAIL — `ChatBrain` has no `_tool_get_time`.

- [ ] **Step 3: Implement in `chat.py`**

Add the schema to `TOOLS` (after `get_weather`):

```python
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
```

Register it in `_execute_tool`'s `handlers` dict:

```python
            "get_weather": self._tool_get_weather,
            "get_time": self._tool_get_time,
```

Add the handler (near `_tool_get_weather`):

```python
    def _tool_get_time(self, args: dict) -> str:
        now = datetime.now().astimezone()
        return f"the time is {now:%A %d %B %Y, %H:%M} ({now.tzname()})"
```

Mention it in `PERSONALITY` — extend the tools sentence:

```python
        "get_weather checks the live weather outside the lab; "
        "get_time tells the current local date and time."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat_brain.py -k get_time -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: get_time tool"
```

---

### Task 2: `web_search` tool (Tavily) with credit-aware handling

**Files:**
- Modify: `src/reachy_vec/config.py` (add `web_search`)
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Consumes: `TAVILY_API_KEY` (env, via `os.getenv`); `settings.web_search` (new bool).
- Produces:
  - `config.py`: `web_search: bool = False`.
  - `chat.py`: module-level `WEB_SEARCH_TOOL` schema and `WEB_SEARCH_HINT` prompt fragment; `fetch_tavily(query, api_key, max_results=3, timeout=5.0) -> str` returning the `answer` field.
  - `ChatBrain.__init__(..., web_search: bool = False, web_search_fetch=None)` — resolves an effective fetch: if `web_search_fetch` is given, use it (tests); else if `web_search` and a key is present, build `lambda q: fetch_tavily(q, key)`; else `None` (disabled, logs a warning if the flag was set but the key was missing). `self._web_search_fetch is not None` ⇒ the tool is offered and the hint is appended.
  - `_tool_web_search(args)` — maps Tavily failures to spoken strings; **`HTTPError` 432/433 → an explicit out-of-credits message**.
  - `_complete`/`_complete_streaming` build their tool list and system prompt from `self._web_search_fetch`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_chat_brain.py
import io
import urllib.error


def _http_error(code):
    return urllib.error.HTTPError("https://api.tavily.com/search", code, "err", {}, io.BytesIO(b""))


def test_web_search_returns_answer(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=lambda q: "Paris is the capital.")
    assert brain._tool_web_search({"query": "capital of France"}) == "Paris is the capital."


def test_web_search_out_of_credits_tells_user(tmp_path):
    def boom(q):
        raise _http_error(432)

    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=boom)
    msg = brain._tool_web_search({"query": "latest news"})
    assert "credit" in msg.lower() or "allowance" in msg.lower()


def test_web_search_paygo_limit_also_out_of_credits(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=lambda q: (_ for _ in ()).throw(_http_error(433)))
    assert "allowance" in brain._tool_web_search({"query": "x"}).lower()


def test_web_search_rate_limited_is_distinct(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=lambda q: (_ for _ in ()).throw(_http_error(429)))
    msg = brain._tool_web_search({"query": "x"}).lower()
    assert "moment" in msg or "rate" in msg
    assert "allowance" not in msg  # not confused with credit exhaustion


def test_web_search_tool_offered_only_when_enabled(tmp_path):
    off = make_brain(tmp_path, FakeLLMClient())
    on = make_brain(tmp_path, FakeLLMClient(), web_search_fetch=lambda q: "ok")
    assert not any(t["function"]["name"] == "web_search" for t in off._active_tools())
    assert any(t["function"]["name"] == "web_search" for t in on._active_tools())
```

Update the `make_brain` helper at the top of the file to forward the new kwargs:

```python
def make_brain(tmp_path, client, opener=None, web_search_fetch=None):
    return ChatBrain(
        store=seeded_store(tmp_path),
        embedder=FakeEmbedder(),
        client=client,
        model="gpt-4o",
        opener=opener or (lambda url: None),
        web_search_fetch=web_search_fetch,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_brain.py -k web_search -v`
Expected: FAIL — no `_tool_web_search` / `_active_tools` / `web_search_fetch` kwarg.

- [ ] **Step 3: Add the config knob**

In `config.py`, under `# Weather` or a new tools group:

```python
    web_search: bool = False  # enable the Tavily web_search tool (needs TAVILY_API_KEY)
```

- [ ] **Step 4: Implement in `chat.py`**

Add `import os` at the top. Add the schema and hint near `TOOLS`:

```python
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
    " web_search looks things up on the live web — use it only when the user wants "
    "current or up-to-the-minute info you can't give from the team library or your "
    "own knowledge (it costs a limited search budget, so don't use it for chit-chat)."
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
```

In `ChatBrain.__init__`, add params and resolve the fetch (place after `self._opener = opener`):

```python
        web_search: bool = False,
        web_search_fetch=None,
```

```python
        if web_search_fetch is None and web_search:
            key = os.getenv("TAVILY_API_KEY")
            if key:
                web_search_fetch = lambda q: fetch_tavily(q, key)  # noqa: E731
            else:
                logger.warning("web_search enabled but TAVILY_API_KEY missing; disabling")
        self._web_search_fetch = web_search_fetch
```

Add a helper that both `_complete` paths use, and the system-prompt builder:

```python
    def _active_tools(self) -> list:
        return [*TOOLS, WEB_SEARCH_TOOL] if self._web_search_fetch else TOOLS

    def _system_prompt(self) -> str:
        return PERSONALITY + WEB_SEARCH_HINT if self._web_search_fetch else PERSONALITY
```

Replace the two `tools=TOOLS` and the two `{"role": "system", "content": PERSONALITY}` occurrences in `_complete` and `_complete_streaming` with `tools=self._active_tools()` and `{"role": "system", "content": self._system_prompt()}`. (The `_summarize_and_store` system message can stay `PERSONALITY` — no tools there.)

Register and implement the handler:

```python
            "get_time": self._tool_get_time,
            "web_search": self._tool_web_search,
```

```python
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
                    "that up — tell the user plainly."
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_brain.py -v`
Expected: PASS (existing + new)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/config.py src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: web_search tool (Tavily) with credit-aware graceful failure"
```

---

### Task 3: CLI wiring + docs

**Files:**
- Modify: `src/reachy_vec/cli/run.py`, `src/reachy_vec/cli/chat.py` (pass `web_search=settings.web_search`)
- Modify: `docs/pipelines.md` (tools table), `docs/configuration.md`, `.env.example`
- Test: covered by Task 2 (no new code paths; CLI wiring verified by the existing `test_cli.py` smoke of `chat`/`run` construction if present, else manual)

**Interfaces:**
- Consumes: `settings.web_search`, `ChatBrain(web_search=...)` (Task 2).

- [ ] **Step 1: Wire both CLI entry points**

In `cli/chat.py`, the `ChatBrain(...)` construction — add:

```python
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
```

In `cli/run.py`, the `ChatBrain(...)` construction — add the same `web_search=settings.web_search,` line.

- [ ] **Step 2: Docs**

- `docs/pipelines.md` → the **Tools** table: add
  - `get_time` | current local date/time | none
  - `web_search` | live web answer via Tavily (`answer` field) | off unless `WEB_SEARCH=true` + `TAVILY_API_KEY`; 5 s timeout; out-of-credits (Tavily 432/433) → spoken "used up my allowance"
- `docs/configuration.md` → add `TAVILY_API_KEY` to the **Secrets** table ("web_search tool; free tier is limited — the tool tells the user when credits run out") and `WEB_SEARCH` (`false`) to a tools/interaction section.
- `.env.example` → add:

```bash
# --- Web search tool (optional; limited free credits) ---
# REACHY_VEC_WEB_SEARCH=true
# TAVILY_API_KEY=tvly-...
```

- [ ] **Step 3: Full suite + lint, then commit**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass

```bash
git add -A
git commit -m "feat: wire web_search into chat/run + docs"
```

---

## Self-Review

**Coverage of the ask:**
- `get_time` → Task 1. ✓
- `web_search` via Tavily, `answer` field spoken, no new dep → Task 2. ✓
- **Credit exhaustion tells the user** → Task 2 maps Tavily 432/433 to an explicit spoken message, tested (`test_web_search_out_of_credits_tells_user`, `..._paygo_limit_...`), and kept distinct from 429 rate-limiting. ✓
- Credit thrift → off by default, key required, prompt restricts usage to genuine current-info needs. ✓

**Design notes / deliberate choices:**
- Tool result strings are phrased as instructions/statements the model relays (e.g. "...tell the user plainly") — the model composes the actual spoken sentence, matching how `get_weather` errors already surface. If we wanted the exact wording guaranteed, the Oracle would speak it directly; that's a bigger change and not warranted here.
- `TAVILY_API_KEY` read via `os.getenv` (no `REACHY_VEC_` prefix), consistent with `OPENAI_API_KEY`; never logged.
- No new dependency (raw `urllib`), consistent with `fetch_open_meteo`.

**Placeholder scan:** none — every code step is complete; test steps show assertions.

**Type consistency:** `fetch_tavily(query, api_key, max_results, timeout) -> str` (Task 2) is wrapped by the injected/default `web_search_fetch: Callable[[str], str]`; `_active_tools()`/`_system_prompt()` gate on `self._web_search_fetch`; `ChatBrain(web_search=..., web_search_fetch=...)` kwargs match `make_brain` (Task 2) and the CLI call sites (Task 3). `_tool_get_time`/`_tool_web_search` names match their `handlers` dict entries.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-tools-web-search-and-time.md`.

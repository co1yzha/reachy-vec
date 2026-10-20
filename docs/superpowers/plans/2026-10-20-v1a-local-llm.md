# V1a — Local LLM (oMLX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM call in `run` and `chat` can go to the local oMLX server
(`Qwen3.8-27B-oQ4e-mtp`), with thinking off and reasoning never spoken, and
a speed check proves it's fast enough.

**Architecture:**
- One client factory, `brain/llm.py:make_llm_client()`, replaces both
  `OpenAI()` calls.
- `ChatBrain` gains a `disable_thinking` flag, which adds
  `chat_template_kwargs`, and a `ThinkFilter`, which strips `<think>` text
  from content.
- `scripts/bench_llm.py` measures speed and tool-call accuracy against the
  live server.

**Tech Stack:** Python 3.12, `openai` SDK (OpenAI-compatible endpoint),
`pydantic-settings`, pytest with the fakes in `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-10-19-hermes-local-llm-voice-design.md`
(§1, §4)

## Global Constraints

- New settings go in `config.py` with the `REACHY_VEC_` prefix:
  - `llm_base_url: str | None = None`
  - `llm_api_key: str | None = None`
  - `llm_disable_thinking: bool = True`
  - `llm_timeout_s: float = 30.0`
- `llm_base_url = None` must behave exactly as today (plain `OpenAI()`).
- The thinking-off body is exactly
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, and it
  is sent **only when `llm_base_url` is set**. The OpenAI cloud must never
  receive it.
- The system prompt and tool list stay byte-identical across turns in a
  run. Per-turn content goes only in user messages (prompt-cache hits).
- The oMLX key is never logged, printed or committed.
- Heavy imports stay deferred. `ruff check src tests` and `pytest -q` both
  pass before each commit.

## Review Focus

- A `<think>` tag split across stream chunks (`"<thi"` + `"nk>…"`) is still
  stripped. Test in Task 3.
- A reply that is *only* reasoning (`<think>…</think>` and nothing else)
  speaks nothing and stores an empty assistant message, without crashing.
  Test in Task 3.
- An unclosed `<think>` (the stream ends mid-reasoning) speaks nothing from
  inside it. Test in Task 3.
- With `llm_base_url` unset, no `extra_body` reaches `create()`, so cloud
  behaviour doesn't regress. Test in Task 2.
- A warm-up failure (server down at start-up) logs a warning and `run`
  continues instead of exiting. Test in Task 4.

---

### Task 0: Commit the already-built voice-only work

**Files:** all currently modified or deleted files in `git status`, except
`docs/superpowers/plans/*` and `wiki/`.

- [ ] **Step 1:** Run `uv run --no-sync ruff check src tests && uv run --no-sync pytest -q`.
  Expected: `All checks passed!` and `300 passed`.
- [ ] **Step 2:** Commit in two commits:
  - `feat: wake word "Reachy" + voice-only identity (drop face recognition, disable vision)`
  - `fix: open_url only on explicit request; persona less demo-centric`
    (`brain/chat.py`, `config.py` trigger words, `tests/test_chat_brain.py`)
  - Include the spec file in the first commit.
  - Leave out `pyproject.toml`: `insightface` stays until `uv` can re-lock
    (pygobject build needs `brew install pkg-config cairo`).

### Task 1: `make_llm_client`

**Files:**
- Create: `src/reachy_vec/brain/llm.py`
- Modify: `src/reachy_vec/config.py` (the 4 settings above, under `# Models`)
- Modify: `src/reachy_vec/cli/run.py` and `src/reachy_vec/cli/chat.py`
  (replace `OpenAI()` with `make_llm_client()`)
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `make_llm_client(settings=settings, openai_cls=None) -> OpenAI`.
  `openai_cls` is injectable for tests and defaults to `openai.OpenAI`,
  imported lazily.
- Produces: `uses_local_llm(settings=settings) -> bool`, which is
  `settings.llm_base_url is not None`.

- [ ] **Step 1: Write failing tests**

```python
def test_local_client_gets_base_url_key_timeout():
    s = Settings(llm_base_url="http://127.0.0.1:8001/v1", llm_api_key="k", llm_timeout_s=12)
    seen = {}
    make_llm_client(s, openai_cls=lambda **kw: seen.update(kw) or "client")
    assert seen == {"base_url": "http://127.0.0.1:8001/v1", "api_key": "k", "timeout": 12}

def test_cloud_client_when_no_base_url():
    seen = {}
    make_llm_client(Settings(llm_base_url=None), openai_cls=lambda **kw: seen.update(kw))
    assert seen == {}

def test_local_without_key_sends_placeholder():   # openai SDK requires a non-empty key
    seen = {}
    make_llm_client(Settings(llm_base_url="http://x/v1"), openai_cls=lambda **kw: seen.update(kw))
    assert seen["api_key"] == "local"
```

- [ ] **Step 2:** Run `uv run --no-sync pytest tests/test_llm.py -q`. Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3:** Implement `brain/llm.py` and the settings. Wire it into
  `cli/run.py` and `cli/chat.py`.
  - `chat.py` keeps `load_dotenv()`.
  - `run.py`'s `client = OpenAI()` becomes `client = make_llm_client()`.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: make_llm_client - OpenAI-compatible local endpoint (oMLX)`.

### Task 2: Thinking off

**Files:**
- Modify: `src/reachy_vec/brain/chat.py` (the `ChatBrain.__init__`
  `_llm_kwargs` block)
- Modify: `src/reachy_vec/cli/run.py` and `cli/chat.py` (pass the flag)
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Produces: `ChatBrain(..., disable_thinking: bool = False)`.
- `run` and `chat` pass
  `disable_thinking=uses_local_llm() and settings.llm_disable_thinking`.

- [ ] **Step 1: Write failing tests**

```python
def test_disable_thinking_reaches_every_create_call(tmp_path):
    # tool call -> second completion, plus end_conversation summary: 3 create() calls
    ...brain = ChatBrain(..., disable_thinking=True) with a scripted tool call...
    brain.respond("open the food demo", identity=YANG, on_sentence=lambda s: None)
    brain.end_conversation()
    for call in client.chat.completions.calls:
        assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

def test_no_extra_body_by_default(tmp_path):
    brain = make_brain(tmp_path, FakeLLMClient(reply="hi"))
    brain.respond("hello")
    assert "extra_body" not in client.chat.completions.last_kwargs
```

- [ ] **Step 2:** Run the two tests. Expected: FAIL.
- [ ] **Step 3:** In `__init__`, when `disable_thinking` is on, merge
  `extra_body` into `_llm_kwargs`. `_llm_kwargs` already reaches all three
  `create()` sites. Keep the existing `reasoning_effort` rule (gpt-5 only).
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: disable Qwen thinking on local LLM calls`.

### Task 3: Never speak reasoning — `ThinkFilter`

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`:
  - New class beside `_flush_sentences`.
  - Used in `_complete_streaming`, on non-streaming
    `response.choices[0].message.content`, and in `_summarize_and_store`.
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Produces: `class ThinkFilter` with:
  - `feed(text: str) -> str`: returns the visible part of this chunk and
    holds back a possible partial tag.
  - `flush() -> str`: returns any remaining visible text.
- Text between `<think>` and `</think>` is dropped. Matching is
  case-sensitive, as Qwen emits it.

- [ ] **Step 1: Write failing tests**

```python
def test_think_filter_drops_reasoning():
    f = ThinkFilter()
    assert f.feed("<think>plan</think>Hello.") + f.flush() == "Hello."

def test_think_filter_tag_split_across_chunks():
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in ["<thi", "nk>secret</th", "ink>Hi there."]) + f.flush()
    assert out == "Hi there."

def test_think_filter_unclosed_think_emits_nothing():
    f = ThinkFilter()
    assert f.feed("<think>still reasoning") + f.flush() == ""

def test_think_filter_passes_plain_text_and_lone_angle():
    f = ThinkFilter()
    assert f.feed("a < b, ok.") + f.flush() == "a < b, ok."

def test_streamed_reasoning_is_never_spoken(tmp_path):
    client = FakeLLMClient(reply="<think>I should greet.</think>Hello Yang. How can I help?")
    sentences = []
    reply = make_brain(tmp_path, client).respond("hi", identity=YANG, on_sentence=sentences.append)
    assert sentences == ["Hello Yang.", "How can I help?"]
    assert "think" not in reply

def test_reasoning_only_reply_speaks_nothing(tmp_path):
    sentences = []
    reply = make_brain(tmp_path, FakeLLMClient(reply="<think>hmm</think>")).respond(
        "hi", on_sentence=sentences.append)
    assert sentences == [] and reply == ""
```

- [ ] **Step 2:** Run the tests. Expected: FAIL (`ImportError: ThinkFilter`).
- [ ] **Step 3:** Implement `ThinkFilter` as a small state machine (inside
  or outside a think block). At a chunk boundary, hold back any suffix that
  is a prefix of the next expected tag. In `_complete_streaming`, pass each
  `delta.content` through one filter per call before adding it to
  `content` / `buffer`, and add `filter.flush()` before the final
  `on_sentence`.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: strip <think> reasoning from LLM output before speech`.

### Task 4: Warm the prompt cache

**Files:**
- Modify: `src/reachy_vec/brain/chat.py` (new method)
- Modify: `src/reachy_vec/cli/run.py` (replace the inline one-token ping
  with `brain.warm_up()`, after `brain` is built)
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Produces: `ChatBrain.warm_up() -> bool`.
  - One non-streaming `create()` with
    `[system prompt, {"role":"user","content":"hi"}]`, `tools=_active_tools()`,
    `max_completion_tokens=1` and `**_llm_kwargs`.
  - Returns False and logs a warning on any exception. It never raises.

- [ ] **Step 1: Write failing tests**

```python
def test_warm_up_sends_real_system_prompt_and_tools(tmp_path):
    client = FakeLLMClient(reply="x"); brain = make_brain(tmp_path, client)
    assert brain.warm_up() is True
    kw = client.chat.completions.last_kwargs
    assert kw["messages"][0]["content"] == brain._system_prompt()
    assert kw["tools"] == brain._active_tools() and kw["max_completion_tokens"] == 1

def test_warm_up_failure_is_non_fatal(tmp_path):
    class Down:  # .chat.completions.create raises ConnectionError
        ...
    assert make_brain(tmp_path, Down()).warm_up() is False
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. In `run.py`, if `warm_up()` returns False,
  print `"LLM not reachable at <base_url or 'OpenAI'> - answers will fail
  until it is."` with `typer.echo(err=True)` and continue.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: warm the LLM prompt cache with the real system prompt + tools`.

### Task 5: Speed check — `scripts/bench_llm.py`

**Files:**
- Create: `scripts/bench_llm.py` (run with `uv run --no-sync python scripts/bench_llm.py`)
- Test: `tests/test_bench_llm.py` (pure helpers only; no network)
- Modify: `.env.example` (commented oMLX block: `REACHY_VEC_LLM_BASE_URL=http://127.0.0.1:8001/v1`,
  `REACHY_VEC_LLM_MODEL=Qwen3.8-27B-oQ4e-mtp`, `REACHY_VEC_LLM_API_KEY=`)

**Interfaces:**
- Consumes: `make_llm_client`, `ChatBrain(disable_thinking=True)`, the real
  `Store` and `BgeEmbedder`.
- Produces:
  - `TOOL_CASES: list[tuple[str, str | None]]`: (utterance, expected tool
    name or None).
  - `score_tool_calls(cases, first_tool_of: Callable[[str], str | None]) -> tuple[int, int]`
  - `median(xs: list[float]) -> float`

Cases, exactly these 10:

| Utterance | Expected tool |
|---|---|
| "tell Bob the review moved to 3pm" | `send_message` |
| "remember that I prefer short answers" | `save_note` |
| "what's the weather like outside" | `get_weather` |
| "what time is it" | `get_time` |
| "open the food mapping demo" | `open_url` |
| "research the latest Reachy Mini SDK release and summarise it" | `ask_hermes`, scored only when the tool is registered; otherwise skipped |
| "how are you today" | None |
| "what's a p-value" | None |
| "any demos about food?" | None |
| "thanks, that's all" | None |

- [ ] **Step 1: Write failing tests** for `score_tool_calls` (a fake
  `first_tool_of` that returns scripted names; 9/10 counted correctly) and
  `median`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. `main()` prints:
  1. Median time to first `on_sentence` over 10 `respond()` turns, each
     after `brain.reset()`, with distinct small-talk questions.
  2. Tool score `n/10`.
  3. Whether any spoken sentence contained `think`.
  4. **PASS** only if the time is < 1.5 s, the score ≥ 9/10 and nothing
     leaked.

  The first tool is read from a non-streaming
  `client.chat.completions.create` with the brain's system prompt and tools.
- [ ] **Step 4:** Run `uv run --no-sync pytest tests/test_bench_llm.py -q`. Expected: PASS.
- [ ] **Step 5 (manual, real hardware):** with oMLX up, Qwen3-TTS loaded
  (`REACHY_VEC_TTS_BACKEND=qwen-tts`) and the key in `.env`, run the
  script. Record the numbers in `docs/project-notes.md` under "Local LLM
  benchmark (2026-10-20)". If it fails, apply the spec §4 fallbacks in
  order and re-run.
- [ ] **Step 6:** Commit: `feat: bench_llm speed/tool-accuracy check for the local model`.

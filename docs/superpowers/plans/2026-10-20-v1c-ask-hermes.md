# V1c — `ask_hermes` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reachy can hand a multi-step task to the Hermes agent (profile
`reachy`, read-only plus scheduling tools). It waits up to 20 s, then hands
off, delivering the result later as a message to a recognised requester.

**Architecture:**
- `brain/hermes.py` holds an `Agent` protocol, `HermesCli` (list-form
  `subprocess.run` of `hermes -p reachy -z … -t …`), and `HermesJobs`, a
  single-worker executor that turns late results into `messages` rows.
- `ChatBrain` registers `ask_hermes` only when given a `HermesJobs`. It
  speaks a holding line through the turn's `on_sentence`, then waits.
- `cli/run.py` wires it when `hermes_enabled` is on.

**Tech Stack:** Python 3.12, `subprocess`, `concurrent.futures`, pytest
with the fakes.

**Spec:** `docs/superpowers/specs/2026-10-19-hermes-local-llm-voice-design.md`
(§2, Security, Failure behaviour)

**Depends on:** V1b Task 2, so late results are delivered mid-conversation.

## Global Constraints

Settings (in `config.py`, `REACHY_VEC_` prefix):

| Setting | Default |
|---|---|
| `hermes_enabled: bool` | `False` |
| `hermes_bin: str` | `"hermes"` |
| `hermes_profile: str` | `"reachy"` |
| `hermes_toolsets: str` | `"web,browser,memory,skills,session_search,todo,cronjob,kanban"` |
| `hermes_wait_s: float` | `20.0` |
| `hermes_task_timeout_s: float` | `600.0` |

Fixed behaviour and copy:

- Argument list, in order:
  `[hermes_bin, "-p", profile, "-z", prompt, "-t", toolsets]`.
  Never `shell=True`.
- Prompt sent to Hermes, exactly: `"Task from {requester or 'an unidentified
  teammate'}, asked by voice via Reachy: {task}. Reply in at most 3 short
  sentences suitable for speaking aloud."`
- Holding line spoken before waiting: `"Let me look into that."`
- Tool results returned to the model, exactly:
  - Done in time: Hermes's stdout, stripped.
  - Handed off (recognised requester): `"still working; tell the user
    you'll let them know when it's ready"`
  - Too slow (anonymous requester): `"this is taking too long, and I can't
    get back to someone I don't recognise"`
  - Busy: `"I'm still working on your last request"`
  - Error or timeout: `"Hermes couldn't finish that"`
- A late result becomes
  `MessageRow(from_person="hermes", from_name="Hermes", to_person=<requester id>, …)`.
- `ruff check src tests` and `pytest -q` pass before each commit.

## Review Focus

- Transcript text containing shell metacharacters (`; rm -rf ~`, quotes,
  `$(…)`) reaches Hermes verbatim as one argument and is never interpreted.
  Test in Task 1.
- Hermes is not installed (`FileNotFoundError`): a spoken failure, not a
  crash. Test in Task 1.
- The user barges in on the holding line: the wait still happens and the
  tool still returns normally. Test in Task 3.
- A job finishes *after* the conversation ended: the message is stored and
  delivered on the next "Reachy". Test in Task 2, via the store.
- A second `ask_hermes` while one job runs is refused, with no second
  process started. Test in Task 2.

---

### Task 1: `Agent` protocol + `HermesCli`

**Files:**
- Create: `src/reachy_vec/brain/hermes.py`
- Modify: `src/reachy_vec/config.py` (the 6 settings, new `# Hermes agent` block)
- Modify: `tests/conftest.py` (add `FakeAgent`)
- Test: `tests/test_hermes.py`

**Interfaces:**
- Produces:
  - `class HermesError(Exception)`
  - `class Agent(Protocol)`:
    `def run(self, task: str, *, requester: str | None, timeout_s: float) -> str`
  - `class HermesCli`:
    `__init__(self, *, bin: str, profile: str, toolsets: str, runner=subprocess.run)`
    and `run(...)` as in `Agent`.
    - `runner` gets `(args, capture_output=True, text=True, timeout=timeout_s)`.
    - A non-zero return code, empty stdout, `TimeoutExpired` or `OSError`
      raises `HermesError`.
  - `build_prompt(task: str, requester: str | None) -> str`, with the exact
    copy above.
  - `FakeAgent(results: list[str | Exception], delay_s: float = 0.0)` in
    conftest. It records `calls: list[tuple[str, str | None]]`.

- [ ] **Step 1: Write failing tests**

```python
def test_args_exact_and_no_shell():
    seen = {}
    def runner(args, **kw): seen.update(args=args, **kw); return CompletedProcess(args, 0, "Done.\n", "")
    out = HermesCli(bin="hermes", profile="reachy", toolsets="web,todo", runner=runner).run(
        "check X; rm -rf ~ $(whoami)", requester="Bob", timeout_s=5)
    assert out == "Done."
    assert seen["args"][:4] == ["hermes", "-p", "reachy", "-z"]
    assert seen["args"][5:] == ["-t", "web,todo"]
    assert "check X; rm -rf ~ $(whoami)" in seen["args"][4] and "shell" not in seen
    assert seen["timeout"] == 5

def test_prompt_copy_anonymous():
    assert build_prompt("x", None).startswith("Task from an unidentified teammate, asked by voice via Reachy: x.")

@pytest.mark.parametrize("runner", [nonzero_exit, empty_stdout, raises_timeout, raises_filenotfound])
def test_failures_raise_hermes_error(runner): ...
```

- [ ] **Step 2:** Run `uv run --no-sync pytest tests/test_hermes.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5 (manual):** Run
  `hermes -p reachy -z "Reply with the word ready." -t web`. It should print
  a reply. If `-t` rejects the comma list, adjust `HermesCli` to repeat `-t`
  for each toolset, and update the test.
- [ ] **Step 6:** Commit: `feat: HermesCli agent adapter (read-only + scheduling toolsets)`.

### Task 2: `HermesJobs` — wait, then hand off

**Files:**
- Modify: `src/reachy_vec/brain/hermes.py`
- Test: `tests/test_hermes.py`

**Interfaces:**
- Consumes: `Agent`, `Store.add_message`, `MessageRow`.
- Produces: `class HermesJobs`:
  - `__init__(self, agent: Agent, store: Store, *, wait_s: float, task_timeout_s: float)`
  - `ask(self, task: str, *, requester_id: str | None, requester_name: str | None) -> str`
    returns one of the exact tool-result strings above. Uses one worker
    thread (`ThreadPoolExecutor(max_workers=1)`) and
    `future.result(timeout=wait_s)`.
  - On a wait timeout:
    - With `requester_id`: add a done-callback that writes the `MessageRow`
      (or, on error, logs and writes nothing).
    - Without: the callback only logs.
  - `busy` property: a job is still running.
  - Every task, requester, duration and result is logged at INFO.

- [ ] **Step 1: Write failing tests** (use `wait_s=0.05` and `FakeAgent(delay_s=…)`)

```python
def test_fast_result_returned(tmp_path):         # FakeAgent(["SDK v2 is out."]) -> "SDK v2 is out."
def test_slow_result_becomes_message_for_requester(tmp_path):
    jobs = HermesJobs(FakeAgent(["SDK v2 is out."], delay_s=0.2), store, wait_s=0.05, task_timeout_s=5)
    assert jobs.ask("t", requester_id="p2", requester_name="Bob").startswith("still working")
    wait_until(lambda: store.pending_messages_for("p2"))
    msg = store.pending_messages_for("p2")[0]
    assert (msg.from_name, msg.text) == ("Hermes", "SDK v2 is out.")
def test_slow_anonymous_result_not_stored(tmp_path):   # returns "this is taking too long..."; no rows
def test_busy_refuses_second_task(tmp_path):           # second ask -> "I'm still working..."; agent.calls == 1 task
def test_error_returns_spoken_failure(tmp_path):       # FakeAgent([HermesError()]) -> "Hermes couldn't finish that"
def test_late_error_writes_no_message(tmp_path):
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass, with no test slower than 1 s.
- [ ] **Step 5:** Commit: `feat: HermesJobs - wait up to hermes_wait_s, then deliver as a message`.

### Task 3: `ask_hermes` tool in `ChatBrain`

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`:
  - New `ASK_HERMES_TOOL` and `ASK_HERMES_HINT` beside `WEB_SEARCH_TOOL`.
  - `__init__(..., hermes: HermesJobs | None = None)`.
  - `_active_tools` and `_system_prompt`.
  - Store the turn's `on_sentence` as `self._on_sentence` in `respond`.
  - New handler `_tool_ask_hermes`.
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Consumes: `HermesJobs.ask`, from Task 2.
- Tool schema: name `ask_hermes`, one required string argument `task`.
  Description, exactly: "Delegate a multi-step task needing research, web
  browsing, scheduling a reminder, or a task-board entry. Not for questions
  you can answer from the provided context."
- `ASK_HERMES_HINT`, appended to the system prompt when enabled:
  " ask_hermes hands a task to your agent colleague Hermes; reminders and
  task-board entries need a recognised speaker - if the speaker is unknown,
  say you can only schedule things for people you know."
- Handler:
  1. Speak `"Let me look into that."` through `self._on_sentence` when set,
     swallowing `SpeechInterrupted`.
  2. Return
     `hermes.ask(task, requester_id=self._turn.person_id, requester_name=self._turn.name)`.
  3. An empty `task` returns `"can't ask: empty task"`.

- [ ] **Step 1: Write failing tests**

```python
def test_ask_hermes_absent_by_default(tmp_path):     # no "ask_hermes" in last_kwargs["tools"] names
def test_ask_hermes_passes_identity_and_speaks_holding_line(tmp_path):
    # scripted tool call ask_hermes{"task":"research X"} then final "Done: X shipped."
    sentences = []
    brain.respond("research X", identity=TurnIdentity("p1", "Yang"), on_sentence=sentences.append)
    assert sentences[0] == "Let me look into that."
    assert agent.calls == [("research X", "Yang")]
def test_ask_hermes_anonymous_passes_none(tmp_path):   # agent.calls[0][1] is None
def test_barge_in_on_holding_line_still_waits(tmp_path):
    # on_sentence raises SpeechInterrupted for the holding line -> tool result still returned
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: ask_hermes tool - delegate multi-step tasks to Hermes`.

### Task 4: Wire into `run` + docs

**Files:**
- Modify: `src/reachy_vec/cli/run.py`:
  - When `settings.hermes_enabled`, build
    `HermesJobs(HermesCli(bin=…, profile=…, toolsets=…), store, wait_s=…, task_timeout_s=…)`
    and pass `hermes=` to `ChatBrain`.
  - If `shutil.which(settings.hermes_bin)` is None, `typer.echo` a warning
    and leave Hermes off.
- Modify: `.env.example` (commented `REACHY_VEC_HERMES_ENABLED=true` block)
- Modify: `docs/testing.md` (add the spec's 8 smoke-test steps)
- Modify: `CLAUDE.md`:
  - Control-flow paragraph: mention `ask_hermes` and the local LLM.
  - Replace the "Answers require the OpenAI API" gap with "Answers need the
    local oMLX server (or OpenAI if `LLM_BASE_URL` is unset)".
- Modify: `docs/architecture.md`, `docs/pipelines.md`,
  `docs/configuration.md` (remove face references, add the LLM and Hermes
  settings)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `make_hermes(settings=settings, which=shutil.which) -> HermesJobs | None`
  in `cli/run.py`. It returns None when disabled or the binary is missing.

- [ ] **Step 1: Write failing tests**

```python
def test_make_hermes_none_when_disabled(monkeypatch):
def test_make_hermes_none_when_binary_missing(monkeypatch):    # which=lambda b: None
def test_make_hermes_builds_jobs_with_settings(monkeypatch):   # which=lambda b: "/x/hermes"; isinstance HermesJobs
```

- [ ] **Step 2:** Run `uv run --no-sync pytest tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement and update the docs.
- [ ] **Step 4:** Run `uv run --no-sync ruff check src tests && uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5 (manual, real robot):** run the `docs/testing.md` smoke
  checklist with `REACHY_VEC_HERMES_ENABLED=true`. Steps 6 and 7 (a quick
  and a slow Hermes task) must pass before this is called done.
- [ ] **Step 6:** Commit: `feat: wire Hermes into run; docs for local LLM + Hermes`.

---

## Version 2 (future plans, not part of V1)

- **V2a memory handoff** (spec §5): `ChatBrain._summarize_and_store` also
  queues `HermesJobs` "Remember about {name}: {notes}" for each recognised
  participant. Fire-and-forget; anonymous participants are never sent.
- **V2b cron collector** (spec §6): an idle-loop scan of
  `~/.hermes/profiles/reachy/cron/output` for `reachy:{person_id}:*` jobs,
  with a seen-set in `data/hermes_cron_seen.json`, writing `messages` rows.
  A fixture test pins the output format.

Each gets its own plan once V1 has run on the robot.

# Local brain + Hermes agent, voice-only identity

**Date:** 2026-10-20 (supersedes the 2026-10-19 draft)
**Status:** Design agreed in brainstorming — pending written-spec review

## Goal

Run Reachy fully on the Mac, with no OpenAI dependency, and let it hand
multi-step tasks to a Hermes agent. Both run on one local model.

Agreed decisions:

| # | Decision | Choice |
|---|---|---|
| 1 | LLM | Local oMLX server, model `Qwen3.8-27B-oQ4e-mtp`, shared by Reachy and Hermes |
| 2 | Memory | LanceDB stays Reachy's memory; option 2: Reachy in front, Hermes builds its own picture of people from what Reachy sends it (version 2) |
| 3 | Identity | Voice only, no face recognition; Reachy uses people's names naturally |
| 4 | Wake word | "Reachy" |
| 5 | Hermes's role | A: Reachy hands it agent tasks (plus a taste of B: Hermes remembers people) |
| 6 | Hermes's permissions | Read-only plus scheduling |
| 7 | Waiting | Wait about 20 s, then hand off as a message |
| 8 | Scheduled results | Reachy collects them from Hermes's cron output (version 2) |
| 9 | Transport | One-shot `hermes` command, no extra server |

**Guiding constraint:** keep the first version minimal. Anything not needed
for the demo to work goes to version 2 or the non-goals.

Non-goals: face recognition and camera tools (disabled, code kept); a cloud
fallback when oMLX is down; the Hermes gateway API or MCP; letting Hermes
run shell commands, write files or control the Mac; replacing LanceDB.

## Current state (2026-10-20)

- **Built, not yet committed: wake word and voice-only identity.**
  - `brain/oracle.py` listens all the time. `split_wake()` matches
    `WAKE_WORDS` (reachy, reachie, reechy, reachi, richie, ritchie), and any
    words after the wake word become the first question.
  - The voice (`EcapaSpeakerIdentifier` → `perception/fusion.py:from_voice`)
    decides who is talking on each turn.
  - Strangers are offered voice enrollment: name, confirmation, then 2
    spoken sentences.
  - Known voices are greeted by name, outside the 2 h cooldown, and hear
    their pending messages.
  - Face code, the `enroll` command, `--preview` and the `people` table are
    gone.
- **Built, not yet committed: layered config.** `config.py` reads
  `~/.hermes/profiles/reachy/.env`, and the repo `.env` overrides it.
- **LLM calls still use the OpenAI cloud.** `OpenAI()` is built in
  `cli/run.py` and `cli/chat.py`. `ChatBrain` streams chat with tool calls,
  and `end_conversation` distills up to 3 notes per person through the same
  client.
- **Hermes:**
  - Installed at `~/.local/bin/hermes`, with a dedicated profile `reachy`
    (`hermes -p reachy …`).
  - That profile uses `Qwen3.8-27B-oQ4e-mtp` on oMLX at
    `http://127.0.0.1:8001/v1`. oMLX requires an API key.
  - Cron output goes to `~/.hermes/profiles/reachy/cron/output`.
- **Speech-to-text (faster-whisper) and text-to-speech (`say` / Qwen3-TTS)
  are already local.** Unified memory is shared by the LLM, Qwen3-TTS,
  Whisper, ECAPA (voice ID) and BGE (embeddings).

## Architecture

```
 mic ─▶ wake word ─▶ STT ─▶ voice ID ─▶ ChatBrain ──stream──▶ TTS ─▶ speaker
                                          │   ▲
                       every turn         │   └── LanceDB (docs, memories,
                                          │        voices, messages)
                                          ▼
                     oMLX :: Qwen3.8-27B-oQ4e-mtp  (127.0.0.1:8001/v1)
                                          ▲
                       ask_hermes         │
                       (background) ──▶ `hermes -p reachy -z …`
```

- **Reachy owns the conversation.** Every spoken turn goes straight from
  `ChatBrain` to oMLX. No agent framework sits on that path.
- **Hermes is a tool, not the brain.** The model calls `ask_hermes` when a
  task needs more than one turn. Hermes never speaks to the user; Reachy
  speaks what it returns.
- **One model in memory,** shared by both. Reachy's spoken turns and Hermes
  tasks queue on the same server.
- **Identity and the "never guess" rule stay in Reachy.** Hermes gets a name
  only when voice ID has made a confident match.

---

## Version 1 (build now)

### 1. Local LLM client — `brain/llm.py`, `config.py`

New settings, all with the `REACHY_VEC_` prefix:

| Setting | Default | Purpose |
|---|---|---|
| `llm_base_url` | `None` | e.g. `http://127.0.0.1:8001/v1`. `None` = OpenAI cloud (unchanged behaviour). |
| `llm_api_key` | `None` | oMLX key. Lives in the profile `.env` or the repo `.env`, never in code. |
| `llm_model` | existing | Set to `Qwen3.8-27B-oQ4e-mtp`. |
| `llm_disable_thinking` | `True` | Turn off Qwen's thinking mode on every request. |
| `llm_timeout_s` | `30.0` | A hung server fails one turn, not the loop. |

- **`make_llm_client()`** returns
  `OpenAI(base_url=…, api_key=…, timeout=…)` when `llm_base_url` is set,
  and `OpenAI()` otherwise. `cli/run.py` and `cli/chat.py` use it, and no
  other code builds a client.
- **Thinking off:** `ChatBrain` adds
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to
  `_llm_kwargs` when the flag is on. That covers streaming replies, tool
  follow-ups and the memory summary.
- **Never speak reasoning:** `<think>…</think>` text in streamed content is
  removed before it reaches the sentence splitter, including when the tags
  are split across chunks. If the server ignores the flag, the reasoning is
  still never spoken.
- **Warm-up:** the existing one-token ping, then one request with the real
  system prompt and tool list, so oMLX's prompt cache is warm before the
  first real turn.
- **Embeddings stay on BGE** (local, 384-dim). LanceDB schemas don't change.
- `CLAUDE.md`'s "Answers require the OpenAI API" gap is updated once this
  ships.

### 2. Hermes tool — `brain/hermes.py`, `brain/chat.py`

**Protocol and fake,** following the repo pattern:

```python
class Agent(Protocol):
    def run(self, task: str, *, requester: str | None, timeout_s: float) -> str: ...
```

- **`HermesCli`** runs
  `hermes -p {hermes_profile} -z "<prompt>" -t {hermes_toolsets}` with
  `subprocess.run(..., timeout=hermes_task_timeout_s)` and returns stdout.
  A non-zero exit or empty output raises `HermesError`.
- **`FakeAgent`** goes in `tests/conftest.py`. It returns scripted text,
  sleeps, or raises.
- **Prompt sent to Hermes:** "Task from {requester or 'an unidentified
  teammate'}, asked by voice via Reachy: {task}. Reply in at most 3 short
  sentences suitable for speaking aloud." Hermes's own memory of people
  (version 2) personalises the work.

**Settings:**

| Setting | Default | Purpose |
|---|---|---|
| `hermes_enabled` | `False` | Registers the `ask_hermes` tool (opt-in, like `web_search`). |
| `hermes_bin` | `"hermes"` | Executable to run. |
| `hermes_profile` | `"reachy"` | Hermes profile: its own memory, cron and model. |
| `hermes_toolsets` | `"web,browser,memory,skills,session_search,todo,cronjob,kanban"` | Read-only plus scheduling. No terminal, file, code execution or computer use. |
| `hermes_wait_s` | `20.0` | How long Reachy waits before handing off. |
| `hermes_task_timeout_s` | `600.0` | Hard limit for a handed-off task. |

**Tool `ask_hermes(task: string)`:**
- Description: "Delegate a multi-step task needing research, web browsing,
  scheduling a reminder, or a task-board entry. Not for questions you can
  answer from the provided context."
- **Flow:**
  1. Reachy speaks a short holding line through `on_sentence` ("Let me look
     into that."), so barge-in still works.
  2. The agent runs on a background thread (`HermesJobs`, a single worker).
  3. Reachy waits up to `hermes_wait_s` for it.
  4. **Done in time:** the tool returns Hermes's text, and the model speaks a
     1–2 sentence version.
  5. **Still running:**
     - Recognised requester: the tool returns "still working; tell the user
       you'll let them know when it's ready". When the job finishes, the
       result becomes a `messages` row from "Hermes" to the requester. It is
       read out the next time they say "Reachy", or at the next turn
       boundary if they're still in the conversation.
     - Anonymous requester: there is nobody to deliver to. The tool returns
       "this is taking too long, and I can't get back to someone I don't
       recognise", and the job's result is only logged.
  6. **Error or timeout:** the tool returns "Hermes couldn't finish that",
     and Reachy says so plainly, without making up a result.
- **Scheduling needs a recognised requester.** Anonymous requests that ask
  for reminders or kanban entries are refused at the prompt level, the same
  way `send_message` refuses anonymous senders.
- **One job at a time.** If one is already running, the tool returns "I'm
  still working on your last request", so the model can't stack slow jobs on
  the shared model.
- **Logging:** every task, its requester, how long it took and the result go
  to `data/reachy.log`.

**Delivering mid-conversation:** `OracleLoop` checks for pending messages
for the current owner at each turn boundary, before listening again, not
only at the greeting. This reuses `_deliver_messages`.

### 3. Using names — `brain/chat.py`

- **Prompt rule** (added to `PERSONALITY`): "When you know who is speaking,
  use their first name occasionally: in your first reply, when the speaker
  changes, and when it helps. Never in every sentence. If the speaker is
  unknown, use no name."
- **Speaker-change note:** when this turn's recognised person differs from
  the previous turn's, the turn context begins with
  `(Speaker changed: now {name}.)`. Turns where Reachy can't tell who is
  speaking don't count as a change, so the previous name is never carried
  over to them.
- `CONTEXT_TEMPLATE` already includes `speaker`. Nothing else changes.

### 4. Speed check — `scripts/bench_llm.py`

Run against the live oMLX endpoint, with Qwen3-TTS loaded, before
`llm_base_url` becomes the documented default:

| Metric | Target |
|---|---|
| Time to first spoken sentence (median of 10 turns, real system prompt and tools) | **< 1.5 s** |
| Valid tool calls on a fixed set (`send_message`, `save_note`, `get_weather`, `get_time`, `ask_hermes`, and no-tool questions) | **≥ 9/10** |
| Reasoning text in the spoken output | **0** |
| Time to first sentence while a Hermes job is running | Reported, no target (contention baseline) |

**Model choice:** Qwen3.8-27B was picked over Qwen3.6-35B-A3B because it
needs less RAM. Unified memory is shared with Qwen3-TTS, Whisper, ECAPA and
BGE, and memory is the binding constraint on this Mac.

**Risk:** a dense 27B generates fewer tokens per second than a
mixture-of-experts model. Multi-token prediction ("mtp") and the prompt
cache offset part of that. If the 1.5 s target fails, try in this order.
None of these needs more RAM.
1. Shorter system prompt and tool descriptions, which means less prompt to
   process when the cache misses.
2. Make sure the prompt cache hits: keep the system prompt and tools
   byte-identical between turns, and put per-turn context only in the user
   message.
3. Speak a short acknowledgement while the first sentence is being
   generated.
4. Only then, a smaller quantisation, accepting some loss in quality.

The spec doesn't change; only config does.

## Version 2 (decided, built after version 1 is stable)

### 5. Memory handoff to Hermes (memory option 2)
- At `end_conversation`, after Reachy writes up to 3 notes to LanceDB for
  each recognised participant, it also queues one `HermesJobs` task:
  "Remember about {name}: {notes}".
- It's fire-and-forget: a failure is logged and skipped, and LanceDB is
  unaffected.
- Hermes never writes into LanceDB. Anonymous participants are never sent.

### 6. Cron collector
- Reachy asks Hermes to name scheduled jobs `reachy:{person_id}:{slug}`.
  The person id comes from the `ask_hermes` context, never from the model's
  guess.
- When the idle loop is between listens, Reachy scans
  `~/.hermes/profiles/reachy/cron/output` for new runs of `reachy:*` jobs.
  It keeps a seen-set in a small state file under `data/`.
- Each new run becomes a `messages` row from "Hermes" to that person.
- Format risk: a fixture test pins the output format, so a Hermes upgrade
  that changes it fails loudly instead of silently.

---

## Failure behaviour

| Failure | Behaviour |
|---|---|
| oMLX down or timing out | That turn gets the existing apology. The loop keeps listening. A failed warm-up logs a clear warning at startup. No cloud fallback. |
| Hermes missing, crashing or timing out | `ask_hermes` returns an error and Reachy says it couldn't do it. Normal conversation is unaffected. |
| Hermes still running at hand-off | Result is delivered later as a message (recognised requester), or only logged (anonymous requester). |
| Voice ID broken | Everyone is anonymous. Reachy answers, stores nothing, and can't hand off. |
| oMLX busy with a Hermes job | Reachy's turn may be slower. Measured in §4, and only one Hermes job runs at a time. |

## Security

- Hermes runs with `-t` set to read-only plus scheduling toolsets. Its
  profile-wide toolsets are irrelevant because `-t` restricts each call.
- The task text is passed as a single argument (no shell, list-form
  `subprocess`), so transcripts can't inject shell syntax.
- Only the task text and the requester's first name go to Hermes. Voice
  embeddings, transcripts and LanceDB memories don't (memories do in version
  2, by design).
- The oMLX key lives in an env file, never in code or logs.

## Testing

All with fakes, no devices or network:

- `make_llm_client` uses base URL, key and timeout when set, and plain
  `OpenAI()` when not.
- The thinking flag reaches every `create()` call.
- `<think>` stripping works when the tags are split across stream chunks.
- `ask_hermes` registration:
  - Present only when `hermes_enabled` is on.
  - The requester's name is passed, and `None` when anonymous.
- `ask_hermes` outcomes:
  - Fast result is returned.
  - Slow result with a recognised requester becomes a message.
  - Slow result with an anonymous requester is not stored.
  - Error and timeout give the spoken failure.
  - A second request while busy is refused.
- `HermesCli` builds the expected argument list (profile, `-z`, `-t`) and
  maps non-zero exits to `HermesError`. Tested with a fake `subprocess`.
- Pending messages are delivered at turn boundaries mid-conversation.
- Speaker-change note:
  - Added when the recognised speaker changes.
  - Not added for turns where Reachy can't tell.
  - No name on anonymous turns.

Smoke test on the robot (added to `docs/testing.md`):

1. With Wi-Fi off and only oMLX running, Reachy starts and answers.
2. Wake from across the room.
3. A known voice is greeted by name.
4. Two people take turns and Reachy names each correctly.
5. A stranger enrolls.
6. A quick Hermes task is answered within 20 s.
7. A slow Hermes task is handed off and read out on the next "Reachy".
8. Stopping oMLX gives the apology, not a crash.

## Rollout

1. Commit the already-built wake-word, voice-only and layered-config work.
2. §1 local LLM client, plus §4 speed check. Switch `.env.example` to oMLX
   once the targets pass.
3. §3 using names.
4. §2 `ask_hermes`, including mid-conversation delivery.
5. Update docs (`architecture.md`, `pipelines.md`, `configuration.md`,
   `testing.md`, `CLAUDE.md`). They still describe faces and OpenAI.
6. Version 2: §5 memory handoff, then §6 cron collector.

## Open items (not blocking the plan)

- The oMLX key needs to be in `~/.hermes/profiles/reachy/.env` or the repo
  `.env` as `REACHY_VEC_LLM_API_KEY`. The user sets this; Claude never reads
  the key.
- Confirm oMLX honours `chat_template_kwargs.enable_thinking` and streams
  tool-call deltas. The §4 speed check verifies both.
- Demo source of truth (SQLite or Mongo) is out of scope here. `sync-mongo`
  is unchanged.

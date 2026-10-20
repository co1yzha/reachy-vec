# V1b — Using names + mid-conversation delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reachy addresses recognised speakers by first name naturally,
notices when the speaker changes, never names an unidentified speaker, and
delivers queued messages at turn boundaries (needed by V1c's Hermes
hand-offs).

**Architecture:**
- A prompt rule goes in `PERSONALITY`.
- `ChatBrain` tracks the last recognised speaker and adds a one-line note
  to the **user message** (never the system prompt, to keep the prompt
  cache hitting).
- `OracleLoop._converse` calls the existing `_deliver_messages` for each
  turn's recognised speaker.

**Tech Stack:** Python 3.12, pytest with the fakes.

**Spec:** `docs/superpowers/specs/2026-10-19-hermes-local-llm-voice-design.md`
(§3, and "Delivering mid-conversation" in §2)

## Global Constraints

- The speaker-change note is exactly `(Speaker changed: now {name}.)` on its
  own line at the start of the user message content.
- `PERSONALITY` gains exactly: "When you know who is speaking, use their
  first name occasionally: in your first reply, when the speaker changes,
  and when it helps. Never in every sentence. If the speaker is unknown, use
  no name."
- Anonymous or can't-tell turns never produce a name and never count as a
  speaker change.
- `ruff check src tests` and `pytest -q` pass before each commit.

## Review Focus

- A → can't tell → A: no change note on the return to A, since the
  anonymous turn doesn't reset the last speaker. Test in Task 1.
- A → can't tell → B: one note naming B. Test in Task 1.
- The first identified turn of a conversation gets no "changed" note. The
  greeting already names them. Test in Task 1.
- A message queued *during* the conversation (as a Hermes hand-off will
  do) is spoken at the next turn boundary, and not repeated afterwards.
  Test in Task 2.
- After barge-in, nothing is delivered on that boundary: the user is
  already talking. Test in Task 2.

---

### Task 1: Name rule + speaker-change note

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`:
  - `PERSONALITY`
  - `ChatBrain.__init__` and `reset()` (new `_last_speaker_id: str | None`)
  - `respond()` (prefix the user content)
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- No public signature changes. The behaviour is visible in the user
  message sent to `create()`.

- [ ] **Step 1: Write failing tests** (use `FakeLLMClient`, then read
  `client.chat.completions.last_kwargs["messages"][-1]["content"]`)

```python
ALICE, BOB = TurnIdentity("p1", "Alice"), TurnIdentity("p2", "Bob")
def last_user(client): return client.chat.completions.last_kwargs["messages"][-1]["content"]

def test_no_change_note_on_first_identified_turn(tmp_path):   # -> no "Speaker changed"
def test_change_note_when_recognised_speaker_changes(tmp_path):
    brain.respond("q1", identity=ALICE); brain.respond("q2", identity=BOB)
    assert last_user(client).startswith("(Speaker changed: now Bob.)\n")
def test_anonymous_turn_is_not_a_change_and_has_no_name(tmp_path):
    brain.respond("q1", identity=ALICE); brain.respond("q2", identity=None)
    assert "Speaker changed" not in last_user(client) and "Alice" not in last_user(client)
    brain.respond("q3", identity=ALICE)
    assert "Speaker changed" not in last_user(client)
def test_change_across_anonymous_turn(tmp_path):    # ALICE, None, BOB -> note for Bob
def test_reset_forgets_last_speaker(tmp_path):      # ALICE, reset(), BOB -> no note
def test_personality_has_name_rule():
    assert "use their first name occasionally" in PERSONALITY
```

- [ ] **Step 2:** Run `uv run --no-sync pytest tests/test_chat_brain.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement:
  - Prepend the note when `self._turn.person_id` is set and differs from
    `_last_speaker_id`, and `_last_speaker_id` is not None.
  - Then set `_last_speaker_id` only when `self._turn.person_id` is set.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: use speaker names naturally + speaker-change note`.

### Task 2: Deliver messages at turn boundaries

**Files:**
- Modify: `src/reachy_vec/brain/oracle.py` (`_converse`: after the
  `nod`/bank step of a non-interrupted turn)
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: the existing `OracleLoop._deliver_messages(person_id)` and
  `Store.add_message` / `pending_messages_for` / `mark_delivered`.
- Behaviour: after a non-interrupted turn whose `turn_voice.person_id` is
  set, call `_deliver_messages(turn_voice.person_id)`. Deliver to the
  recognised speaker of that turn, not only the conversation owner, so a
  hand-off for Bob reaches Bob when he chimes in.

- [ ] **Step 1: Write failing tests**

```python
def test_message_queued_mid_conversation_is_delivered_next_boundary(tmp_path):
    # brain whose respond() adds a MessageRow(to_person="p1", from_name="Hermes",
    # text="the SDK shipped v2") to the store on its first call
    loop, speaker, *_ = make_loop(tmp_path, utterances=[said("Reachy"), said("q1"), said("q2")],
                                  voices=[ALICE_VOICE, ALICE_VOICE, ALICE_VOICE], brain=QueueingBrain(store))
    loop.run_once()
    assert sum("Hermes left you a message: the SDK shipped v2" in s for s in speaker.spoken) == 1

def test_no_delivery_after_barge_in(tmp_path):
    # InterruptingBrain that also queues a message -> not spoken on that boundary
def test_no_delivery_for_anonymous_turn(tmp_path):
    # message for p1 pending; turn voice None -> not delivered
```

- [ ] **Step 2:** Run `uv run --no-sync pytest tests/test_oracle.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the one call in `_converse`.
- [ ] **Step 4:** Run `uv run --no-sync pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: deliver queued messages at conversation turn boundaries`.

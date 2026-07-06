# Phase 1d: Oracle Loop — State Machine + `reachy-vec run`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Phase 1 milestone: `reachy-vec run` — recognized face → cooldown-aware greeting → voice Q&A via Phase 0 RAG → robot-led enrollment of unknowns.

**Architecture:** `brain/oracle.py` holds `OracleLoop`, a synchronous state machine with every dependency injected (`sight`, `transcriber`, `speaker`, `body`, `answer_fn`, `enroll_capture`, `store`, `clock`) so the whole flow runs in pytest with fakes. Note: the parent spec sketched threads+queues; implementation simplifies to polling — `listen_once(timeout)` already blocks as the pacing element, and sight is polled between utterances. Same observable behavior, less machinery; the silence timeout covers "person left mid-listen".

**Tech Stack:** stdlib only (composes 1a/1b/1c + Phase 0), pytest.

**Depends on:** 1a (`Body`, `make_body`), 1b (`Transcriber`, `Speaker`, fakes), 1c (`Observation`, `FaceMatcher`, `Camera`, `enroll_person`, people/greetings store). **Delivers:** the Phase 1 milestone.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- New settings: `greet_cooldown_s: float = 7200`, `silence_timeout_s: float = 30` on `Settings`.
- Tests use only fakes from `tests/conftest.py`; no devices, no models, no network.
- Never guess identity; enrollment requires an explicit spoken yes + name confirmation (parent spec).
- Commit after every green test cycle; conventional-commit messages.

---

### Task 1: OracleLoop state machine

**Files:**
- Create: `src/reachy_vec/brain/oracle.py`
- Modify: `src/reachy_vec/config.py` (add the two settings above)
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `Observation` (1c), `Answer` (Phase 0 rag), fakes (`FakeSpeaker`, `FakeTranscriber`, `FakeFaceMatcher`) from conftest.
- Produces: `OracleLoop(*, sight, transcriber, speaker, body, answer_fn, enroll_capture, store, clock=time.time, greet_cooldown_s=7200.0, silence_timeout_s=30.0, unknown_stable_polls=3)` with `run_once() -> str` (returns a terminal event: `"conversation"`, `"enrolled"`, `"enroll-declined"`, `"no-face"`) and `run_forever()`. Signatures: `sight() -> Observation | None`; `answer_fn(question: str) -> Answer`; `enroll_capture(name: str) -> str | None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/conftest.py`:

```python
class FakeBody:
    def __init__(self):
        self.motions: list[str] = []

    def perform(self, motion: str) -> None:
        self.motions.append(motion)
```

`tests/test_oracle.py`:

```python
from reachy_vec.brain.oracle import OracleLoop
from reachy_vec.brain.rag import Answer
from reachy_vec.perception.face import Observation
from reachy_vec.store.db import Store

from tests.conftest import FakeBody, FakeSpeaker, FakeTranscriber

ALICE = Observation(person_id="p1", name="Alice", score=0.9)
UNKNOWN = Observation(person_id=None, name=None, score=0.1)


def make_loop(tmp_path, *, sights, utterances, enroll_result="p9", clock=None):
    sights_iter = iter(sights)
    speaker, body = FakeSpeaker(), FakeBody()
    store = Store(tmp_path / "db")
    loop = OracleLoop(
        sight=lambda: next(sights_iter, None),
        transcriber=FakeTranscriber(utterances),
        speaker=speaker,
        body=body,
        answer_fn=lambda q: Answer(text=f"answer to {q}", sources=[]),
        enroll_capture=lambda name: enroll_result,
        store=store,
        clock=clock or (lambda: 1000.0),
        unknown_stable_polls=2,
    )
    return loop, speaker, body, store


def test_known_person_greet_question_answer_goodbye(tmp_path):
    loop, speaker, body, store = make_loop(
        tmp_path, sights=[ALICE], utterances=["when is standup?"]
    )
    assert loop.run_once() == "conversation"
    assert any("Alice" in s for s in speaker.spoken)          # spoken greeting
    assert any("answer to when is standup?" in s for s in speaker.spoken)
    assert "greet" in body.motions and "goodbye" in body.motions
    assert store.get_last_greeted("p1") is not None           # cooldown recorded


def test_cooldown_suppresses_spoken_greeting(tmp_path):
    loop, speaker, body, store = make_loop(tmp_path, sights=[ALICE], utterances=[])
    store.set_last_greeted("p1", "el")  # placeholder, overwritten next line
    loop._record_greeting("p1")        # greeted "now" per fake clock
    speaker.spoken.clear(); body.motions.clear()
    loop2, speaker2, body2, _ = make_loop(tmp_path, sights=[ALICE], utterances=[])
    assert loop2.run_once() == "conversation"
    assert not any("Alice" in s for s in speaker2.spoken)     # silent acknowledgment
    assert "acknowledge" in body2.motions


def test_unknown_face_enrolls_on_yes_and_confirm(tmp_path):
    loop, speaker, body, _ = make_loop(
        tmp_path,
        sights=[UNKNOWN, UNKNOWN],                 # stable unknown (2 polls)
        utterances=["yes please", "Bob", "yes"],   # offer-yes, name, confirm
    )
    assert loop.run_once() == "enrolled"
    assert any("Bob" in s for s in speaker.spoken)            # confirmation used name


def test_unknown_face_declines_enrollment(tmp_path):
    loop, speaker, _, _ = make_loop(
        tmp_path, sights=[UNKNOWN, UNKNOWN], utterances=["no thanks"]
    )
    assert loop.run_once() == "enroll-declined"


def test_silence_ends_conversation_with_goodbye(tmp_path):
    loop, _, body, _ = make_loop(tmp_path, sights=[ALICE], utterances=[])
    assert loop.run_once() == "conversation"
    assert body.motions[-1] == "goodbye"


def test_answer_failure_apologizes_and_continues(tmp_path):
    sights_iter = iter([ALICE])
    speaker, body = FakeSpeaker(), FakeBody()

    def broken(q):
        raise RuntimeError("api down")

    loop = OracleLoop(
        sight=lambda: next(sights_iter, None),
        transcriber=FakeTranscriber(["hello?"]),
        speaker=speaker,
        body=body,
        answer_fn=broken,
        enroll_capture=lambda name: None,
        store=Store(tmp_path / "db"),
        clock=lambda: 1000.0,
    )
    assert loop.run_once() == "conversation"
    assert any("sorry" in s.lower() for s in speaker.spoken)


def test_no_face_at_all(tmp_path):
    loop, _, _, _ = make_loop(tmp_path, sights=[None, None, None], utterances=[])
    assert loop.run_once() == "no-face"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reachy_vec.brain.oracle'`.

- [ ] **Step 3: Write the state machine**

`src/reachy_vec/brain/oracle.py`:

```python
"""The Oracle loop: face-triggered greeting, voice Q&A, robot-led enrollment.

Synchronous state machine; all dependencies injected for testability.
sight() is polled; transcriber.listen_once(timeout) blocks and paces the loop.
"""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

APOLOGY = "Sorry, my brain isn't responding right now."
OFFER = "Hi! I don't think we've met. Would you like me to remember you? Say yes or no."


def _is_yes(text: str | None) -> bool:
    return text is not None and "yes" in text.lower()


def _clean_name(text: str) -> str:
    return text.strip().strip(".!?,").title()


class OracleLoop:
    def __init__(
        self,
        *,
        sight,
        transcriber,
        speaker,
        body,
        answer_fn,
        enroll_capture,
        store,
        clock=time.time,
        greet_cooldown_s: float = 7200.0,
        silence_timeout_s: float = 30.0,
        unknown_stable_polls: int = 3,
    ):
        self._sight = sight
        self._transcriber = transcriber
        self._speaker = speaker
        self._body = body
        self._answer_fn = answer_fn
        self._enroll_capture = enroll_capture
        self._store = store
        self._clock = clock
        self._greet_cooldown_s = greet_cooldown_s
        self._silence_timeout_s = silence_timeout_s
        self._unknown_stable_polls = unknown_stable_polls

    # -- public ---------------------------------------------------------

    def run_once(self) -> str:
        """One interaction: wait for a face, converse or enroll, return event."""
        unknown_streak = 0
        while True:
            obs = self._sight()
            if obs is None:
                unknown_streak = 0
                if not self._more_sight_expected():
                    return "no-face"
                continue
            if obs.person_id is not None:
                self._converse(obs.person_id, obs.name)
                return "conversation"
            unknown_streak += 1
            if unknown_streak >= self._unknown_stable_polls:
                return self._offer_enroll()

    def run_forever(self) -> None:
        self._body.perform("idle")
        while True:
            event = self.run_once()
            logger.info("interaction ended: %s", event)
            if event == "no-face":
                time.sleep(0.5)

    # -- states ----------------------------------------------------------

    def _converse(self, person_id: str, name: str) -> None:
        if self._cooldown_expired(person_id):
            self._speaker.speak(f"Hi {name}! What can I help you with?")
            self._body.perform("greet")
            self._record_greeting(person_id)
        else:
            self._body.perform("acknowledge")
        while True:
            self._body.perform("listen")
            question = self._transcriber.listen_once(self._silence_timeout_s)
            if question is None:
                self._body.perform("goodbye")
                return
            try:
                answer = self._answer_fn(question)
                self._speaker.speak(answer.text)
                self._body.perform("nod")
            except Exception:
                logger.exception("answer_fn failed")
                self._speaker.speak(APOLOGY)

    def _offer_enroll(self) -> str:
        self._speaker.speak(OFFER)
        if not _is_yes(self._transcriber.listen_once(10)):
            self._speaker.speak("No problem! I'm around if you need me.")
            return "enroll-declined"
        for _attempt in range(2):
            self._speaker.speak("Great! What's your name?")
            heard = self._transcriber.listen_once(10)
            if heard is None:
                continue
            name = _clean_name(heard)
            self._speaker.speak(f"Nice to meet you, {name} - did I get that right?")
            if _is_yes(self._transcriber.listen_once(10)):
                self._speaker.speak("Hold still while I take a good look at you.")
                person_id = self._enroll_capture(name)
                if person_id is None:
                    self._speaker.speak("I couldn't see you well - let's try another time.")
                    return "enroll-declined"
                self._record_greeting(person_id)
                self._speaker.speak(f"All set, {name}! Ask me anything.")
                self._body.perform("greet")
                return "enrolled"
        self._speaker.speak("Let's try again another time.")
        return "enroll-declined"

    # -- helpers ----------------------------------------------------------

    def _cooldown_expired(self, person_id: str) -> bool:
        last = self._store.get_last_greeted(person_id)
        if last is None:
            return True
        elapsed = self._clock() - datetime.fromisoformat(last).timestamp()
        return elapsed >= self._greet_cooldown_s

    def _record_greeting(self, person_id: str) -> None:
        now_iso = datetime.fromtimestamp(self._clock(), tz=timezone.utc).isoformat()
        self._store.set_last_greeted(person_id, now_iso)

    def _more_sight_expected(self) -> bool:
        """In production sight() never exhausts; fakes return None forever.

        Distinguish 'camera returned no face this poll' (keep polling, real
        run) from 'scripted sights exhausted' (tests). Production wraps
        sight in an endless camera loop, so run_once ending on no-face only
        happens in tests and at shutdown.
        """
        return False
```

Add to `Settings` in `src/reachy_vec/config.py`:

```python
    # Interaction
    greet_cooldown_s: float = 7200.0  # full spoken greeting at most every 2h
    silence_timeout_s: float = 30.0   # end conversation after this much quiet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS. (If `test_cooldown_suppresses_spoken_greeting` needs it, expose `_record_greeting` usage exactly as tested — it is intentionally exercised directly.)

- [ ] **Step 5: Commit**

```bash
git add src/reachy_vec/brain/oracle.py src/reachy_vec/config.py tests/conftest.py tests/test_oracle.py
git commit -m "feat: Oracle state machine - greeting, Q&A, robot-led enrollment"
```

---

### Task 2: Wire `reachy-vec run`

**Files:**
- Modify: `src/reachy_vec/cli/run.py`

**Interfaces:**
- Consumes: everything above plus `make_body` (1a), `MicTranscriber`/`make_speaker` (1b), `WebcamCamera`/`InsightFaceMatcher`/`enroll_person` (1c), Phase 0 `Store`/`BgeEmbedder`/`rag.answer`.
- Produces: working `reachy-vec run` — the Phase 1 milestone.

- [ ] **Step 1: Implement the command**

Replace `src/reachy_vec/cli/run.py` with:

```python
import typer

from reachy_vec.config import settings


def run() -> None:
    """Run the Oracle: face-triggered voice Q&A on webcam + mic (+ sim body)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from reachy_vec.audio.listen import MicTranscriber
    from reachy_vec.audio.speak import make_speaker
    from reachy_vec.body.robot import make_body
    from reachy_vec.brain.oracle import OracleLoop
    from reachy_vec.brain.rag import answer
    from reachy_vec.perception.camera import WebcamCamera
    from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder

    load_dotenv()
    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)

    camera = WebcamCamera(settings.camera_index)
    if camera.read() is None:
        typer.echo("No camera frame - check webcam permission/index.", err=True)
        raise typer.Exit(code=1)

    matcher = InsightFaceMatcher(store)
    speaker = make_speaker()
    embedder = BgeEmbedder(settings.embedding_model)
    client = OpenAI()

    loop = OracleLoop(
        sight=lambda: matcher.observe(camera.read()),
        transcriber=MicTranscriber(),
        speaker=speaker,
        body=make_body(),
        answer_fn=lambda q: answer(
            q, store=store, embedder=embedder, client=client, model=settings.llm_model
        ),
        enroll_capture=lambda name: enroll_person(
            name, camera, matcher, store, speaker.speak
        ),
        store=store,
        greet_cooldown_s=settings.greet_cooldown_s,
        silence_timeout_s=settings.silence_timeout_s,
    )
    typer.echo("Oracle running - walk into frame. Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
```

- [ ] **Step 2: Full suite**

Run: `uv run pytest -q` — expected: all PASS.

- [ ] **Step 3: Manual milestone smoke test**

Prereqs: docs ingested (Phase 0), yourself enrolled (1c smoke), daemon running (`uv run mjpython .venv/bin/reachy-mini-daemon --sim`), `OPENAI_API_KEY` in `.env`, camera + mic permissions granted.

```bash
uv run reachy-vec run
```

Expected sequence: you walk into frame → sim robot greets ("Hi <name>!" + greet motion in the viewer) → you ask a question about your ingested docs → spoken answer → stay silent 30s → goodbye nod → back to idle. Then cover the camera, have a colleague (or your phone photo won't work — insightface anti-trivial) appear → enrollment offer → complete the yes/name/confirm flow.

- [ ] **Step 4: Commit**

```bash
git add src/reachy_vec/cli/run.py
git commit -m "feat: wire reachy-vec run - Phase 1 Oracle milestone"
```

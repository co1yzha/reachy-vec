# Phase 2b — Speaker ID + Identity Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute every utterance to the right person by voice (ECAPA embeddings) fused with face ID, so notes/memories/messages follow whoever actually spoke; unknown voices are answered but never written to the store.

**Architecture:** The transcriber returns the captured audio alongside the text (`Utterance`); a `SpeakerIdentifier` (ECAPA behind a protocol) matches it against a new `voices` LanceDB table; a pure `fuse()` function combines the face observation with the voice observation into a per-turn `TurnIdentity` that `ChatBrain.respond` uses for retrieval and tool attribution. Voice profiles come from one spoken phrase at enrollment plus passive backfill from confident solo face matches.

**Tech Stack:** speechbrain (ECAPA-TDNN, `speechbrain/spkrec-ecapa-voxceleb`, 192-dim), LanceDB, existing protocol-with-fake test pattern.

**Spec:** `docs/superpowers/specs/2026-07-07-phase2b-speaker-id-fusion-design.md`

## Global Constraints

- Never guess: below `voice_threshold` = anonymous; within 0.05 under = "can't tell" (fall back to face).
- New settings (in `config.py`, `REACHY_VEC_` prefix): `voice_threshold=0.30`, `voice_min_utterance_s=1.0`, `voice_passive_cap=10`.
- ECAPA dim = 192. Raw audio is never persisted — embeddings only.
- speechbrain moves from the `perception` optional extra into main dependencies.
- Failures in speaker ID / passive backfill are logged and swallowed — never block a reply.
- Every task: `uv run pytest -q` green and `uv run ruff check src tests` clean before commit.

---

### Task 1: `Utterance` — transcriber returns text + audio

**Files:**
- Modify: `src/reachy_vec/audio/listen.py`
- Modify: `src/reachy_vec/brain/oracle.py` (call sites only — use `.text`)
- Modify: `tests/conftest.py` (`FakeTranscriber`)
- Test: `tests/test_listen.py`

**Interfaces:**
- Produces: `Utterance` frozen dataclass (`text: str`, `audio: np.ndarray | None = None`) in `audio/listen.py`; `Transcriber.listen_once(timeout_s) -> Utterance | None`. Later tasks read `utterance.text` and `utterance.audio`.

- [ ] **Step 1: Write the failing tests** — in `tests/test_listen.py` add:

```python
from reachy_vec.audio.listen import Utterance


def test_utterance_carries_text_and_audio():
    audio = np.zeros(16000, dtype=np.float32)
    utt = Utterance(text="hello", audio=audio)
    assert utt.text == "hello"
    assert utt.audio is audio


def test_mic_transcriber_returns_utterance(monkeypatch):
    t = MicTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(t, "_capture", lambda timeout_s: audio)

    class FakeSeg:
        text = " hi there "

    t._whisper = type("W", (), {"transcribe": lambda self, a, **kw: ([FakeSeg()], None)})()
    monkeypatch.setattr(t, "_load", lambda: None)
    utt = t.listen_once(5)
    assert utt.text == "hi there"
    assert utt.audio is audio
```

(Adapt to the file's existing test style/imports — it already tests `collect_utterance`; keep those tests untouched.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_listen.py -q` — expected: FAIL (`ImportError: cannot import name 'Utterance'`).

- [ ] **Step 3: Implement** in `src/reachy_vec/audio/listen.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Utterance:
    """One transcribed utterance plus the raw mono 16 kHz float32 audio."""

    text: str
    audio: np.ndarray | None = field(default=None, repr=False)


class Transcriber(Protocol):
    def listen_once(self, timeout_s: float) -> Utterance | None: ...
```

In `MicTranscriber.listen_once`, replace the tail:

```python
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("heard: %r", text)
        return Utterance(text=text, audio=audio) if text else None
```

Same change in `OpenAITranscriber.listen_once` (`return Utterance(text=text, audio=audio) if text else None`). `warm_up` unchanged.

- [ ] **Step 4: Update call sites.** In `src/reachy_vec/brain/oracle.py`:

```python
def _is_yes(utterance) -> bool:
    return utterance is not None and "yes" in utterance.text.lower()
```

In `_converse`: `question = self._transcriber.listen_once(...)` stays; the respond call becomes `self._brain.respond(question.text, speaker_name=name, on_sentence=...)` (identity comes in Task 6). In `_offer_enroll`: `heard = self._transcriber.listen_once(10)`; `name = _clean_name(heard.text)`.

In `tests/conftest.py`, `FakeTranscriber` wraps scripted strings:

```python
from reachy_vec.audio.listen import Utterance


class FakeTranscriber:
    """Returns scripted utterances, then None (silence). Accepts plain strings
    or Utterance objects (for tests that script audio)."""

    def __init__(self, utterances: list):
        self._it = iter(utterances)

    def listen_once(self, timeout_s: float) -> Utterance | None:
        nxt = next(self._it, None)
        if nxt is None or isinstance(nxt, Utterance):
            return nxt
        return Utterance(text=nxt)
```

- [ ] **Step 5: Full suite + lint pass**

Run: `uv run pytest -q && uv run ruff check src tests` — expected: all pass (oracle/listen/cli tests still green).

- [ ] **Step 6: Commit** — `git commit -m "refactor: transcriber returns Utterance(text, audio) for speaker ID"`

---

### Task 2: `voices` table — schema, store methods, config knobs

**Files:**
- Modify: `src/reachy_vec/store/schemas.py`, `src/reachy_vec/store/db.py`, `src/reachy_vec/config.py`
- Test: `tests/test_voices_store.py` (create)

**Interfaces:**
- Produces: `VoiceRow` (`voice_id, person_id, name, vector[192], created_at, source`), `VOICE_EMBEDDING_DIM = 192`; `Store.add_voice_rows(rows)`, `Store.match_voice(vector, k=5) -> (person_id, name, score) | None`, `Store.passive_voice_count(person_id) -> int`, `Store.prune_passive_voices(person_id, keep)`; settings `voice_threshold=0.30`, `voice_min_utterance_s=1.0`, `voice_passive_cap=10`.

- [ ] **Step 1: Write failing tests** — `tests/test_voices_store.py`:

```python
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow


def vec(seed: float) -> list[float]:
    return [seed] * VOICE_EMBEDDING_DIM


def row(i: int, person: str, name: str, seed: float, source: str = "enrolled") -> VoiceRow:
    return VoiceRow(
        voice_id=f"{person}:{i}", person_id=person, name=name,
        vector=vec(seed), created_at=f"2026-07-08T00:00:{i:02d}+00:00", source=source,
    )


def test_match_voice_empty_store_returns_none(tmp_path):
    assert Store(tmp_path / "db").match_voice(vec(0.5)) is None


def test_match_voice_majority_vote(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows([row(0, "p1", "Alice", 0.9), row(1, "p1", "Alice", 0.9),
                          row(2, "p2", "Bob", 0.1)])
    person_id, name, score = store.match_voice(vec(0.9))
    assert (person_id, name) == ("p1", "Alice")
    assert score > 0.99


def test_passive_prune_keeps_newest(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows([row(i, "p1", "Alice", 0.5, source="passive") for i in range(4)])
    store.add_voice_rows([row(9, "p1", "Alice", 0.5)])  # enrolled row never pruned
    store.prune_passive_voices("p1", keep=2)
    assert store.passive_voice_count("p1") == 2
    # enrolled row survives; the two newest passive rows survive
    remaining = {r["voice_id"] for r in
                 store._table("voices", VoiceRow).to_arrow().to_pylist()}
    assert remaining == {"p1:9", "p1:2", "p1:3"}
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_voices_store.py -q` → ImportError on `VoiceRow`.

- [ ] **Step 3: Implement.** `schemas.py`:

```python
VOICE_EMBEDDING_DIM = 192


class VoiceRow(LanceModel):
    voice_id: str
    person_id: str
    name: str
    vector: Vector(VOICE_EMBEDDING_DIM)
    created_at: str  # ISO-8601 UTC
    source: str      # "enrolled" | "passive"
```

`db.py` (add `VOICES_TABLE = "voices"`, import `VoiceRow`; mirror the faces section):

```python
    # -- voices (Phase 2b) ---------------------------------------------------

    def add_voice_rows(self, rows: list[VoiceRow]) -> None:
        if rows:
            self._table(VOICES_TABLE, VoiceRow).add(rows)

    def match_voice(self, vector: list[float], k: int = 5) -> tuple[str, str, float] | None:
        """k-NN majority vote over voice rows; None if nobody has a profile."""
        table = self._table(VOICES_TABLE, VoiceRow)
        if table.count_rows() == 0:
            return None
        hits = table.search(vector).metric("cosine").limit(k).to_list()
        counts: dict[str, int] = {}
        for r in hits:
            counts[r["person_id"]] = counts.get(r["person_id"], 0) + 1
        winner = max(counts, key=counts.get)
        best = min(r["_distance"] for r in hits if r["person_id"] == winner)
        name = next(r["name"] for r in hits if r["person_id"] == winner)
        return winner, name, 1.0 - best

    def passive_voice_count(self, person_id: str) -> int:
        table = self._table(VOICES_TABLE, VoiceRow)
        return sum(
            1 for r in table.to_arrow().to_pylist()
            if r["person_id"] == person_id and r["source"] == "passive"
        )

    def prune_passive_voices(self, person_id: str, keep: int) -> None:
        """Delete oldest passive rows beyond `keep`; enrolled rows untouched."""
        table = self._table(VOICES_TABLE, VoiceRow)
        passive = sorted(
            (r for r in table.to_arrow().to_pylist()
             if r["person_id"] == person_id and r["source"] == "passive"),
            key=lambda r: r["created_at"],
        )
        for r in passive[: max(0, len(passive) - keep)]:
            escaped = r["voice_id"].replace("'", "''")
            table.delete(f"voice_id = '{escaped}'")
```

`config.py` (after the Perception block):

```python
    # Voice ID (Phase 2b) — ECAPA cosine scores run lower than face scores
    voice_threshold: float = 0.30   # below = unknown; within 0.05 under = "can't tell"
    voice_min_utterance_s: float = 1.0  # shorter audio -> can't tell
    voice_passive_cap: int = 10     # max passively-banked embeddings per person
```

- [ ] **Step 4: Run tests + lint** — `uv run pytest tests/test_voices_store.py -q && uv run ruff check src tests` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: voices table, store voice matching, voice-ID config knobs"`

---

### Task 3: `perception/voice.py` — SpeakerIdentifier + ECAPA

**Files:**
- Modify: `src/reachy_vec/perception/voice.py` (currently a docstring stub), `pyproject.toml`
- Test: `tests/test_voice.py` (create)

**Interfaces:**
- Consumes: `Store.match_voice` (Task 2), `Observation` + `BORDERLINE_MARGIN` from `perception/face.py`.
- Produces: `SpeakerIdentifier` protocol (`identify(audio) -> Observation | None`, `embed(audio) -> list[float] | None`); `EcapaSpeakerIdentifier(store, threshold=None, min_utterance_s=None)`.

- [ ] **Step 1: Move speechbrain to main dependencies**

Run: `uv add speechbrain` then remove the `[project.optional-dependencies]` `perception` block from `pyproject.toml` (delete the whole block — speechbrain was its only member).

- [ ] **Step 2: Write failing tests** — `tests/test_voice.py`. Test the decision logic by faking `embed` (never load speechbrain in tests):

```python
import numpy as np

from reachy_vec.perception.voice import EcapaSpeakerIdentifier
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow

SR = 16000


def enrolled_store(tmp_path) -> Store:
    store = Store(tmp_path / "db")
    store.add_voice_rows([VoiceRow(
        voice_id="p1:0", person_id="p1", name="Alice",
        vector=[1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 1),
        created_at="2026-07-08T00:00:00+00:00", source="enrolled",
    )])
    return store


def identifier(store, vector):
    ident = EcapaSpeakerIdentifier(store, threshold=0.30)
    ident.embed = lambda audio: vector  # bypass the model
    return ident


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def test_confident_match(tmp_path):
    obs = identifier(enrolled_store(tmp_path),
                     [1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 1)).identify(audio(2))
    assert (obs.person_id, obs.name) == ("p1", "Alice")


def test_below_threshold_is_unknown(tmp_path):
    # orthogonal vector -> cosine 0.0 < 0.30
    vec = [0.0, 1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 2)
    obs = identifier(enrolled_store(tmp_path), vec).identify(audio(2))
    assert obs is not None and obs.person_id is None


def test_borderline_is_cant_tell(tmp_path, monkeypatch):
    ident = EcapaSpeakerIdentifier(enrolled_store(tmp_path), threshold=0.30)
    ident.embed = lambda a: [1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 1)
    monkeypatch.setattr(ident._store, "match_voice", lambda v, k=5: ("p1", "Alice", 0.27))
    assert ident.identify(audio(2)) is None  # 0.27 within 0.05 under 0.30


def test_short_audio_is_cant_tell(tmp_path):
    real = EcapaSpeakerIdentifier(enrolled_store(tmp_path))
    assert real.embed(audio(0.5)) is None       # never loads the model
    assert real.identify(audio(0.5)) is None


def test_empty_voice_store_is_cant_tell(tmp_path):
    assert identifier(Store(tmp_path / "db"),
                      [1.0] + [0.0] * (VOICE_EMBEDDING_DIM - 1)).identify(audio(2)) is None
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_voice.py -q` → ImportError.

- [ ] **Step 4: Implement** `src/reachy_vec/perception/voice.py`:

```python
"""Speaker identification: ECAPA voice embeddings matched against the voices
table. Mirrors face.py: never guesses — below-threshold = unknown speaker,
borderline (within the shared margin under threshold) or too-short audio =
None ("can't tell", fusion falls back to face).
"""

import logging
from typing import Protocol

import numpy as np

from reachy_vec.config import settings
from reachy_vec.perception.face import BORDERLINE_MARGIN, Observation
from reachy_vec.store.db import Store

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpeakerIdentifier(Protocol):
    def identify(self, audio: np.ndarray | None) -> Observation | None: ...
    def embed(self, audio: np.ndarray | None) -> list[float] | None: ...


class EcapaSpeakerIdentifier:
    """speechbrain ECAPA-TDNN (spkrec-ecapa-voxceleb), 192-dim, lazy-loaded.

    Model-load failures degrade to "can't tell" forever (logged once) — the
    robot keeps working face-only.
    """

    def __init__(
        self,
        store: Store,
        threshold: float | None = None,
        min_utterance_s: float | None = None,
    ):
        self._store = store
        self._threshold = threshold if threshold is not None else settings.voice_threshold
        self._min_samples = int(
            SAMPLE_RATE
            * (min_utterance_s if min_utterance_s is not None else settings.voice_min_utterance_s)
        )
        self._model = None
        self._broken = False

    def _load(self):
        if self._model is None and not self._broken:
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb"
                )
            except Exception:
                logger.exception("ECAPA load failed - voice ID disabled")
                self._broken = True

    def embed(self, audio: np.ndarray | None) -> list[float] | None:
        if audio is None or len(audio) < self._min_samples:
            return None
        self._load()
        if self._model is None:
            return None
        import torch

        with torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(audio).unsqueeze(0))
        return emb.squeeze().tolist()

    def identify(self, audio: np.ndarray | None) -> Observation | None:
        vector = self.embed(audio)
        if vector is None:
            return None
        match = self._store.match_voice(vector)
        if match is None:
            return None  # nobody has a voice profile yet
        person_id, name, score = match
        if score >= self._threshold:
            return Observation(person_id=person_id, name=name, score=score)
        if score >= self._threshold - BORDERLINE_MARGIN:
            return None
        return Observation(person_id=None, name=None, score=score)
```

- [ ] **Step 5: Run tests + lint** — `uv run pytest tests/test_voice.py -q && uv run ruff check src tests` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat: ECAPA speaker identifier behind SpeakerIdentifier protocol"`

---

### Task 4: `perception/fusion.py` — TurnIdentity + fuse()

**Files:**
- Modify: `src/reachy_vec/perception/fusion.py` (stub), `src/reachy_vec/perception/face.py` (add `face_count`)
- Test: `tests/test_fusion.py` (create)

**Interfaces:**
- Produces: `TurnIdentity` frozen dataclass (`person_id: str | None`, `name: str | None`), constant `ANONYMOUS = TurnIdentity(None, None)`, `fuse(face_obs, voice_obs) -> TurnIdentity`. `Observation` gains `face_count: int = 1`.

- [ ] **Step 1: Write failing table-driven test** — `tests/test_fusion.py`:

```python
import pytest

from reachy_vec.perception.face import Observation
from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity, fuse

ALICE_FACE = Observation(person_id="p1", name="Alice", score=0.9)
BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.1)
UNKNOWN_FACE = Observation(person_id=None, name=None, score=0.1)


@pytest.mark.parametrize(
    ("face", "voice", "expected"),
    [
        # voice knows -> voice wins, even against a confident face
        (ALICE_FACE, BOB_VOICE, TurnIdentity("p2", "Bob")),
        (None, BOB_VOICE, TurnIdentity("p2", "Bob")),
        # confident unknown voice -> anonymous, face cannot override
        (ALICE_FACE, UNKNOWN_VOICE, ANONYMOUS),
        (None, UNKNOWN_VOICE, ANONYMOUS),
        # voice can't tell -> face decides
        (ALICE_FACE, None, TurnIdentity("p1", "Alice")),
        (UNKNOWN_FACE, None, ANONYMOUS),
        (None, None, ANONYMOUS),
    ],
)
def test_fusion_truth_table(face, voice, expected):
    assert fuse(face, voice) == expected
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_fusion.py -q` → ImportError.

- [ ] **Step 3: Implement.** `src/reachy_vec/perception/fusion.py`:

```python
"""Identity fusion: voice is the authority, face the tie-breaker; never guess.

Truth table (per utterance):
  voice known      -> that person (the speaker may be off-camera)
  voice unknown    -> anonymous (a stranger is talking, whoever is in frame)
  voice can't tell -> the recognized face, else anonymous
"""

from dataclasses import dataclass

from reachy_vec.perception.face import Observation


@dataclass(frozen=True)
class TurnIdentity:
    person_id: str | None
    name: str | None


ANONYMOUS = TurnIdentity(None, None)


def fuse(face_obs: Observation | None, voice_obs: Observation | None) -> TurnIdentity:
    if voice_obs is not None:
        if voice_obs.person_id is not None:
            return TurnIdentity(voice_obs.person_id, voice_obs.name)
        return ANONYMOUS
    if face_obs is not None and face_obs.person_id is not None:
        return TurnIdentity(face_obs.person_id, face_obs.name)
    return ANONYMOUS
```

In `face.py`, add the frame's face count to `Observation` (passive backfill needs "solo in frame"):

```python
@dataclass(frozen=True)
class Observation:
    person_id: str | None  # None = face present but unknown
    name: str | None
    score: float
    face_count: int = 1  # faces detected in the frame this came from
```

In `InsightFaceMatcher.embed`, record the count: after `faces = self._app.get(frame)` add `self.last_face_count = len(faces)` (and `self.last_face_count = 0` in the no-faces branch; initialize `self.last_face_count = 0` in `__init__`). In `observe`, pass `face_count=self.last_face_count` to both `Observation(...)` constructions.

- [ ] **Step 4: Run + lint** — `uv run pytest -q && uv run ruff check src tests` → PASS (default `face_count=1` keeps existing tests green).

- [ ] **Step 5: Commit** — `git commit -m "feat: identity fusion (voice authority, face tie-breaker) + frame face count"`

---

### Task 5: `ChatBrain` — per-turn identity, per-speaker distillation

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`, `tests/conftest.py` (`FakeBrain`), `src/reachy_vec/brain/loop.py` (no change needed — verify), `tests/test_chat_brain.py`
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Consumes: `TurnIdentity`, `ANONYMOUS` from `perception/fusion.py`.
- Produces: `ChatBrain.respond(question, identity: TurnIdentity | None = None, on_sentence=None) -> str` (replaces `speaker_name=`). `identity=None` ≡ anonymous. `FakeBrain.respond` mirrors the new signature and records `(question, identity)`.

- [ ] **Step 1: Update existing tests + add new ones.** In `tests/test_chat_brain.py`, mechanical change first: every `brain.respond(q, speaker_name="Yang")` becomes `brain.respond(q, identity=TurnIdentity("p1", "Yang"))` (import `from reachy_vec.perception.fusion import TurnIdentity`); calls relying on `begin_conversation("p1", "Yang")` for attribution now pass the identity explicitly. `test_save_note_without_person_is_refused` uses plain `brain.respond("remember this")` (anonymous) — unchanged. Then add:

```python
def test_turn_identity_switches_attribution(tmp_path):
    """Bob chimes into Alice's conversation; his note goes to Bob."""
    client = FakeLLMClient(messages=[
        FakeChoiceMessage(None, tool_calls=[
            FakeToolCall("save_note", '{"note": "Bob prefers tea"}')]),
        FakeChoiceMessage("Noted, Bob!"),
    ])
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("remember I prefer tea", identity=TurnIdentity("p2", "Bob"))
    store = brain._store
    hits = store.search_memories(FakeEmbedder().embed(["tea"])[0], person_id="p2", k=3)
    assert any("tea" in h.text for h in hits)
    assert store.search_memories(FakeEmbedder().embed(["tea"])[0], person_id="p1", k=3) == []


def test_anonymous_turn_skips_memories_and_refuses_notes(tmp_path):
    client = FakeLLMClient(messages=[
        FakeChoiceMessage(None, tool_calls=[FakeToolCall("save_note", '{"note": "x"}')]),
        FakeChoiceMessage("Sorry, I don't know who's asking."),
    ])
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("remember this", identity=None)  # anonymous despite owner
    assert brain._store.search_memories(
        FakeEmbedder().embed(["x"])[0], person_id="p1", k=3) == []


def test_distillation_covers_every_speaker(tmp_path):
    client = FakeLLMClient(messages=[
        FakeChoiceMessage("hi alice"), FakeChoiceMessage("hi bob"),
        FakeChoiceMessage("- Alice likes charts"),   # summary for Alice
        FakeChoiceMessage("- Bob is visiting"),      # summary for Bob
    ])
    brain = make_brain(tmp_path, client)
    brain.begin_conversation("p1", "Alice")
    brain.respond("hello", identity=TurnIdentity("p1", "Alice"))
    brain.respond("hello from bob", identity=TurnIdentity("p2", "Bob"))
    brain.end_conversation()
    emb = FakeEmbedder()
    assert brain._store.search_memories(emb.embed(["charts"])[0], person_id="p1", k=3)
    assert brain._store.search_memories(emb.embed(["visiting"])[0], person_id="p2", k=3)
```

(`make_brain` keeps `brain._store` reachable; if not, return the store from `make_brain` — follow the file's existing helper.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_chat_brain.py -q` → TypeError (unexpected keyword `identity`).

- [ ] **Step 3: Implement in `chat.py`.**

Replace the identity plumbing:

```python
from reachy_vec.perception.fusion import ANONYMOUS, TurnIdentity
```

In `__init__` / `reset`: add `self._turn = ANONYMOUS` and `self._participants: dict[str, str] = {}` (reset clears both). `begin_conversation` seeds `self._participants = {person_id: name} if person_id else {}`.

`respond` becomes:

```python
    def respond(
        self,
        question: str,
        identity: TurnIdentity | None = None,
        on_sentence: Callable[[str], None] | None = None,
    ) -> str:
        self._turn = identity or ANONYMOUS
        if self._turn.person_id:
            self._participants[self._turn.person_id] = self._turn.name
        vector = self._embedder.embed([question])[0]
        self._history.append(
            {
                "role": "user",
                "content": CONTEXT_TEMPLATE.format(
                    context=self._retrieve_docs(vector) or "(nothing relevant found)",
                    memories=self._retrieve_memories(vector),
                    speaker=self._turn.name or "User",
                    question=question,
                ),
            }
        )
        ...  # rest unchanged
```

`_retrieve_memories` uses `self._turn.person_id` / `self._turn.name` instead of `self._person_id` / `self._person_name`. `_tool_save_note` and `_tool_send_message` guard on `self._turn.person_id` and attribute to `self._turn` (`from_person=self._turn.person_id, from_name=self._turn.name or "someone"`). `_store_memories` gains a `person_id` parameter (callers: `_tool_save_note` passes `self._turn.person_id`; `_summarize_and_store` passes each participant's id); `_is_duplicate_memory` likewise takes `person_id`.

`end_conversation` condition becomes `if self._participants and self._exchanges > 0`. `_summarize_and_store` loops:

```python
    def _summarize_and_store(self) -> None:
        for person_id, name in self._participants.items():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": PERSONALITY},
                    *self._history,
                    {"role": "user", "content": SUMMARY_PROMPT.format(name=name or "them")},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if text.upper() == "NONE":
                continue
            notes = [line.lstrip("- ").strip() for line in text.splitlines() if line.strip()]
            self._store_memories([n for n in notes if n][:3], person_id=person_id)
```

Keep `self._person_id`/`self._person_name` (visit owner) — greeting continuity and `begin_conversation` semantics are unchanged.

- [ ] **Step 4: Update `FakeBrain`** in `tests/conftest.py`:

```python
    def respond(self, question: str, identity=None, on_sentence=None) -> str:
        if self._fail:
            raise RuntimeError("api down")
        self.asked.append((question, identity))
        reply = f"answer to {question}"
        if on_sentence is not None:
            on_sentence(reply)
        return reply
```

`tests/test_oracle.py` assertion `brain.asked == [("when is standup?", "Alice")]` will fail until Task 6 wires identity — update it in this task to expect `TurnIdentity("p1", "Alice")` only if Task 6 lands together; otherwise temporarily assert `brain.asked[0][0] == "when is standup?"` and tighten in Task 6. (`brain/loop.py` calls `brain.respond(question)` — anonymous — no change.)

- [ ] **Step 5: Full suite + lint** — `uv run pytest -q && uv run ruff check src tests` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat: ChatBrain per-turn identity; distillation covers every speaker"`

---

### Task 6: OracleLoop — voice ID per turn, passive backfill, enrollment phrase

**Files:**
- Modify: `src/reachy_vec/brain/oracle.py`, `tests/conftest.py` (add `FakeSpeakerIdentifier`), `tests/test_oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `Utterance` (Task 1), `fuse`/`TurnIdentity` (Task 4), `ChatBrain.respond(text, identity, on_sentence)` (Task 5), `Store` voice methods (Task 2), `SpeakerIdentifier` (Task 3).
- Produces: `OracleLoop(..., speaker_id=None, voice_passive_cap=10)` — `speaker_id=None` disables voice ID (face-only behavior, also the graceful-degradation path).

- [ ] **Step 1: Add `FakeSpeakerIdentifier`** to `tests/conftest.py`:

```python
class FakeSpeakerIdentifier:
    """Scripted voice observations (repeats last); records banked embeddings."""

    def __init__(self, observations: list | None = None, embedding=None):
        self._observations = observations or []
        self._embedding = embedding  # what embed() returns; None = too short/broken
        self.embed_calls = 0

    def identify(self, audio):
        if not self._observations:
            return None
        return self._observations.pop(0)

    def embed(self, audio):
        self.embed_calls += 1
        return self._embedding
```

- [ ] **Step 2: Write failing oracle tests** (extend `make_loop` with `speaker_id=None` passthrough; import `TurnIdentity`, `VoiceRow`, `VOICE_EMBEDDING_DIM`):

```python
BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.05)
VOICE_VEC = [0.5] * VOICE_EMBEDDING_DIM


def test_turn_identity_follows_voice(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"],
        speaker_id=FakeSpeakerIdentifier([BOB_VOICE]),
    )
    loop.run_once()
    assert brain.asked == [("a question", TurnIdentity("p2", "Bob"))]


def test_unknown_voice_is_anonymous_despite_face(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path, sights=[ALICE], utterances=["save this"],
        speaker_id=FakeSpeakerIdentifier([UNKNOWN_VOICE]),
    )
    loop.run_once()
    assert brain.asked[0][1] == TurnIdentity(None, None)


def test_no_speaker_id_falls_back_to_face(tmp_path):
    loop, _, _, _, brain = make_loop(tmp_path, sights=[ALICE], utterances=["hi"])
    loop.run_once()
    assert brain.asked == [("hi", TurnIdentity("p1", "Alice"))]


def test_passive_backfill_banks_solo_confident_face(tmp_path):
    ident = FakeSpeakerIdentifier([None], embedding=VOICE_VEC)
    loop, _, _, store, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"], speaker_id=ident)
    loop.run_once()
    assert store.passive_voice_count("p1") == 1


def test_no_backfill_when_voice_is_someone_else(tmp_path):
    ident = FakeSpeakerIdentifier([BOB_VOICE], embedding=VOICE_VEC)
    loop, _, _, store, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"], speaker_id=ident)
    loop.run_once()
    assert store.passive_voice_count("p1") == 0
    assert store.passive_voice_count("p2") == 0


def test_no_backfill_with_two_faces_in_frame(tmp_path):
    crowded = Observation(person_id="p1", name="Alice", score=0.9, face_count=2)
    ident = FakeSpeakerIdentifier([None], embedding=VOICE_VEC)
    loop, _, _, store, _ = make_loop(
        tmp_path, sights=[crowded], utterances=["a question"], speaker_id=ident)
    loop.run_once()
    assert store.passive_voice_count("p1") == 0


def test_backfill_respects_cap(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows([VoiceRow(
        voice_id=f"p1:{i}", person_id="p1", name="Alice", vector=VOICE_VEC,
        created_at=f"2026-07-08T00:00:{i:02d}+00:00", source="passive",
    ) for i in range(10)])
    ident = FakeSpeakerIdentifier([None], embedding=VOICE_VEC)
    loop, _, _, _, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"],
        speaker_id=ident, store=store)
    loop.run_once()
    assert store.passive_voice_count("p1") == 10  # capped, not 11


def test_enrollment_captures_voice_phrase(tmp_path):
    ident = FakeSpeakerIdentifier(embedding=VOICE_VEC)
    loop, speaker, _, store, _ = make_loop(
        tmp_path,
        sights=[UNKNOWN, UNKNOWN],
        utterances=["yes please", "Bob", "yes", "the quick brown fox"],
        enroll_result="p9", speaker_id=ident,
    )
    assert loop.run_once() == "enrolled"
    assert any("voice" in s.lower() for s in speaker.spoken)
    assert store.match_voice(VOICE_VEC)[0] == "p9"


def test_enrollment_survives_missing_voice(tmp_path):
    ident = FakeSpeakerIdentifier(embedding=None)  # too short / model broken
    loop, _, _, store, _ = make_loop(
        tmp_path,
        sights=[UNKNOWN, UNKNOWN],
        utterances=["yes please", "Bob", "yes"],  # then silence for the phrase
        enroll_result="p9", speaker_id=ident,
    )
    assert loop.run_once() == "enrolled"          # face-only enrollment still succeeds
    assert store.match_voice(VOICE_VEC) is None
```

Also update `make_loop` to accept and forward `speaker_id=None`, and tighten the Task-5 placeholder assert in `test_known_person_greet_question_answer_goodbye` to `brain.asked == [("when is standup?", TurnIdentity("p1", "Alice"))]`.

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_oracle.py -q` → TypeError (`speaker_id` unexpected).

- [ ] **Step 4: Implement in `oracle.py`.**

Constructor gains `speaker_id=None, voice_passive_cap: int = 10`; store both. Imports: `from reachy_vec.perception.fusion import fuse`.

`run_once` passes the face observation into `_converse(obs)` (it already has `obs`). `_converse` becomes:

```python
    def _converse(self, face_obs) -> None:
        person_id, name = face_obs.person_id, face_obs.name
        self._brain.begin_conversation(person_id, name)
        # greeting/cooldown/message delivery unchanged, keyed on person_id
        ...
        while True:
            self._body.perform("listen")
            utterance = self._transcriber.listen_once(self._silence_timeout_s)
            if utterance is None:
                self._body.perform("goodbye")
                self._brain.end_conversation()
                return
            voice_obs = self._identify_voice(utterance.audio)
            identity = fuse(face_obs, voice_obs)
            try:
                self._brain.respond(
                    utterance.text, identity=identity, on_sentence=self._speaker.speak
                )
                self._body.perform("nod")
            except Exception:
                logger.exception("brain.respond failed")
                self._speaker.speak(APOLOGY)
            self._maybe_bank_voice(face_obs, voice_obs, utterance.audio)

    def _identify_voice(self, audio):
        if self._speaker_id is None:
            return None
        try:
            return self._speaker_id.identify(audio)
        except Exception:
            logger.exception("speaker ID failed - treating as can't tell")
            return None

    def _maybe_bank_voice(self, face_obs, voice_obs, audio) -> None:
        """Passively grow the voice profile of a confident solo face match."""
        if self._speaker_id is None or face_obs.person_id is None:
            return
        if face_obs.face_count != 1:
            return
        if voice_obs is not None and voice_obs.person_id != face_obs.person_id:
            return
        try:
            vector = self._speaker_id.embed(audio)
            if vector is None:
                return
            from datetime import datetime, timezone  # match file's UTC import style
            from uuid import uuid4

            from reachy_vec.store.schemas import VoiceRow

            self._store.add_voice_rows([VoiceRow(
                voice_id=f"{face_obs.person_id}:{uuid4().hex[:8]}",
                person_id=face_obs.person_id,
                name=face_obs.name,
                vector=vector,
                created_at=datetime.now(timezone.utc).isoformat(),
                source="passive",
            )])
            self._store.prune_passive_voices(face_obs.person_id, keep=self._voice_passive_cap)
        except Exception:
            logger.exception("passive voice backfill failed - skipping")
```

(Use whatever UTC import style ruff left in the file — `from datetime import UTC, datetime` — and module-level imports where they don't create cycles.)

In `_offer_enroll`, after `self._record_greeting(person_id)` and before "All set":

```python
                self._capture_voice(person_id, name)

    def _capture_voice(self, person_id: str, name: str) -> None:
        if self._speaker_id is None:
            return
        self._speaker.speak("Now say a sentence so I learn your voice - anything you like.")
        utterance = self._transcriber.listen_once(10)
        vector = self._speaker_id.embed(utterance.audio) if utterance else None
        if vector is None:
            self._speaker.speak("No worries - I'll learn your voice as we talk.")
            return
        from uuid import uuid4

        from reachy_vec.store.schemas import VoiceRow

        self._store.add_voice_rows([VoiceRow(
            voice_id=f"{person_id}:{uuid4().hex[:8]}",
            person_id=person_id, name=name, vector=vector,
            created_at=datetime.now(UTC).isoformat(), source="enrolled",
        )])
```

- [ ] **Step 5: Full suite + lint** — `uv run pytest -q && uv run ruff check src tests` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat: Oracle fuses voice+face per turn; passive backfill; voice enrollment"`

---

### Task 7: Wiring, docs, smoke checklist

**Files:**
- Modify: `src/reachy_vec/cli/run.py`, `docs/architecture.md`, `docs/pipelines.md`, `docs/configuration.md`, `docs/testing.md`, `CLAUDE.md`

**Interfaces:** consumes everything above; no new interfaces.

- [ ] **Step 1: Wire into `run.py`.** After `matcher = InsightFaceMatcher(store)` add:

```python
    from reachy_vec.perception.voice import EcapaSpeakerIdentifier

    speaker_id = EcapaSpeakerIdentifier(store)
```

Warm-up block gains `speaker_id.embed(np.zeros(16000 * 2, dtype=np.float32))` (guard with try/except like the OpenAI warm-up; `import numpy as np` at top of function). OracleLoop call gains `speaker_id=speaker_id, voice_passive_cap=settings.voice_passive_cap`.

- [ ] **Step 2: Docs.**
  - `docs/pipelines.md`: add ECAPA row to the models table (`Speaker ID | speechbrain ECAPA spkrec-ecapa-voxceleb, 192-dim | local | perception/voice.py | REACHY_VEC_VOICE_THRESHOLD`); add a "Voice identity pipeline" subsection after the face pipeline covering identify → fuse truth table → passive backfill; note in the chat pipeline that retrieval/tools follow the per-turn identity.
  - `docs/configuration.md`: add the three new settings to a "Voice ID" table; move `speechbrain` note; drop the "unknown-face stable polls" style constants list additions as needed.
  - `docs/architecture.md`: diagram gains `voice: ECAPA speaker ID` under ears; tables list gains `voices`; run-loop THINKING para notes per-turn fusion.
  - `docs/testing.md`: manual checklist gains: "Two enrolled people take turns talking (one out of frame) → each reply addresses the actual speaker; 'remember I…' lands on the speaker's memories"; troubleshooting gains `REACHY_VEC_VOICE_THRESHOLD` tuning note.
  - `CLAUDE.md`: protocol list gains `SpeakerIdentifier`; table list gains `voices`.

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q && uv run ruff check src tests` → PASS. Then `uv run reachy-vec --help` → command list renders (import graph intact).

- [ ] **Step 4: Commit** — `git commit -m "feat: wire speaker ID into run; document voice pipeline and config"`

---

## Self-review notes

- Spec coverage: Utterance (T1), voices table + knobs (T2), ECAPA + degradation (T3), fusion table + supersession (T4), per-turn brain + per-speaker distillation + anonymous chat CLI (T5), loop wiring + backfill guardrails + enrollment phrase + cap (T6), run wiring + docs + smoke tests (T7). Privacy: only embeddings stored (T6 code stores vectors only).
- `speechbrain` main-dep move: T3 step 1.
- Type consistency: `TurnIdentity(person_id, name)` used identically in T4–T6; `Store.match_voice` returns the same triple shape as `match_face`.

from reachy_vec.brain.oracle import OracleLoop
from reachy_vec.brain.rag import Answer
from reachy_vec.perception.face import Observation
from reachy_vec.store.db import Store

from tests.conftest import FakeBody, FakeSpeaker, FakeTranscriber

ALICE = Observation(person_id="p1", name="Alice", score=0.9)
UNKNOWN = Observation(person_id=None, name=None, score=0.1)


def make_loop(tmp_path, *, sights, utterances, enroll_result="p9", clock=None, store=None):
    sights_iter = iter(sights)
    speaker, body = FakeSpeaker(), FakeBody()
    store = store or Store(tmp_path / "db")
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
    store = Store(tmp_path / "db")
    first, _, _, _ = make_loop(tmp_path, sights=[], utterances=[], store=store)
    first._record_greeting("p1")  # greeted "now" per the fake clock
    loop, speaker, body, _ = make_loop(tmp_path, sights=[ALICE], utterances=[], store=store)
    assert loop.run_once() == "conversation"
    assert not any("Alice" in s for s in speaker.spoken)      # silent acknowledgment
    assert "acknowledge" in body.motions


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

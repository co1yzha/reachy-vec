from reachy_vec.brain.oracle import OracleLoop
from reachy_vec.perception.face import Observation
from reachy_vec.store.db import Store
from tests.conftest import FakeBody, FakeBrain, FakeSpeaker, FakeTranscriber

ALICE = Observation(person_id="p1", name="Alice", score=0.9)
UNKNOWN = Observation(person_id=None, name=None, score=0.1)


def make_loop(tmp_path, *, sights, utterances, enroll_result="p9", store=None, brain=None):
    sights_iter = iter(sights)
    speaker, body = FakeSpeaker(), FakeBody()
    store = store or Store(tmp_path / "db")
    brain = brain or FakeBrain()
    loop = OracleLoop(
        sight=lambda: next(sights_iter, None),
        transcriber=FakeTranscriber(utterances),
        speaker=speaker,
        body=body,
        brain=brain,
        enroll_capture=lambda name: enroll_result,
        store=store,
        clock=lambda: 1000.0,
        unknown_stable_polls=2,
    )
    return loop, speaker, body, store, brain


def test_known_person_greet_question_answer_goodbye(tmp_path):
    loop, speaker, body, store, brain = make_loop(
        tmp_path, sights=[ALICE], utterances=["when is standup?"]
    )
    assert loop.run_once() == "conversation"
    assert any("Alice" in s for s in speaker.spoken)          # spoken greeting
    assert any("answer to when is standup?" in s for s in speaker.spoken)
    assert "greet" in body.motions and "goodbye" in body.motions
    assert store.get_last_greeted("p1") is not None           # cooldown recorded
    assert brain.begun == [("p1", "Alice")]                   # conversation opened
    assert brain.ended == 1                                   # memories distilled
    assert brain.asked == [("when is standup?", "Alice")]     # speaker attributed


def test_cooldown_suppresses_spoken_greeting(tmp_path):
    store = Store(tmp_path / "db")
    first, _, _, _, _ = make_loop(tmp_path, sights=[], utterances=[], store=store)
    first._record_greeting("p1")  # greeted "now" per the fake clock
    loop, speaker, body, _, _ = make_loop(tmp_path, sights=[ALICE], utterances=[], store=store)
    assert loop.run_once() == "conversation"
    assert not any("Alice" in s for s in speaker.spoken)      # silent acknowledgment
    assert "acknowledge" in body.motions


def test_unknown_face_enrolls_on_yes_and_confirm(tmp_path):
    loop, speaker, body, _, _ = make_loop(
        tmp_path,
        sights=[UNKNOWN, UNKNOWN],                 # stable unknown (2 polls)
        utterances=["yes please", "Bob", "yes"],   # offer-yes, name, confirm
    )
    assert loop.run_once() == "enrolled"
    assert any("Bob" in s for s in speaker.spoken)            # confirmation used name


def test_unknown_face_declines_enrollment(tmp_path):
    loop, speaker, _, _, _ = make_loop(
        tmp_path, sights=[UNKNOWN, UNKNOWN], utterances=["no thanks"]
    )
    assert loop.run_once() == "enroll-declined"


def test_silence_ends_conversation_with_goodbye(tmp_path):
    loop, _, body, _, _ = make_loop(tmp_path, sights=[ALICE], utterances=[])
    assert loop.run_once() == "conversation"
    assert body.motions[-1] == "goodbye"


def test_brain_failure_apologizes_and_continues(tmp_path):
    loop, speaker, _, _, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["hello?"], brain=FakeBrain(fail=True)
    )
    assert loop.run_once() == "conversation"
    assert any("sorry" in s.lower() for s in speaker.spoken)


def test_no_face_at_all(tmp_path):
    loop, _, _, _, _ = make_loop(tmp_path, sights=[None, None, None], utterances=[])
    assert loop.run_once() == "no-face"


def test_pending_messages_delivered_on_greet(tmp_path):
    from reachy_vec.store.schemas import MessageRow

    store = Store(tmp_path / "db")
    store.add_message(
        MessageRow(
            message_id="msg1",
            from_person="p2",
            from_name="Bob",
            to_person="p1",
            to_name="Alice",
            text="the meeting moved to 3",
            created_at="2026-07-07T00:00:00+00:00",
            delivered_at="",
        )
    )
    loop, speaker, _, _, _ = make_loop(tmp_path, sights=[ALICE], utterances=[], store=store)
    assert loop.run_once() == "conversation"
    assert any("Bob left you a message: the meeting moved to 3" in s for s in speaker.spoken)
    assert store.pending_messages_for("p1") == []  # marked delivered


def test_sleeps_after_idle_and_wakes_on_face(tmp_path):
    now = {"t": 1000.0}
    sights: list = []
    speaker, body = FakeSpeaker(), FakeBody()
    loop = OracleLoop(
        sight=lambda: sights.pop(0) if sights else None,
        transcriber=FakeTranscriber([]),
        speaker=speaker,
        body=body,
        brain=FakeBrain(),
        enroll_capture=lambda name: None,
        store=Store(tmp_path / "db"),
        clock=lambda: now["t"],
        idle_sleep_s=300.0,
    )
    assert loop.run_once() == "no-face"          # t=1000: recent face-time, stays awake
    assert "sleep" not in body.motions
    now["t"] = 1400.0                            # 400s idle > 300s threshold
    assert loop.run_once() == "no-face"
    assert body.motions == ["sleep"]
    assert loop.run_once() == "no-face"          # still asleep: no second sleep motion
    assert body.motions == ["sleep"]
    sights.append(ALICE)                         # someone walks up
    assert loop.run_once() == "conversation"
    assert body.motions[1] == "wake"             # woke before greeting

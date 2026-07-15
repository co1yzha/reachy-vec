from reachy_vec.brain.oracle import OracleLoop
from reachy_vec.perception.face import Observation
from reachy_vec.perception.fusion import TurnIdentity
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, VoiceRow
from tests.conftest import (
    FakeBargeInMonitor,
    FakeBody,
    FakeBrain,
    FakeSpeaker,
    FakeSpeakerIdentifier,
    FakeTranscriber,
)

ALICE = Observation(person_id="p1", name="Alice", score=0.9)
UNKNOWN = Observation(person_id=None, name=None, score=0.1)
BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.05)
VOICE_VEC = [0.5] * VOICE_EMBEDDING_DIM


def make_loop(
    tmp_path,
    *,
    sights,
    utterances,
    enroll_result="p9",
    store=None,
    brain=None,
    speaker_id=None,
    barge_in_factory=None,
):
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
        speaker_id=speaker_id,
        barge_in_factory=barge_in_factory,
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
    assert brain.asked == [("when is standup?", TurnIdentity("p1", "Alice"))]  # attributed


def test_barge_in_stops_speaker_and_skips_nod(tmp_path):
    from reachy_vec.brain.chat import SpeechInterrupted

    monitor = FakeBargeInMonitor()

    class InterruptingBrain:
        def begin_conversation(self, *a):
            pass

        def end_conversation(self):
            pass

        def respond(self, question, identity=None, on_sentence=None):
            monitor.trip()  # user starts talking mid-reply
            try:
                on_sentence("half a sentence")  # guarded -> raises
            except SpeechInterrupted:
                return "half a sentence"  # ChatBrain swallows it internally
            return "full reply"

    loop, speaker, body, _store, _brain = make_loop(
        tmp_path,
        sights=[ALICE],
        utterances=["a question", None],
        brain=InterruptingBrain(),
        barge_in_factory=lambda: monitor,
    )
    assert loop.run_once() == "conversation"
    assert monitor.started == 1
    assert monitor.stopped == 1
    assert speaker.stopped == 1          # on_fire -> speaker.stop
    assert "half a sentence" not in speaker.spoken  # guarded sentence never spoken
    assert "nod" not in body.motions     # nod skipped on interrupt


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
    from reachy_vec.brain.oracle import WAKE_LINES

    assert body.motions[1:3] == ["wake", "wakeup"]   # woke with a flourish, before greeting
    assert speaker.spoken[0] in WAKE_LINES           # announced it's awake, before the greeting
    assert len(speaker.spoken) >= 2                  # normal greeting still follows


# -- Phase 2b: voice fusion, passive backfill, voice enrollment ---------------


def test_turn_identity_follows_voice(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path,
        sights=[ALICE],
        utterances=["a question"],
        speaker_id=FakeSpeakerIdentifier([BOB_VOICE]),
    )
    loop.run_once()
    assert brain.asked == [("a question", TurnIdentity("p2", "Bob"))]


def test_unknown_voice_is_anonymous_despite_face(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path,
        sights=[ALICE],
        utterances=["save this"],
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
        tmp_path, sights=[ALICE], utterances=["a question"], speaker_id=ident
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 1


def test_no_backfill_when_voice_is_someone_else(tmp_path):
    ident = FakeSpeakerIdentifier([BOB_VOICE], embedding=VOICE_VEC)
    loop, _, _, store, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"], speaker_id=ident
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 0
    assert store.passive_voice_count("p2") == 0


def test_no_backfill_with_two_faces_in_frame(tmp_path):
    crowded = Observation(person_id="p1", name="Alice", score=0.9, face_count=2)
    ident = FakeSpeakerIdentifier([None], embedding=VOICE_VEC)
    loop, _, _, store, _ = make_loop(
        tmp_path, sights=[crowded], utterances=["a question"], speaker_id=ident
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 0


def test_backfill_respects_cap(tmp_path):
    store = Store(tmp_path / "db")
    store.add_voice_rows(
        [
            VoiceRow(
                voice_id=f"p1:{i}",
                person_id="p1",
                name="Alice",
                vector=VOICE_VEC,
                created_at=f"2026-07-08T00:00:{i:02d}+00:00",
                source="passive",
            )
            for i in range(10)
        ]
    )
    ident = FakeSpeakerIdentifier([None], embedding=VOICE_VEC)
    loop, _, _, _, _ = make_loop(
        tmp_path, sights=[ALICE], utterances=["a question"], speaker_id=ident, store=store
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 10  # capped, not 11


def test_enrollment_captures_voice_phrase(tmp_path):
    ident = FakeSpeakerIdentifier(embedding=VOICE_VEC)
    loop, speaker, _, store, _ = make_loop(
        tmp_path,
        sights=[UNKNOWN, UNKNOWN],
        utterances=["yes please", "Bob", "yes", "the quick brown fox"],
        enroll_result="p9",
        speaker_id=ident,
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
        enroll_result="p9",
        speaker_id=ident,
    )
    assert loop.run_once() == "enrolled"  # face-only enrollment still succeeds
    assert store.match_voice(VOICE_VEC) is None

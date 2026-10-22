import numpy as np
import pytest

from reachy_vec.audio.listen import Utterance
from reachy_vec.brain.oracle import WAKE_LINES, OracleLoop, split_wake
from reachy_vec.perception.fusion import TurnIdentity
from reachy_vec.perception.voice import Observation
from reachy_vec.store.db import Store
from reachy_vec.store.schemas import VOICE_EMBEDDING_DIM, MessageRow, VoiceRow
from tests.conftest import (
    FakeBargeInMonitor,
    FakeBody,
    FakeBrain,
    FakeSpeaker,
    FakeSpeakerIdentifier,
    FakeTranscriber,
)

ALICE_VOICE = Observation(person_id="p1", name="Alice", score=0.9)
BOB_VOICE = Observation(person_id="p2", name="Bob", score=0.5)
BORDERLINE_ALICE = Observation(person_id="p1", name="Alice", score=0.32)  # just over 0.30
UNKNOWN_VOICE = Observation(person_id=None, name=None, score=0.05)
VOICE_VEC = [0.5] * VOICE_EMBEDDING_DIM
AUDIO = np.zeros(32000, dtype=np.float32)


def said(text: str) -> Utterance:
    return Utterance(text=text, audio=AUDIO)


def make_loop(
    tmp_path,
    *,
    utterances,
    voices=None,
    embedding=VOICE_VEC,
    store=None,
    brain=None,
    barge_in_factory=None,
    settle_s=0.0,
    sleep=None,
    **kwargs,
):
    """voices: scripted identify() results in call order (None = can't tell);
    voices=None disables voice ID entirely."""
    speaker, body = FakeSpeaker(), FakeBody()
    store = store or Store(tmp_path / "db")
    brain = brain or FakeBrain()
    loop = OracleLoop(
        transcriber=FakeTranscriber(utterances),
        speaker=speaker,
        body=body,
        brain=brain,
        store=store,
        speaker_id=FakeSpeakerIdentifier(voices, embedding=embedding)
        if voices is not None
        else None,
        clock=kwargs.pop("clock", lambda: 1000.0),
        barge_in_factory=barge_in_factory,
        settle_s=settle_s,
        sleep=sleep,
        **kwargs,
    )
    return loop, speaker, body, store, brain


# -- wake word -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reachy", ""),
        ("Hey Reachy, what's the weather?", "what's the weather?"),
        ("richie when is standup", "when is standup"),   # common STT mishearing
        ("REACHY!", ""),
        ("I was reaching for the cup", None),              # not a whole-word match
        ("when is standup?", None),
    ],
)
def test_split_wake(text, expected):
    assert split_wake(text) == expected


def test_speech_without_wake_word_is_ignored(tmp_path):
    loop, speaker, body, _, brain = make_loop(
        tmp_path, utterances=["so anyway the meeting is at three"]
    )
    assert loop.run_once() == "idle"
    assert speaker.spoken == [] and brain.asked == [] and brain.begun == []


def test_silence_is_idle(tmp_path):
    loop, *_ = make_loop(tmp_path, utterances=[])
    assert loop.run_once() == "idle"


# -- known voice -----------------------------------------------------------


def test_known_voice_greet_question_answer_goodbye(tmp_path):
    loop, speaker, body, store, brain = make_loop(
        tmp_path,
        utterances=[said("Hey Reachy"), said("when is standup?")],
        voices=[ALICE_VOICE, ALICE_VOICE],
    )
    assert loop.run_once() == "conversation"
    assert any("Hi Alice" in s for s in speaker.spoken)       # greeted by voice
    assert any("answer to when is standup?" in s for s in speaker.spoken)
    assert "greet" in body.motions and body.motions[-1] == "goodbye"
    assert store.get_last_greeted("p1") is not None           # cooldown recorded
    assert brain.begun == [("p1", "Alice")]
    assert brain.ended == 1                                   # memories distilled
    assert brain.asked == [("when is standup?", TurnIdentity("p1", "Alice"))]


def test_question_in_same_breath_as_wake_word(tmp_path):
    loop, speaker, _, _, brain = make_loop(
        tmp_path, utterances=[said("Reachy, what's the time?")], voices=[ALICE_VOICE]
    )
    assert loop.run_once() == "conversation"
    assert brain.asked == [("what's the time?", TurnIdentity("p1", "Alice"))]


def test_cooldown_suppresses_spoken_greeting(tmp_path):
    store = Store(tmp_path / "db")
    store.set_last_greeted("p1", "1970-01-01T00:16:40+00:00")  # = clock 1000.0
    loop, speaker, body, _, _ = make_loop(
        tmp_path, utterances=[said("Reachy")], voices=[ALICE_VOICE], store=store
    )
    assert loop.run_once() == "conversation"
    assert not any("Alice" in s for s in speaker.spoken)
    assert "acknowledge" in body.motions


def test_pending_messages_delivered_on_greet(tmp_path):
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
    loop, speaker, _, _, _ = make_loop(
        tmp_path, utterances=[said("Reachy")], voices=[ALICE_VOICE], store=store
    )
    assert loop.run_once() == "conversation"
    assert any("Bob left you a message: the meeting moved to 3" in s for s in speaker.spoken)
    assert store.pending_messages_for("p1") == []


# -- can't tell / anonymous ------------------------------------------------


def test_cant_tell_on_wake_then_recognized_mid_conversation(tmp_path):
    """A bare "Reachy" is too short to embed; the next turn identifies them."""
    loop, speaker, _, _, brain = make_loop(
        tmp_path,
        utterances=[said("Reachy"), said("what's on today?")],
        voices=[None, ALICE_VOICE],
    )
    assert loop.run_once() == "conversation"
    assert speaker.spoken[0].startswith("Yes?")
    assert brain.begun == [(None, None)]
    assert brain.asked == [("what's on today?", TurnIdentity("p1", "Alice"))]


def test_voice_identity_per_turn(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path,
        utterances=[said("Reachy"), said("one"), said("two")],
        voices=[ALICE_VOICE, ALICE_VOICE, BOB_VOICE],
    )
    loop.run_once()
    assert [identity for _, identity in brain.asked] == [
        TurnIdentity("p1", "Alice"),
        TurnIdentity("p2", "Bob"),       # Bob chimed in
    ]


def test_no_speaker_id_everyone_is_anonymous(tmp_path):
    loop, _, _, _, brain = make_loop(tmp_path, utterances=["Reachy", "hi"])
    assert loop.run_once() == "conversation"
    assert brain.asked == [("hi", TurnIdentity(None, None))]


def test_brain_failure_apologizes_and_continues(tmp_path):
    loop, speaker, _, _, _ = make_loop(
        tmp_path, utterances=["Reachy", "hello?"], brain=FakeBrain(fail=True)
    )
    assert loop.run_once() == "conversation"
    assert any("sorry" in s.lower() for s in speaker.spoken)


def test_barge_in_stops_speaker_and_skips_nod(tmp_path):
    from reachy_vec.brain.chat import SpeechInterrupted

    monitor = FakeBargeInMonitor()

    class InterruptingBrain(FakeBrain):
        def respond(self, question, identity=None, on_sentence=None):
            monitor.trip()  # user starts talking mid-reply
            try:
                on_sentence("half a sentence")  # guarded -> raises
            except SpeechInterrupted:
                return "half a sentence"  # ChatBrain swallows it internally
            return "full reply"

    loop, speaker, body, _, _ = make_loop(
        tmp_path,
        utterances=["Reachy", "a question"],
        brain=InterruptingBrain(),
        barge_in_factory=lambda: monitor,
    )
    assert loop.run_once() == "conversation"
    assert monitor.started == 1 and monitor.stopped == 1
    assert speaker.stopped == 1
    assert "half a sentence" not in speaker.spoken
    assert "nod" not in body.motions


# -- voice enrollment ------------------------------------------------------


def test_unknown_voice_enrolls_on_yes_and_confirm(tmp_path):
    loop, speaker, body, store, brain = make_loop(
        tmp_path, auto_offer=True,
        utterances=[
            said("Hi Reachy, I'm new here"),   # wake: stranger's voice
            said("yes please"), said("Bob"), said("yes"),
            said("I work on the data platform"), said("and I like coffee"),
        ],
        voices=[UNKNOWN_VOICE],
    )
    assert loop.run_once() == "enrolled"
    assert any("Nice to meet you, Bob" in s for s in speaker.spoken)
    person_id, name, _ = store.match_voice(VOICE_VEC)
    assert name == "Bob"
    voices = store.dump_tables()["voices"]
    # wake utterance + two enrollment sentences, all "enrolled"
    assert len(voices) == 3 and {v["source"] for v in voices} == {"enrolled"}
    assert store.get_last_greeted(person_id) is not None
    assert brain.begun == [(person_id, "Bob")]   # conversation continues after enrolling


def test_unknown_voice_declines_enrollment(tmp_path):
    loop, _, _, store, _ = make_loop(
        tmp_path, auto_offer=True, utterances=[said("Reachy"), said("no thanks")],
        voices=[UNKNOWN_VOICE],
    )
    assert loop.run_once() == "enroll-declined"
    assert store.match_voice(VOICE_VEC) is None


def test_enrollment_fails_without_any_usable_voice(tmp_path):
    loop, speaker, _, store, _ = make_loop(
        tmp_path, auto_offer=True,
        utterances=[said("Reachy"), said("yes"), said("Bob"), said("yes")],  # then silence
        voices=[UNKNOWN_VOICE],
        embedding=None,  # too short / model broken
    )
    assert loop.run_once() == "enroll-declined"
    assert any("couldn't catch your voice" in s for s in speaker.spoken)
    assert store.match_voice(VOICE_VEC) is None


def test_stranger_after_bare_wake_is_offered_enrollment_once(tmp_path):
    loop, speaker, _, store, _ = make_loop(
        tmp_path, auto_offer=True,
        utterances=[said("Reachy"), said("what's the weather?"), said("no"), said("thanks")],
        voices=[None, UNKNOWN_VOICE, UNKNOWN_VOICE],
    )
    assert loop.run_once() == "conversation"
    assert sum("remember your voice" in s for s in speaker.spoken) == 1
    assert store.match_voice(VOICE_VEC) is None


# -- passive banking ---------------------------------------------------------


def test_passive_backfill_banks_clear_match(tmp_path):
    loop, _, _, store, _ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("a question")],
        voices=[ALICE_VOICE, ALICE_VOICE], voice_threshold=0.30,
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 1


def test_no_backfill_for_borderline_match(tmp_path):
    loop, _, _, store, _ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("a question")],
        voices=[ALICE_VOICE, BORDERLINE_ALICE], voice_threshold=0.30,
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 0


def test_no_backfill_when_cant_tell(tmp_path):
    loop, _, _, store, _ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("a question")], voices=[ALICE_VOICE, None],
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
    loop, *_ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("a question")],
        voices=[ALICE_VOICE, ALICE_VOICE], store=store,
    )
    loop.run_once()
    assert store.passive_voice_count("p1") == 10  # capped, not 11


# -- sleep/wake, settle --------------------------------------------------------


def test_sleeps_after_idle_and_wakes_on_wake_word(tmp_path):
    now = {"t": 1000.0}
    loop, speaker, body, _, _ = make_loop(
        tmp_path, utterances=[], voices=[ALICE_VOICE],
        clock=lambda: now["t"], idle_sleep_s=300.0,
    )
    assert loop.run_once() == "idle"              # t=1000: recently awake
    assert "sleep" not in body.motions
    now["t"] = 1400.0                             # 400s idle > 300s threshold
    assert loop.run_once() == "idle"
    assert body.motions == ["sleep"]
    assert loop.run_once() == "idle"              # still asleep: no second sleep motion
    assert body.motions == ["sleep"]
    loop._transcriber = FakeTranscriber([said("Reachy")])  # someone calls it
    assert loop.run_once() == "conversation"
    assert body.motions[1:3] == ["wake", "wakeup"]
    assert speaker.spoken[0] in WAKE_LINES
    assert any("Hi Alice" in s for s in speaker.spoken)


def test_start_asleep_wakes_with_flourish_on_first_wake_word(tmp_path):
    loop, speaker, body, _, _ = make_loop(
        tmp_path, utterances=["Reachy"], start_asleep=True
    )
    assert loop.run_once() == "conversation"
    assert body.motions[0:2] == ["wake", "wakeup"]
    assert speaker.spoken[0] in WAKE_LINES


def test_ignored_speech_does_not_wake_a_sleeping_robot(tmp_path):
    loop, speaker, body, _, _ = make_loop(
        tmp_path, utterances=["chatting nearby"], start_asleep=True
    )
    assert loop.run_once() == "idle"
    assert body.motions == [] and speaker.spoken == []


def test_settles_before_each_conversation_listen(tmp_path):
    """The mic opens a beat after the robot stops talking; without the pause
    the reverb tail of its own speech becomes the next 'utterance'."""
    naps = []
    loop, *_ = make_loop(
        tmp_path, utterances=["Reachy", "when is standup?"], settle_s=0.4, sleep=naps.append
    )
    assert loop.run_once() == "conversation"
    # two in-conversation listens (the question, then the silence that ends it)
    assert naps == [0.4, 0.4]


def test_no_settle_by_default(tmp_path):
    naps = []
    loop, *_ = make_loop(tmp_path, utterances=["Reachy", "hi"], sleep=naps.append)
    loop.run_once()
    assert naps == []


# -- turn-boundary message delivery ----------------------------------------


def _hermes_msg(to_person="p1"):
    return MessageRow(
        message_id="hm1",
        from_person="",
        from_name="Hermes",
        to_person=to_person,
        to_name="Alice",
        text="the SDK shipped v2",
        created_at="2026-07-07T00:00:00+00:00",
        delivered_at="",
    )


class _QueueingBrain(FakeBrain):
    """Queues a message for p1 during its first respond() (i.e. after the greeting)."""

    def __init__(self, store):
        super().__init__()
        self._store = store

    def respond(self, question, identity=None, on_sentence=None):
        if not self.asked:
            self._store.add_message(_hermes_msg())
        return super().respond(question, identity, on_sentence)


def test_message_queued_mid_conversation_is_delivered_next_boundary(tmp_path):
    store = Store(tmp_path / "db")
    loop, speaker, *_ = make_loop(
        tmp_path,
        utterances=[said("Reachy"), said("q1"), said("q2")],
        voices=[ALICE_VOICE, ALICE_VOICE, ALICE_VOICE],
        store=store,
        brain=_QueueingBrain(store),
    )
    loop.run_once()
    needle = "Hermes left you a message: the SDK shipped v2"
    assert sum(needle in s for s in speaker.spoken) == 1
    assert store.pending_messages_for("p1") == []


def test_no_delivery_after_barge_in(tmp_path):
    from reachy_vec.brain.chat import SpeechInterrupted

    store = Store(tmp_path / "db")
    monitor = FakeBargeInMonitor()

    class InterruptingBrain(FakeBrain):
        def respond(self, question, identity=None, on_sentence=None):
            store.add_message(_hermes_msg())
            monitor.trip()
            try:
                on_sentence("half a sentence")
            except SpeechInterrupted:
                return "half a sentence"
            return "full reply"

    loop, speaker, *_ = make_loop(
        tmp_path,
        utterances=[said("Reachy"), said("a question")],
        voices=[ALICE_VOICE, ALICE_VOICE],
        store=store,
        brain=InterruptingBrain(),
        barge_in_factory=lambda: monitor,
    )
    loop.run_once()
    assert not any("left you a message" in s for s in speaker.spoken)
    assert len(store.pending_messages_for("p1")) == 1


def test_no_delivery_for_anonymous_turn(tmp_path):
    store = Store(tmp_path / "db")
    loop, speaker, *_ = make_loop(
        tmp_path,
        utterances=[said("Reachy"), said("q1")],
        voices=[ALICE_VOICE, None],
        store=store,
        brain=_QueueingBrain(store),
    )
    loop.run_once()
    assert not any("left you a message" in s for s in speaker.spoken)
    assert len(store.pending_messages_for("p1")) == 1


# -- acknowledgement fillers ---------------------------------------------------


class _RecordingSpeaker(FakeSpeaker):
    """Records speak() order with the thread each line came from."""

    def __init__(self):
        super().__init__()
        import threading

        self.lock = threading.Lock()

    def speak(self, text):
        import time

        if text in ("Mm.", "Right."):
            time.sleep(0.05)  # a filler takes a moment to play
        with self.lock:
            self.spoken.append(text)


def test_filler_spoken_before_first_sentence(tmp_path):
    loop, _, _, _, brain = make_loop(
        tmp_path, utterances=["Reachy", "when is standup?"], ack_fillers=["Mm."],
    )
    speaker = _RecordingSpeaker()
    loop._speaker = speaker
    assert loop.run_once() == "conversation"
    i = speaker.spoken.index("Mm.")
    assert speaker.spoken[i + 1] == "answer to when is standup?"  # never interleaved


def test_no_filler_on_turn_right_after_greeting(tmp_path):
    loop, speaker, *_ = make_loop(
        tmp_path, utterances=[said("Reachy, what's the time?")], voices=[ALICE_VOICE],
        ack_fillers=["Mm."],
    )
    loop.run_once()
    assert "Mm." not in speaker.spoken     # greeted, then answered: no filler


def test_filler_on_later_turns_after_greeting(tmp_path):
    loop, speaker, *_ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("q1")], voices=[ALICE_VOICE, ALICE_VOICE],
        ack_fillers=["Mm."],
    )
    loop.run_once()
    # "Hi Alice!..." then the listen turn: filler before the answer
    assert speaker.spoken.index("Mm.") < speaker.spoken.index("answer to q1")


def test_no_filler_by_default(tmp_path):
    loop, speaker, *_ = make_loop(tmp_path, utterances=["Reachy", "hi"])
    loop.run_once()
    assert not {"Mm.", "Right.", "Okay."} & set(speaker.spoken)


def test_filler_then_apology_when_brain_fails(tmp_path):
    loop, speaker, *_ = make_loop(
        tmp_path, utterances=["Reachy", "hi"], brain=FakeBrain(fail=True), ack_fillers=["Mm."],
    )
    assert loop.run_once() == "conversation"
    assert speaker.spoken.index("Mm.") < next(
        i for i, s in enumerate(speaker.spoken) if "sorry" in s.lower())


WAKE_VARIANTS = ["reachy", "reachie", "reechy", "reachi", "reache", "reachee", "reechie",
                 "reechi", "rechy", "rachie", "richie", "ritchie", "ritchy",
                 "riachi", "riyachi", "ryachi", "haruchi"]


@pytest.mark.parametrize("word", WAKE_VARIANTS)
def test_default_wake_words_cover_mishearings(word):
    from reachy_vec.config import Settings

    words = Settings.model_fields["wake_words"].default
    assert split_wake(f"hey {word.title()}, what's up", words) == "what's up"


def test_possessive_still_wakes():
    from reachy_vec.config import Settings

    assert split_wake("Reachy's here", Settings.model_fields["wake_words"].default) is not None


@pytest.mark.parametrize("text", ["I'm reaching.", "can you reach it", "we're ready",
                                  "they reached out", "that's rich", "peachy keen"])
def test_everyday_words_do_not_wake(text):
    from reachy_vec.config import Settings

    assert split_wake(text, Settings.model_fields["wake_words"].default) is None


def test_oracle_default_matches_config_default():
    from reachy_vec.brain.oracle import DEFAULT_WAKE_WORDS
    from reachy_vec.config import Settings

    assert list(DEFAULT_WAKE_WORDS) == Settings.model_fields["wake_words"].default


def test_wordless_noise_does_not_end_the_conversation(tmp_path):
    """A noise blip transcribed as "" used to look like silence and end the chat."""
    from reachy_vec.audio.listen import Utterance

    loop, speaker, body, _, brain = make_loop(
        tmp_path,
        utterances=[said("Reachy"), Utterance(text="", audio=AUDIO), said("what's new?")],
    )
    assert loop.run_once() == "conversation"
    assert [q for q, _ in brain.asked] == ["what's new?"]   # still listening after the blip
    assert body.motions.count("goodbye") == 1               # ended once, on real silence


def test_wordless_noise_is_not_a_wake():
    from reachy_vec.audio.listen import Utterance  # noqa: F401  (shape check below)

    assert split_wake("") is None


@pytest.mark.parametrize(("text", "expected"), [
    ("Reachy, stop.", True), ("Hey Reachy wait", True), ("stop", True),
    ("hold on a second", True), ("okay, be quiet", True), ("Richie!", True),
    ("The latest demo in the hub is Metaliverpool.", False),
    ("It tracks food access across the city.", False), ("", False),
])
def test_is_interrupt(text, expected):
    from reachy_vec.brain.oracle import is_interrupt

    assert is_interrupt(text) is expected


@pytest.mark.parametrize("ask", ["remember me", "can you remember my voice?"])
def test_unrecognised_speaker_can_ask_to_be_remembered(tmp_path, ask):
    """Asking to be remembered starts real enrollment instead of the model
    promising "I'll remember you" with nothing stored."""
    loop, speaker, _, store, brain = make_loop(
        tmp_path, brain=FakeBrain(routes={ask: "remember_me"}),
        utterances=[said("Reachy"), said(ask), said("yes"), said("Yang"), said("yes"),
                    said("I work on data"), said("and I like tea")],
        voices=[None, None],
    )
    loop.run_once()
    assert store.match_voice(VOICE_VEC)[1] == "Yang"
    assert ask not in [q for q, _ in brain.asked]       # handled by enrollment, not the LLM


def test_recognised_speaker_asking_to_be_remembered_goes_to_the_brain(tmp_path):
    loop, speaker, _, store, brain = make_loop(
        tmp_path, utterances=[said("Reachy"), said("remember me")],
        voices=[ALICE_VOICE, ALICE_VOICE], brain=FakeBrain(routes={"remember me": "remember_me"}),
    )
    loop.run_once()
    assert any("already know your voice, Alice" in s for s in speaker.spoken)
    assert store.match_voice(VOICE_VEC) is None            # no second enrollment


# -- enrollment: assistant first, names extracted, spelling fallback -----------


@pytest.mark.parametrize(("text", "name"), [
    ("My name is Yang.", "Yang"), ("I'm Yang", "Yang"), ("it's Hannah Smith", "Hannah Smith"),
    ("call me Dave", "Dave"), ("Yang", "Yang"), ("No, it's Yang.", "Yang"),
    ("no", None), ("Nope.", None), ("", None),
])
def test_extract_name(text, name):
    from reachy_vec.brain.oracle import extract_name

    assert extract_name(text) == name


@pytest.mark.parametrize(("text", "name"), [
    ("Y, A, N, G.", "Yang"), ("Y-A-N-G", "Yang"), ("why a n g", "Yang"),
    ("H A N N A H", "Hannah"), ("it's Yang", None),
])
def test_spelled_name(text, name):
    from reachy_vec.brain.oracle import spelled_name

    assert spelled_name(text) == name


def test_strangers_are_just_answered_no_unasked_offer(tmp_path):
    loop, speaker, _, store, brain = make_loop(
        tmp_path, utterances=[said("Reachy, what's the time?")], voices=[UNKNOWN_VOICE])
    assert loop.run_once() == "conversation"
    assert not any("remember your voice" in s for s in speaker.spoken)
    assert [q for q, _ in brain.asked] == ["what's the time?"]


def test_asked_enrollment_extracts_name_and_spells_after_a_mishearing(tmp_path):
    loop, speaker, _, store, _ = make_loop(
        tmp_path, brain=FakeBrain(routes={"remember me": "remember_me"}),
        utterances=[said("Reachy, remember me"),
                    said("My name is Jan."), said("No."),          # misheard
                    said("Y, A, N, G."), said("Yes."),             # spelled, confirmed
                    said("I work on data"), said("and I like tea")],
        voices=[UNKNOWN_VOICE],
    )
    loop.run_once()
    assert store.match_voice(VOICE_VEC)[1] == "Yang"
    assert any("spell" in s.lower() for s in speaker.spoken)
    assert not any("My Name Is" in s for s in speaker.spoken)


def test_correction_in_the_no_answer_is_used(tmp_path):
    loop, _, _, store, _ = make_loop(
        tmp_path, brain=FakeBrain(routes={"remember me": "remember_me"}),
        utterances=[said("Reachy, remember me"), said("I'm Young"), said("No, it's Yang."),
                    said("yes"), said("one sentence"), said("two sentences")],
        voices=[UNKNOWN_VOICE],
    )
    loop.run_once()
    assert store.match_voice(VOICE_VEC)[1] == "Yang"


def test_failed_enrollment_keeps_the_conversation_going(tmp_path):
    loop, speaker, _, _, brain = make_loop(
        tmp_path, brain=FakeBrain(routes={"remember me": "remember_me"}),
        utterances=[said("Reachy, remember me"), said("mumble"), said("no"),
                    said("not sure"), said("no"), said("forget it"), said("no"),
                    said("what's the weather?")],
        voices=[UNKNOWN_VOICE],
    )
    loop.run_once()
    assert "what's the weather?" in [q for q, _ in brain.asked]   # carried on
    assert any("remember me" in s for s in speaker.spoken)          # told how to retry


def test_auto_offer_is_paused_after_a_decline(tmp_path):
    now = {"t": 1000.0}
    loop, speaker, _, _, _ = make_loop(
        tmp_path, utterances=[said("Reachy"), said("no thanks")], voices=[UNKNOWN_VOICE],
        auto_offer=True, clock=lambda: now["t"])
    loop.run_once()
    loop._transcriber = FakeTranscriber([said("Reachy, hi")])
    loop._speaker_id = FakeSpeakerIdentifier([UNKNOWN_VOICE], embedding=VOICE_VEC)
    now["t"] = 1300.0                                    # 5 min later: still paused
    offers_before = sum("remember your voice" in s for s in speaker.spoken)
    loop.run_once()
    assert sum("remember your voice" in s for s in speaker.spoken) == offers_before


# -- message delivery: condensed and interruptible --------------------------------


def _queue(store, text, frm="Hermes"):
    store.add_message(MessageRow(message_id=f"m{abs(hash(text)) % 10**6}", from_person="x",
                                 from_name=frm, to_person="p1", to_name="Alice", text=text,
                                 created_at="2026-10-22T00:00:00+00:00", delivered_at=""))


def test_long_messages_are_condensed_for_speech(tmp_path):
    store = Store(tmp_path / "db")
    _queue(store, "**VEC Team:**\n- **Yang** leads data\n- **David** builds demos\n" * 5)

    class CondensingBrain(FakeBrain):
        def condense_for_speech(self, text):
            return "The team is Yang on data and David on demos."

    loop, speaker, *_ = make_loop(tmp_path, utterances=[said("Reachy")], voices=[ALICE_VOICE],
                                  store=store, brain=CondensingBrain())
    loop.run_once()
    spoken = " ".join(speaker.spoken)
    assert "Hermes left you a message: The team is Yang on data and David on demos." in spoken
    assert "**" not in spoken
    assert store.pending_messages_for("p1") == []


def test_message_delivery_can_be_interrupted(tmp_path):
    store = Store(tmp_path / "db")
    _queue(store, "First news item. Second news item. Third news item.")
    monitor = FakeBargeInMonitor()

    class TrippingSpeaker(FakeSpeaker):
        def speak(self, text):
            super().speak(text)
            if "First news item" in text:
                monitor.trip()          # "Reachy, stop" while the message plays

    loop, _, *_ = make_loop(tmp_path, utterances=[said("Reachy")], voices=[ALICE_VOICE],
                            store=store, barge_in_factory=lambda: monitor)
    speaker = TrippingSpeaker()
    loop._speaker = speaker
    loop.run_once()
    assert not any("Third news item" in s for s in speaker.spoken)
    assert store.pending_messages_for("p1") == []           # heard enough: delivered


def test_emojis_become_gestures_not_words(tmp_path):
    class EmojiBrain(FakeBrain):
        def respond(self, question, identity=None, on_sentence=None):
            on_sentence("That's brilliant news! 🎉😄")
            on_sentence("Anything else? 🤔")
            return "x"

    loop, speaker, body, *_ = make_loop(tmp_path, utterances=["Reachy", "guess what"],
                                        brain=EmojiBrain(), spawn=lambda f: f())
    loop.run_once()
    assert "That's brilliant news!" in speaker.spoken and "Anything else?" in speaker.spoken
    assert not any(ch in "".join(speaker.spoken) for ch in "🎉😄🤔")
    for emotion in ("excited", "happy", "thinking"):
        assert emotion in body.motions


def test_delivered_message_full_text_goes_to_the_brain_context(tmp_path):
    store = Store(tmp_path / "db")
    _queue(store, "Long Hermes report about the whole team. " * 10)

    class ContextBrain(FakeBrain):
        def __init__(self):
            super().__init__()
            self.context = []

        def condense_for_speech(self, text):
            return "Short version."

        def add_context(self, full_text, spoken):
            self.context.append((full_text, spoken))

    brain = ContextBrain()
    loop, *_ = make_loop(tmp_path, utterances=[said("Reachy")], voices=[ALICE_VOICE],
                         store=store, brain=brain)
    loop.run_once()
    assert brain.context and "Long Hermes report" in brain.context[0][0]
    assert "Short version." in brain.context[0][1]

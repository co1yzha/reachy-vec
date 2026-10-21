"""The Oracle loop: wake-word-triggered voice Q&A, voice-led enrollment.

Synchronous state machine; all dependencies injected for testability.
The mic is always open: transcriber.listen_once(timeout) blocks and paces
the loop. An utterance containing the wake word ("Reachy") starts a
conversation; who is talking is decided by voice alone (never guessed).
"""

import logging
import random
import re
import time
import uuid
from datetime import UTC, datetime

from reachy_vec.audio.pipeline import SpeechPipeline
from reachy_vec.brain.chat import SpeechInterrupted
from reachy_vec.perception.fusion import from_voice
from reachy_vec.perception.voice import BORDERLINE_MARGIN, Observation
from reachy_vec.store.schemas import VoiceRow

logger = logging.getLogger(__name__)

APOLOGY = "Sorry, my brain isn't responding right now."
OFFER = "Hi! I don't think we've met. Would you like me to remember your voice? Say yes or no."
DEFAULT_WAKE_WORDS = (
    "reachy", "reachie", "reechy", "reachi", "reache", "reachee", "reechie", "reechi",
    "rechy", "rachie", "richie", "ritchie", "ritchy", "riachi", "riyachi", "ryachi", "haruchi",
)

# Short lines spoken on waking from idle sleep, so a wake is unmissable.
WAKE_LINES = (
    "Mm — I'm up!",
    "Oh! Hello again.",
    "Yawn... I'm awake now.",
    "Back with you!",
)


def _is_yes(utterance) -> bool:
    return utterance is not None and "yes" in utterance.text.lower()


def _clean_name(text: str) -> str:
    return text.strip().strip(".!?,").title()


def split_wake(text: str, wake_words=DEFAULT_WAKE_WORDS) -> str | None:
    """Text after the wake word ("" if nothing follows), or None if absent.

    "Hey Reachy, what's the weather?" -> "what's the weather?"
    """
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in wake_words) + r")\b[\s,.!?:;-]*"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return text[match.end():].strip()


INTERRUPT_WORDS = re.compile(
    r"\b(stop|wait|hold on|be quiet|quiet|shush|pause|enough|shut up)\b", re.IGNORECASE
)


def is_interrupt(text: str, wake_words=DEFAULT_WAKE_WORDS) -> bool:
    """Speech over a reply counts as an interruption only if it says Reachy's
    name or a stop word - Reachy's own reply (heard by the mic) never does."""
    if not text or not text.strip():
        return False
    return split_wake(text, wake_words) is not None or bool(INTERRUPT_WORDS.search(text))


class OracleLoop:
    def __init__(
        self,
        *,
        transcriber,
        speaker,
        body,
        brain,
        store,
        speaker_id=None,
        clock=time.time,
        wake_words=DEFAULT_WAKE_WORDS,
        greet_cooldown_s: float = 7200.0,
        silence_timeout_s: float = 30.0,
        wake_listen_s: float = 5.0,
        idle_sleep_s: float = 300.0,
        start_asleep: bool = False,
        voice_threshold: float = 0.30,
        voice_passive_cap: int = 10,
        enroll_phrases: int = 2,
        barge_in_factory=None,
        ack_fillers=None,
        settle_s: float = 0.0,
        sleep=time.sleep,
    ):
        self._transcriber = transcriber
        self._speaker = speaker
        self._body = body
        self._brain = brain
        self._store = store
        self._speaker_id = speaker_id  # None = voice ID disabled (everyone anonymous)
        self._wake_words = tuple(wake_words)
        self._voice_threshold = voice_threshold
        self._voice_passive_cap = voice_passive_cap
        self._enroll_phrases = enroll_phrases
        self._barge_in_factory = barge_in_factory  # None = barge-in disabled
        # short lines ("Mm.") played while the LLM reads the prompt; empty = off
        self._ack_fillers = list(ack_fillers or [])
        # Pause before opening the mic so the reverb tail of the robot's own
        # speech isn't captured as the next utterance.
        self._settle_s = settle_s
        self._sleep = sleep or time.sleep
        self._clock = clock
        self._greet_cooldown_s = greet_cooldown_s
        self._silence_timeout_s = silence_timeout_s
        self._wake_listen_s = wake_listen_s
        self._idle_sleep_s = idle_sleep_s
        self._last_wake_at = clock()
        # Start asleep so a robot physically left in sleep pose (previous
        # session, power cycle) is woken by the first "Reachy" it hears.
        self._asleep = start_asleep

    # -- public ---------------------------------------------------------

    def run_once(self) -> str:
        """Listen for the wake word; converse or enroll; return the event.

        Returns "idle" when nothing (or nothing addressed to Reachy) was heard;
        production wraps run_once in run_forever, which retries.
        """
        heard = self._transcriber.listen_once(self._wake_listen_s)
        rest = split_wake(heard.text, self._wake_words) if heard is not None else None
        if rest is None:
            self._maybe_sleep()
            return "idle"
        logger.info("wake word heard: %r", heard.text)
        woke_speaking = self._note_wake()
        voice_obs = self._identify_voice(heard.audio)
        if voice_obs is not None and voice_obs.person_id is None:
            event, person = self._offer_enroll(first_audio=heard.audio)
            if person is not None:
                # "Ask me anything" - keep listening
                self._converse(person, just_spoke=True)
            return event
        self._converse(voice_obs, first_question=rest, just_spoke=woke_speaking)
        return "conversation"

    def run_forever(self) -> None:
        self._body.perform("idle")
        while True:
            event = self.run_once()
            if event != "idle":
                logger.info("interaction ended: %s", event)

    # -- states ----------------------------------------------------------

    def _converse(self, voice_obs, first_question: str = "", just_spoke: bool = False) -> None:
        """voice_obs: known speaker, or None (couldn't tell from the wake
        utterance - e.g. a bare "Reachy" is too short to embed).
        just_spoke: Reachy already said something since the wake word, so a
        question asked with the wake word needs no filler."""
        owner = voice_obs if voice_obs is not None and voice_obs.person_id else None
        self._brain.begin_conversation(owner.person_id if owner else None,
                                       owner.name if owner else None)
        if owner is not None:
            just_spoke = self._greet(owner) or just_spoke
        elif not first_question:
            self._speaker.speak("Yes? What can I help you with?")
            self._body.perform("acknowledge")
        pending = first_question  # question asked in the same breath as the wake word
        offered = False
        while True:
            ack = True
            if pending:
                text, audio, pending = pending, None, ""
                turn_voice = voice_obs  # same utterance as the wake word
                ack = not just_spoke  # a greeting already covered the wait
            else:
                self._body.perform("listen")
                utterance = self._listen(self._silence_timeout_s)
                if utterance is None:
                    self._body.perform("goodbye")
                    self._brain.end_conversation()  # distill memories of the visit
                    return
                text, audio = utterance.text, utterance.audio
                turn_voice = self._identify_voice(audio)
                if owner is None and turn_voice is not None and turn_voice.person_id:
                    owner = turn_voice  # recognized mid-conversation
                    self._deliver_messages(owner.person_id)
            if self._reply(text, turn_voice, ack=ack):
                self._maybe_bank_voice(turn_voice, audio)
                continue  # interrupted: user is already talking, skip the nod
            self._body.perform("nod")
            self._maybe_bank_voice(turn_voice, audio)
            if turn_voice is not None and turn_voice.person_id:
                # messages queued mid-conversation reach whoever just spoke
                self._deliver_messages(turn_voice.person_id)
            if owner is None and not offered and turn_voice is not None \
                    and turn_voice.person_id is None:
                # a stranger woke us with a bare "Reachy"; now we know
                offered = True
                _event, owner = self._offer_enroll(first_audio=audio)

    def _greet(self, person) -> bool:
        """Greet (outside the cooldown) and deliver messages; True if spoken."""
        spoke = False
        if self._cooldown_expired(person.person_id):
            self._speaker.speak(f"Hi {person.name}! What can I help you with?")
            self._body.perform("greet")
            self._record_greeting(person.person_id)
            spoke = True
        else:
            self._body.perform("acknowledge")
        return self._deliver_messages(person.person_id) or spoke

    def _reply(self, text: str, voice_obs, ack: bool = True) -> bool:
        """Answer one turn; returns True if the reply was barged in on.

        Speech goes through a SpeechPipeline: with ack a short filler plays first
        (pre-generated, instant) while the LLM reads the prompt, and each next
        sentence is synthesized while the current one plays. The turn ends once
        everything queued has been spoken.
        """
        speech = SpeechPipeline(self._speaker)
        monitor = self._barge_in_factory() if self._barge_in_factory else None
        if monitor is not None:
            monitor.start(on_fire=speech.stop)
        if ack and self._ack_fillers:
            speech.say(random.choice(self._ack_fillers))

        if monitor is not None:

            def on_sentence(sentence, _m=monitor):
                if _m.fired:
                    raise SpeechInterrupted()
                speech.say(sentence)
        else:
            on_sentence = speech.say
        try:
            # sentences are queued as they stream in; respond returns when the
            # reply is complete, then wait() lets the queue finish playing
            self._brain.respond(text, identity=from_voice(voice_obs), on_sentence=on_sentence)
        except Exception:
            logger.exception("brain.respond failed")
            speech.say(APOLOGY)
        finally:
            speech.wait()
            if monitor is not None:
                monitor.stop()
        return monitor is not None and monitor.fired

    def _listen(self, timeout_s: float):
        """Next utterance with words, or None after timeout_s of no speech.

        A sound transcribed as "" (noise, a cough) is skipped and listening
        resumes for the rest of the window - it used to read as silence and
        end the conversation right after a reply.
        """
        if self._settle_s > 0:
            self._sleep(self._settle_s)
        deadline = time.monotonic() + timeout_s
        while True:
            utterance = self._transcriber.listen_once(timeout_s)
            if utterance is None or utterance.text.strip():
                return utterance
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                return None

    def _identify_voice(self, audio):
        if self._speaker_id is None:
            return None
        try:
            return self._speaker_id.identify(audio)
        except Exception:
            logger.exception("speaker ID failed - treating as can't tell")
            return None

    def _maybe_bank_voice(self, voice_obs, audio) -> None:
        """Passively grow the voice profile of a *clearly* matched speaker.

        With no second modality to confirm the match, only matches a full
        margin above threshold are banked, so a near-miss can't drift a
        profile toward someone else's voice.
        """
        if self._speaker_id is None or audio is None:
            return
        if voice_obs is None or voice_obs.person_id is None:
            return
        if voice_obs.score < self._voice_threshold + BORDERLINE_MARGIN:
            return
        try:
            vector = self._speaker_id.embed(audio)
            if vector is None:
                return
            self._store.add_voice_rows([self._voice_row(voice_obs.person_id, voice_obs.name,
                                                        vector, source="passive")])
            self._store.prune_passive_voices(voice_obs.person_id, keep=self._voice_passive_cap)
        except Exception:
            logger.exception("passive voice backfill failed - skipping")

    def _voice_row(self, person_id: str, name: str, vector, *, source: str) -> VoiceRow:
        return VoiceRow(
            voice_id=f"{person_id}:{uuid.uuid4().hex[:8]}",
            person_id=person_id,
            name=name,
            vector=vector,
            created_at=datetime.now(UTC).isoformat(),
            source=source,
        )

    def _offer_enroll(self, first_audio=None) -> tuple[str, Observation | None]:
        """Offer voice enrollment; returns (event, the new person or None)."""
        self._speaker.speak(OFFER)
        if not _is_yes(self._listen(10)):
            self._speaker.speak("No problem! Just say Reachy if you need me.")
            return "enroll-declined", None
        for _attempt in range(2):
            self._speaker.speak("Great! What's your name?")
            heard = self._listen(10)
            if heard is None:
                continue
            name = _clean_name(heard.text)
            self._speaker.speak(f"Nice to meet you, {name} - did I get that right?")
            if _is_yes(self._listen(10)):
                person_id = f"person-{uuid.uuid4().hex[:8]}"
                if not self._capture_voice(person_id, name, extra_audio=[first_audio]):
                    self._speaker.speak("I couldn't catch your voice - let's try another time.")
                    return "enroll-declined", None
                self._record_greeting(person_id)
                self._speaker.speak(f"All set, {name}! Ask me anything.")
                self._body.perform("greet")
                return "enrolled", Observation(person_id=person_id, name=name, score=1.0)
        self._speaker.speak("Let's try again another time.")
        return "enroll-declined", None

    def _capture_voice(self, person_id: str, name: str, extra_audio=()) -> bool:
        """Record a few spoken sentences as the enrolled voice profile.

        extra_audio: earlier audio of the same person (the wake utterance),
        used too if long enough to embed. Returns False if nothing embedded.
        """
        if self._speaker_id is None:
            return False
        vectors = [v for v in (self._speaker_id.embed(a) for a in extra_audio if a is not None)
                   if v is not None]
        self._speaker.speak("Now tell me a little about yourself, so I learn your voice.")
        for i in range(self._enroll_phrases):
            if i > 0:
                self._speaker.speak("Lovely - and one more sentence, anything you like.")
            utterance = self._listen(10)
            vector = self._speaker_id.embed(utterance.audio) if utterance is not None else None
            if vector is not None:
                vectors.append(vector)
        if not vectors:
            return False
        self._store.add_voice_rows(
            [self._voice_row(person_id, name, v, source="enrolled") for v in vectors]
        )
        return True

    def _deliver_messages(self, person_id: str) -> bool:
        """Speak pending messages; True if any were delivered."""
        delivered = False
        for msg in self._store.pending_messages_for(person_id):
            self._speaker.speak(
                f"By the way, {msg.from_name} left you a message: {msg.text}"
            )
            self._store.mark_delivered(msg.message_id)
            logger.info("delivered message %s to %s", msg.message_id, msg.to_name)
            delivered = True
        return delivered

    # -- wake/sleep --------------------------------------------------------

    def _maybe_sleep(self) -> None:
        if not self._asleep and self._clock() - self._last_wake_at >= self._idle_sleep_s:
            logger.info("not called for %.0fs - going to sleep", self._idle_sleep_s)
            self._body.perform("sleep")
            self._asleep = True

    def _note_wake(self) -> bool:
        """Reset the idle timer; wake with a spoken flourish. True if spoken."""
        self._last_wake_at = self._clock()
        if self._asleep:
            logger.info("wake word while asleep - waking up")
            self._body.perform("wake")
            self._body.perform("wakeup")
            self._speaker.speak(random.choice(WAKE_LINES))
            self._asleep = False
            return True
        return False

    # -- helpers ----------------------------------------------------------

    def _cooldown_expired(self, person_id: str) -> bool:
        last = self._store.get_last_greeted(person_id)
        if last is None:
            return True
        elapsed = self._clock() - datetime.fromisoformat(last).timestamp()
        return elapsed >= self._greet_cooldown_s

    def _record_greeting(self, person_id: str) -> None:
        now_iso = datetime.fromtimestamp(self._clock(), tz=UTC).isoformat()
        self._store.set_last_greeted(person_id, now_iso)

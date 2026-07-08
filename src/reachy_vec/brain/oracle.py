"""The Oracle loop: face-triggered greeting, voice Q&A, robot-led enrollment.

Synchronous state machine; all dependencies injected for testability.
sight() is polled; transcriber.listen_once(timeout) blocks and paces the loop.
"""

import logging
import time
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

APOLOGY = "Sorry, my brain isn't responding right now."
OFFER = "Hi! I don't think we've met. Would you like me to remember you? Say yes or no."


def _is_yes(utterance) -> bool:
    return utterance is not None and "yes" in utterance.text.lower()


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
        brain,
        enroll_capture,
        store,
        clock=time.time,
        greet_cooldown_s: float = 7200.0,
        silence_timeout_s: float = 30.0,
        unknown_stable_polls: int = 3,
        idle_sleep_s: float = 300.0,
    ):
        self._sight = sight
        self._transcriber = transcriber
        self._speaker = speaker
        self._body = body
        self._brain = brain
        self._enroll_capture = enroll_capture
        self._store = store
        self._clock = clock
        self._greet_cooldown_s = greet_cooldown_s
        self._silence_timeout_s = silence_timeout_s
        self._unknown_stable_polls = unknown_stable_polls
        self._idle_sleep_s = idle_sleep_s
        self._last_face_at = clock()
        self._asleep = False

    # -- public ---------------------------------------------------------

    def run_once(self) -> str:
        """One interaction: wait for a face, converse or enroll, return event.

        With scripted sights (tests) a None observation ends the wait as
        "no-face"; production wraps run_once in run_forever, which retries.
        """
        unknown_streak = 0
        while True:
            obs = self._sight()
            if obs is None:
                self._maybe_sleep()
                return "no-face"
            self._note_presence()
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
            if event != "no-face":
                logger.info("interaction ended: %s", event)
            time.sleep(0.5)

    # -- states ----------------------------------------------------------

    def _converse(self, person_id: str, name: str) -> None:
        self._brain.begin_conversation(person_id, name)
        if self._cooldown_expired(person_id):
            self._speaker.speak(f"Hi {name}! What can I help you with?")
            self._body.perform("greet")
            self._record_greeting(person_id)
        else:
            self._body.perform("acknowledge")
        self._deliver_messages(person_id)
        while True:
            self._body.perform("listen")
            utterance = self._transcriber.listen_once(self._silence_timeout_s)
            if utterance is None:
                self._body.perform("goodbye")
                self._brain.end_conversation()  # distill memories of the visit
                return
            try:
                # sentences are spoken as they stream in; respond blocks
                # until the reply is complete
                self._brain.respond(
                    utterance.text, speaker_name=name, on_sentence=self._speaker.speak
                )
                self._body.perform("nod")
            except Exception:
                logger.exception("brain.respond failed")
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
            name = _clean_name(heard.text)
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

    def _deliver_messages(self, person_id: str) -> None:
        for msg in self._store.pending_messages_for(person_id):
            self._speaker.speak(
                f"By the way, {msg.from_name} left you a message: {msg.text}"
            )
            self._store.mark_delivered(msg.message_id)
            logger.info("delivered message %s to %s", msg.message_id, msg.to_name)

    # -- wake/sleep --------------------------------------------------------

    def _maybe_sleep(self) -> None:
        if not self._asleep and self._clock() - self._last_face_at >= self._idle_sleep_s:
            logger.info("no faces for %.0fs - going to sleep", self._idle_sleep_s)
            self._body.perform("sleep")
            self._asleep = True

    def _note_presence(self) -> None:
        self._last_face_at = self._clock()
        if self._asleep:
            logger.info("face detected - waking up")
            self._body.perform("wake")
            self._asleep = False

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

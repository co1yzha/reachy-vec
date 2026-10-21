import time

from reachy_vec.body.sway import SpeakingSway, SwayingSpeaker
from tests.test_body import FlakyBody


class SlowBody:
    """Records motions; each perform takes a beat, like real keyframes."""

    def __init__(self):
        self.motions: list[str] = []

    def perform(self, motion: str) -> None:
        self.motions.append(motion)
        time.sleep(0.01)


def _wait_for_motions(body, deadline_s=2.0):
    deadline = time.time() + deadline_s
    while not body.motions and time.time() < deadline:
        time.sleep(0.005)


def test_speaking_sway_performs_only_between_start_and_stop():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.start()
    _wait_for_motions(body)
    sway.stop()
    count = len(body.motions)
    assert count >= 1
    assert set(body.motions) == {"sway"}
    time.sleep(0.05)
    assert len(body.motions) == count  # nothing after stop (thread joined)


def test_speaking_sway_start_and_stop_are_idempotent():
    body = SlowBody()
    sway = SpeakingSway(body)
    sway.stop()  # stop before start: no-op
    sway.start()
    sway.start()  # second start: no second thread
    _wait_for_motions(body)
    sway.stop()
    sway.stop()  # double stop: no error


def test_speaking_sway_ends_quietly_when_body_raises():
    sway = SpeakingSway(FlakyBody(fail=99))  # every perform raises
    sway.start()
    time.sleep(0.05)
    sway.stop()  # thread already dead from the exception; join is clean


class RecordingSway:
    def __init__(self):
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


class ScriptedSpeaker:
    def __init__(self, boom=False):
        self.spoken: list[str] = []
        self.stopped = 0
        self._boom = boom

    def speak(self, text: str) -> None:
        if self._boom:
            raise RuntimeError("tts hiccup")
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped += 1


def test_swaying_speaker_brackets_each_sentence():
    sway, inner = RecordingSway(), ScriptedSpeaker()
    speaker = SwayingSpeaker(inner, sway)
    speaker.speak("hello")
    speaker.speak("world")
    assert inner.spoken == ["hello", "world"]
    assert sway.events == ["start", "stop", "start", "stop"]


def test_swaying_speaker_stops_sway_on_speak_error():
    sway = RecordingSway()
    speaker = SwayingSpeaker(ScriptedSpeaker(boom=True), sway)
    try:
        speaker.speak("hello")
    except RuntimeError:
        pass
    assert sway.events == ["start", "stop"]  # finally clause ran


def test_swaying_speaker_stop_halts_sway_too():
    sway, inner = RecordingSway(), ScriptedSpeaker()
    speaker = SwayingSpeaker(inner, sway)
    speaker.stop()  # barge-in path
    assert inner.stopped == 1
    assert sway.events == ["stop"]


def test_swaying_speaker_exposes_clips_only_when_inner_does():
    from reachy_vec.body.sway import SwayingSpeaker
    from tests.conftest import FakeSpeaker

    class Sway:
        def __init__(self):
            self.log = []

        def start(self):
            self.log.append("start")

        def stop(self):
            self.log.append("stop")

    plain = SwayingSpeaker(FakeSpeaker(), Sway())
    assert not hasattr(plain, "synthesize")

    class ClipInner(FakeSpeaker):
        def synthesize(self, text):
            return f"clip<{text}>"

        def play_clip(self, clip):
            self.spoken.append(clip)

    sway = Sway()
    s = SwayingSpeaker(ClipInner(), sway)
    s.play_clip(s.synthesize("Hi."))
    assert s._speaker.spoken == ["clip<Hi.>"] and sway.log == ["start", "stop"]

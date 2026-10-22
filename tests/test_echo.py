from reachy_vec.audio.echo import EchoGuard, is_echo
from tests.conftest import FakeSpeaker


def test_echo_of_the_reply_is_recognised():
    said = ["I can't switch myself off, Yang, but I'll stay quiet if you'd like some peace."]
    assert is_echo("But I'll stay quiet if you'd like to.", said)


def test_a_real_interruption_is_not_an_echo():
    said = ["I'll stay quiet if you'd like some peace."]
    assert not is_echo("Reachy, stop", said)
    assert not is_echo("wait, what about the weather tomorrow", said)
    assert not is_echo("quiet", said)        # too short to tell - never filtered


def test_guard_remembers_what_the_speaker_said_recently():
    now = [100.0]
    guard = EchoGuard(FakeSpeaker(), clock=lambda: now[0], window_s=20.0)
    guard.speak("Sure, I'll stay quiet if you like.")
    assert guard.is_echo("I'll stay quiet if you like")
    now[0] += 30
    assert not guard.is_echo("I'll stay quiet if you like")   # aged out


def test_guard_delegates_to_the_wrapped_speaker():
    inner = FakeSpeaker()
    guard = EchoGuard(inner)
    guard.speak("hello there")
    guard.stop()
    guard.precache(["hi"])
    assert inner.spoken == ["hello there"] and inner.stopped == 1
    assert inner.precached == [["hi"]]
    assert not hasattr(guard, "synthesize")   # keeps SpeechPipeline's mode choice


def test_guard_records_two_stage_synthesis():
    class TwoStage(FakeSpeaker):
        def synthesize(self, text):
            return text

        def play_clip(self, clip):
            self.spoken.append(clip)

    guard = EchoGuard(TwoStage())
    assert guard.synthesize("let me open that page for you") == "let me open that page for you"
    assert guard.is_echo("open that page for you")

import threading
import time

from reachy_vec.audio.pipeline import SpeechPipeline
from tests.conftest import FakeSpeaker


class ClipSpeaker(FakeSpeaker):
    """Two-stage speaker: synthesize(text) -> clip, play_clip(clip). Records the
    order of events; `play_s` makes playback take a while."""

    def __init__(self, play_s=0.0, fail_on=None):
        super().__init__()
        self.events: list[str] = []
        self.play_s = play_s
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def synthesize(self, text):
        if text == self.fail_on:
            raise RuntimeError("tts broke")
        with self.lock:
            self.events.append(f"synth:{text}")
        return f"clip<{text}>"

    def play_clip(self, clip):
        with self.lock:
            self.events.append(f"play-start:{clip}")
        time.sleep(self.play_s)
        with self.lock:
            self.events.append(f"play-end:{clip}")
            self.spoken.append(clip)


def test_plays_in_order_and_wait_blocks_until_done():
    sp = ClipSpeaker()
    p = SpeechPipeline(sp)
    for s in ["One.", "Two.", "Three."]:
        p.say(s)
    p.wait()
    assert sp.spoken == ["clip<One.>", "clip<Two.>", "clip<Three.>"]


def test_next_sentence_is_synthesized_while_current_plays():
    sp = ClipSpeaker(play_s=0.2)
    p = SpeechPipeline(sp)
    p.say("One.")
    p.say("Two.")
    p.wait()
    ev = sp.events
    assert ev.index("synth:Two.") < ev.index("play-end:clip<One.>")   # overlap


def test_say_does_not_block_the_caller():
    sp = ClipSpeaker(play_s=0.3)
    p = SpeechPipeline(sp)
    t = time.monotonic()
    p.say("One.")
    p.say("Two.")
    assert time.monotonic() - t < 0.1          # LLM keeps streaming meanwhile
    p.wait()


def test_stop_cancels_pending_and_stops_current():
    sp = ClipSpeaker(play_s=0.2)
    p = SpeechPipeline(sp)
    for s in ["One.", "Two.", "Three."]:
        p.say(s)
    time.sleep(0.05)
    p.stop()
    p.wait()
    assert sp.stopped == 1
    assert "clip<Three.>" not in sp.spoken
    p.say("Four.")                              # ignored after stop
    p.wait()
    assert "clip<Four.>" not in sp.spoken


def test_failed_synthesis_skips_only_that_sentence():
    sp = ClipSpeaker(fail_on="Two.")
    p = SpeechPipeline(sp)
    for s in ["One.", "Two.", "Three."]:
        p.say(s)
    p.wait()
    assert sp.spoken == ["clip<One.>", "clip<Three.>"]


def test_single_stage_speaker_still_ordered():
    sp = FakeSpeaker()                          # speak() only, like `say`
    p = SpeechPipeline(sp)
    for s in ["One.", "Two."]:
        p.say(s)
    p.wait()
    assert sp.spoken == ["One.", "Two."]


def test_blank_text_is_skipped():
    sp = FakeSpeaker()
    p = SpeechPipeline(sp)
    p.say("   ")
    p.wait()
    assert sp.spoken == []

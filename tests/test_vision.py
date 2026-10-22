import base64

import numpy as np

from reachy_vec.perception.vision import (
    encode_frame_jpeg,
    make_look_around_fn,
    make_look_fn,
    make_selfie_fn,
)
from tests.conftest import FakeBody, FakeCamera, FakeLLMClient


def _frame(h=10, w=20):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_encode_frame_jpeg_returns_data_url():
    url = encode_frame_jpeg(_frame(), max_px=1024)
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_encode_frame_jpeg_downscales_long_edge():
    import cv2

    url = encode_frame_jpeg(_frame(h=100, w=400), max_px=50)
    raw = base64.b64decode(url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 50


def test_look_fn_performs_gesture_then_answers():
    body = FakeBody()
    look = make_look_fn(
        FakeCamera([_frame()]), FakeLLMClient(reply="a tidy desk"),
        model="gpt-5-mini", max_px=1024, body=body,
    )
    assert look("what do you see?") == "a tidy desk"
    assert body.motions == ["look"]


def test_look_fn_sends_image_and_question():
    client = FakeLLMClient(reply="ok")
    look = make_look_fn(FakeCamera([_frame()]), client, model="gpt-5-mini", max_px=1024)
    look("count the people")
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    assert any(p["type"] == "text" and "count the people" in p["text"] for p in content)


def test_look_fn_empty_question_uses_default_prompt():
    client = FakeLLMClient(reply="ok")
    look = make_look_fn(FakeCamera([_frame()]), client, model="gpt-5-mini", max_px=1024)
    look("")
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert any(p["type"] == "text" and "Describe what you see" in p["text"] for p in content)


def test_look_fn_no_frame_is_friendly():
    look = make_look_fn(FakeCamera([None]), FakeLLMClient(reply="x"),
                        model="gpt-5-mini", max_px=1024)
    assert "can't see" in look("hi").lower()


def test_look_fn_gesture_failure_does_not_block():
    class BoomBody:
        def perform(self, motion):
            raise RuntimeError("wifi drop")

    look = make_look_fn(FakeCamera([_frame()]), FakeLLMClient(reply="a wall"),
                        model="gpt-5-mini", max_px=1024, body=BoomBody())
    assert look("what's there?") == "a wall"


def test_look_fn_vision_error_is_friendly():
    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("api down")

    look = make_look_fn(FakeCamera([_frame()]), BoomClient(),
                        model="gpt-5-mini", max_px=1024)
    assert "trouble" in look("hi").lower()


def _image_parts(client):
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    return [p for p in content if p["type"] == "image_url"]


def test_look_around_turns_both_ways_then_recenters_and_captures_twice():
    body = FakeBody()
    client = FakeLLMClient(reply="a desk to my left and a window to my right")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), client,
        model="gpt-5-mini", max_px=1024, body=body, sleep=lambda s: None,
    )
    assert look_around() == "a desk to my left and a window to my right"
    assert body.motions == ["turn_left", "turn_right", "turn_front"]
    assert len(client.chat.completions.calls) == 1  # exactly one vision call


def test_look_around_sends_both_images_labelled_left_and_right():
    client = FakeLLMClient(reply="ok")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), client,
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    look_around()
    content = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert len(_image_parts(client)) == 2
    text = " ".join(p["text"].lower() for p in content if p["type"] == "text")
    assert "left" in text and "right" in text


def test_look_around_uses_injected_sleep_between_turns():
    slept = []
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), FakeLLMClient(reply="ok"),
        model="gpt-5-mini", max_px=1024, body=FakeBody(),
        settle_s=0.5, sleep=slept.append,
    )
    look_around()
    assert slept == [0.5, 0.5]  # settled after each turn, no real time.sleep


def test_look_around_no_body_captures_once_forward():
    client = FakeLLMClient(reply="a tidy room")
    look_around = make_look_around_fn(
        FakeCamera([_frame()]), client,
        model="gpt-5-mini", max_px=1024, body=None, sleep=lambda s: None,
    )
    assert look_around() == "a tidy room"
    assert len(_image_parts(client)) == 1  # single forward image


def test_look_around_drops_a_missing_direction():
    client = FakeLLMClient(reply="just the left side")
    look_around = make_look_around_fn(
        FakeCamera([_frame(), None]), client,      # right frame missing
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    assert look_around() == "just the left side"
    assert len(_image_parts(client)) == 1


def test_look_around_no_frames_at_all_is_friendly():
    look_around = make_look_around_fn(
        FakeCamera([None, None]), FakeLLMClient(reply="x"),
        model="gpt-5-mini", max_px=1024, body=FakeBody(), sleep=lambda s: None,
    )
    assert "can't see" in look_around().lower()


def test_look_around_recenters_even_when_vision_call_fails():
    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("api down")

    body = FakeBody()
    look_around = make_look_around_fn(
        FakeCamera([_frame(), _frame()]), BoomClient(),
        model="gpt-5-mini", max_px=1024, body=body, sleep=lambda s: None,
    )
    assert "trouble" in look_around().lower()
    assert body.motions[-1] == "turn_front"  # recentered despite the failure


def test_selfie_saves_file_and_opens_it(tmp_path):
    opened = []
    selfie = make_selfie_fn(
        FakeCamera([_frame()]), tmp_path / "photos",
        body=FakeBody(), speak=lambda t: None, opener=opened.append,
    )
    result = selfie()
    saved = list((tmp_path / "photos").glob("*.jpg"))
    assert len(saved) == 1
    assert opened == [str(saved[0])]
    assert "photo" in result.lower()
    assert saved[0].name in result and "saved" in result   # so "where is it?" is answerable


def test_selfie_ceremony_order(tmp_path):
    events = []

    class RecCamera:
        def read(self):
            events.append("capture")
            return _frame()

    class RecBody:
        def perform(self, motion):
            events.append(f"motion:{motion}")

    selfie = make_selfie_fn(
        RecCamera(), tmp_path / "photos",
        body=RecBody(), speak=lambda t: events.append("speak"),
        opener=lambda p: events.append("open"),
    )
    selfie()
    assert events == ["speak", "motion:pose", "capture", "open"]


def test_selfie_no_frame_writes_nothing(tmp_path):
    selfie = make_selfie_fn(FakeCamera([None]), tmp_path / "photos",
                            body=FakeBody(), speak=lambda t: None, opener=lambda p: None)
    assert "couldn't take" in selfie().lower()
    assert not (tmp_path / "photos").exists() or not list((tmp_path / "photos").glob("*.jpg"))


def test_selfie_two_rapid_calls_do_not_overwrite(tmp_path):
    selfie = make_selfie_fn(
        FakeCamera([_frame(), _frame()]), tmp_path / "photos",
        body=FakeBody(), speak=lambda t: None, opener=lambda p: None,
    )
    selfie()
    selfie()
    assert len(list((tmp_path / "photos").glob("*.jpg"))) == 2


def test_selfie_best_effort_failures_still_save(tmp_path):
    opened = []

    def boom(*a):
        raise RuntimeError("hardware")

    selfie = make_selfie_fn(
        FakeCamera([_frame()]), tmp_path / "photos",
        body=type("B", (), {"perform": boom})(), speak=boom, opener=opened.append,
    )
    result = selfie()
    assert list((tmp_path / "photos").glob("*.jpg"))  # saved despite speak+pose failing
    assert "photo" in result.lower()


THINK_OFF = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_look_fn_passes_llm_kwargs_and_strips_reasoning():
    client = FakeLLMClient(reply="<think>hmm</think>A tidy desk.")
    look = make_look_fn(FakeCamera([_frame()]), client, model="m", max_px=1024,
                        llm_kwargs=THINK_OFF)
    assert look("what do you see?") == "A tidy desk."
    assert client.chat.completions.last_kwargs["extra_body"] == THINK_OFF["extra_body"]


def test_look_around_passes_llm_kwargs():
    client = FakeLLMClient(reply="A room.")
    look_around = make_look_around_fn(FakeCamera([_frame(), _frame(), _frame()]), client,
                                      model="m", max_px=1024, sleep=lambda s: None,
                                      llm_kwargs=THINK_OFF)
    look_around()
    assert client.chat.completions.last_kwargs["extra_body"] == THINK_OFF["extra_body"]

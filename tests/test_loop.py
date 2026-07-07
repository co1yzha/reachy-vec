from reachy_vec.brain.loop import chat_loop

from tests.conftest import FakeBrain


def run_loop(inputs: list[str]) -> tuple[list[str], FakeBrain]:
    inputs_iter = iter(inputs)
    printed: list[str] = []
    brain = FakeBrain()

    def input_fn(prompt=""):
        try:
            return next(inputs_iter)
        except StopIteration:
            raise EOFError

    chat_loop(brain=brain, input_fn=input_fn, print_fn=printed.append)
    return printed, brain


def test_loop_answers_then_exits():
    printed, brain = run_loop(["when does it run?", "exit"])
    assert any("answer to when does it run?" in line for line in printed)
    assert brain.asked == [("when does it run?", None)]


def test_loop_exits_on_eof():
    printed, _ = run_loop(["one question"])  # EOF after -> must return, not raise
    assert any("answer to one question" in line for line in printed)


def test_loop_skips_blank_lines():
    _, brain = run_loop(["", "  ", "quit"])
    assert brain.asked == []

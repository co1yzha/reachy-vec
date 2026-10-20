import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "bench_llm.py"


def _load():
    spec = importlib.util.spec_from_file_location("bench_llm", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_median_odd_even():
    bench = _load()
    assert bench.median([3.0, 1.0, 2.0]) == 2.0
    assert bench.median([4.0, 1.0, 2.0, 3.0]) == 2.5


def test_tool_cases_shape():
    bench = _load()
    assert len(bench.TOOL_CASES) == 10
    assert sum(1 for _, t in bench.TOOL_CASES if t is None) == 4


def test_score_tool_calls_nine_of_ten():
    bench = _load()
    scripted = {u: t for u, t in bench.TOOL_CASES}
    wrong = bench.TOOL_CASES[0][0]
    scripted[wrong] = "get_time"
    correct, total = bench.score_tool_calls(bench.TOOL_CASES, scripted.get)
    assert (correct, total) == (9, 10)


def test_score_tool_calls_skips_unregistered_tool():
    bench = _load()
    cases = [c for c in bench.TOOL_CASES if c[1] != "ask_hermes"]
    scripted = {u: t for u, t in bench.TOOL_CASES}
    correct, total = bench.score_tool_calls(
        bench.TOOL_CASES, scripted.get, available_tools={t for _, t in cases if t}
    )
    assert total == len(cases) == 9
    assert correct == 9


def test_leaked_reasoning_tags_only():
    bench = _load()
    assert bench.leaked_reasoning(["I think that's right."]) is False
    assert bench.leaked_reasoning(["<think>plan"]) is True
    assert bench.leaked_reasoning(["done</think> Hi."]) is True

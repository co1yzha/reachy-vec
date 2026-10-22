"""Score the turn router on varied phrasings against the live local LLM.

    uv run --no-sync python scripts/route_eval.py

Prints accuracy per route, every miss, and the router's latency (median / max).
"""

import statistics
import time

CASES = [
    # hermes: explicit, current info, tasks
    ("ask Hermes to find today's top AI news", "hermes"),
    ("what's happening in AI this week?", "hermes"),
    ("any news about Nvidia?", "hermes"),
    ("who won the Liverpool match yesterday?", "hermes"),
    ("what's the price of bitcoin right now?", "hermes"),
    ("can you look up the newest Python release?", "hermes"),
    ("remind me tomorrow at ten about the review", "hermes"),
    ("research the Reachy Mini SDK and summarise it", "hermes"),
    ("Hermes, add 'fix the mic' to the task board", "hermes"),
    # latest_demos
    ("what is the latest demo in the hub?", "latest_demos"),
    ("which demos were added most recently?", "latest_demos"),
    ("show me the three newest demos", "latest_demos"),
    # fix_body
    ("can you get your head up?", "fix_body"),
    ("bring your noggin up", "fix_body"),
    ("why aren't you moving?", "fix_body"),
    ("are your motors on?", "fix_body"),
    ("your head is drooping", "fix_body"),
    # remember_me
    ("remember me please", "remember_me"),
    ("can you learn my voice?", "remember_me"),
    ("my name is Yang, please remember it", "remember_me"),
    # chat
    ("how are you today?", "chat"),
    ("any demos about food?", "chat"),
    ("tell me about the food mapping demo", "chat"),
    ("what's a p-value?", "chat"),
    ("what's the weather like?", "chat"),
    ("what time is it?", "chat"),
    ("tell Bob the meeting moved to three", "chat"),
    ("open the food mapping demo", "chat"),
    ("what do you see?", "chat"),
    ("take a picture of me", "chat"),
    ("head over to the demos about health", "chat"),
    ("I was reaching for my coffee", "chat"),
]


def main() -> None:
    from reachy_vec.brain.llm import make_llm_client, uses_local_llm
    from reachy_vec.brain.router import Router
    from reachy_vec.config import settings

    kwargs = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
              if uses_local_llm() else {})
    router = Router(make_llm_client(), settings.llm_model, llm_kwargs=kwargs)
    router.route("warm up")
    times, misses, per = [], [], {}
    for text, want in CASES:
        t0 = time.perf_counter()
        got = router.route(text)
        times.append(time.perf_counter() - t0)
        ok, total = per.get(want, (0, 0))
        per[want] = (ok + (got.name == want), total + 1)
        if got.name != want:
            misses.append(f"  {text!r}: wanted {want}, got {got.name}")
        elif got.task:
            print(f"  ok  {text!r} -> task {got.task!r}")
    correct = sum(ok for ok, _ in per.values())
    print(f"\naccuracy {correct}/{len(CASES)}")
    for name, (ok, total) in per.items():
        print(f"  {name:13} {ok}/{total}")
    print("misses:" if misses else "misses: none")
    print("\n".join(misses))
    print(f"latency median {statistics.median(times):.2f}s, max {max(times):.2f}s")


if __name__ == "__main__":
    main()

---
title: Chat tools
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [tools, rag, memory, messaging]
sources: [raw/docs/pipelines.md]
---

All in `brain/chat.py`. Tool calls cost one extra LLM round-trip (sentence
audio suppressed while pending). Camera tools (`look`/`selfie`) are injected
only by `run`.

| Tool | Effect | Guardrails |
|---|---|---|
| `open_url` | opens a demo in the lab browser | http(s) only |
| `save_note` | stores a memory about the current person | requires a recognized visit; near-duplicates (cosine ≥ 0.97) skipped |
| `send_message` | queues a relay in the `messages` table | recipient must be enrolled (case-insensitive) |
| `get_weather` | live conditions via Open-Meteo (no key) | location from `WEATHER_LAT/LON`; 5 s timeout |
| `get_time` | current local date and time | none (pure `datetime`) |
| `web_search` | live web answer via Tavily (`answer`) | off unless `WEB_SEARCH=true` + `TAVILY_API_KEY`; 5 s timeout; 432/433 → "used up my allowance", 429 → rate-limited message |

Feeds [[memory-pipeline]] (`save_note`), [[messenger-pipeline]]
(`send_message`), and the [[rag-pipeline]] turn.


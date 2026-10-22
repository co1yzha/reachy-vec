# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete
> When this file exceeds 500 entries, rotate: rename to log-YYYY.md, start fresh.

## [2026-10-19] create | Wiki initialized
- Domain: reachy-vec / Team Familiar (embodied team assistant on Reachy Mini)
- Location: repo-internal `wiki/` (point `WIKI_PATH` here)
- Structure: SCHEMA.md, index.md, log.md, raw/, entities/, concepts/, comparisons/, queries/
- Tag taxonomy: 20 tags (robot, sim, media-source, face-id, speaker-id, stt, tts, llm, rag, lancedb, state-machine, memory, messaging, barge-in, tools, protocol, config, testing, roadmap, gotcha)

## [2026-10-19] ingest | repo docs (README, architecture, pipelines)
- Sources captured to `raw/docs/`: readme.md, architecture.md, pipelines.md (sha256-stamped)
- Created 22 seed pages: 10 entities, 10 concepts, 2 comparisons
- Entities: reachy-mini, mac-brain, lancedb-store, gpt-5-mini, bge-small-en, faster-whisper, insightface, ecapa-tdnn, qwen3-tts, silero-vad
- Concepts: oracle-loop, identity-rule, rag-pipeline, memory-pipeline, messenger-pipeline, barge-in, media-source, protocol-pattern, roadmap-phases, tools
- Comparisons: stt-backends, tts-backends

## [2026-10-19] lint | initial health check
- Findings: 1 orphan page (mac-brain, no inbound links); 8 pages with <2 outbound links
- Fixed in same session: added cross-references to reachy-mini, media-source, protocol-pattern, bge-small-en, silero-vad, insightface, ecapa-tdnn, qwen3-tts, barge-in, messenger-pipeline
- Re-lint: 22 pages, 0 broken links, 0 orphans, 0 pages below link minimum, index complete

## [2026-10-19] update | Layered configuration (profile .env)
- Change: config.py now loads (profile .env, repo .env) — repo .env overrides, profile .env = deployment home for shared secrets/defaults
- Files: src/reachy_vec/config.py, tests/test_config.py (test_env_file_layering), docs/configuration.md, .env.example
- Verified: ruff clean, 292 tests pass, live smoke test (env_file shows both layers)
- Wiki: new page [[configuration]] (concepts/)
- Note: working tree was mid-refactor by a concurrent session (face recognition removed, wake words added) — pre-existing wiki pages on face ID may be stale until that lands

---
title: Layered configuration
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [config, media-source]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

All robot/brain settings live in `src/reachy_vec/config.py`
(`pydantic-settings`), set as environment variables with the `REACHY_VEC_`
prefix. Unknown keys are ignored. Full reference: `docs/configuration.md`.

## Value precedence (highest first)

1. **Real environment variables** exported by the shell/CI
2. **`.env` at the repo root** — per-machine overrides (gitignored)
3. **The Reachy profile's `.env`** — `~/.hermes/profiles/reachy/.env` —
   the deployment's shared home for secrets and defaults (loaded only if it
   exists)

Since 2026-10-19 the brain loads both files: shared secrets
(`OPENAI_API_KEY`, `MONGODB_URI`) and deployment defaults
(`REACHY_VEC_ROBOT_HOST`, ...) live **once** in the profile `.env` — the same
file the Reachy agent runs from — while the repo `.env` keeps
machine-specific overrides (TTS backend/voice sample, weather coords, idle
sleep). Layering is pinned by
`tests/test_config.py::test_env_file_layering`.

## Robot-specific knobs

`ROBOT_HOST` / `ROBOT_PORT` (network-mode daemon address; unset = local
daemon/simulator), `ROBOT_RECONNECT`, `BODY_RECONNECT_ATTEMPTS`, and
`MEDIA_SOURCE` (see [[media-source]]). `DATA_DIR` (default `data/`) holds the
[[lancedb-store]] tables, enrollment frames, and the privacy-relevant
`reachy.log`.

Note: configuration is read on the [[mac-brain]] — the robot itself holds no
config; it is reached over WiFi at `ROBOT_HOST`.

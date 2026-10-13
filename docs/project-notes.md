# Project notes

Durable, non-obvious project knowledge that isn't derivable from the code, git
history, or `docs/architecture.md` — deferred decisions, operational recovery
steps, and queued work. For architecture and per-step model/config detail see
`docs/architecture.md` and `docs/pipelines.md`; for design specs and plans see
`docs/superpowers/`.

## Queued next task: local small-LLM integration

**Status:** deferred by decision (2026-07-09), queued as the next task
(2026-07-23).

The current version stays **cloud-only** (OpenAI API). A local/offline LLM is a
capability upgrade, not a correctness fix, and a failover layer can't be
meaningfully tested without killing the network mid-conversation — so it pairs
naturally with the other deferred Phase 4d work (autostart, on-device
resilience).

When picked up:
- **Manual-swap path first (~15 lines):** `brain/chat.py:ChatBrain` already
  speaks the OpenAI `/v1/chat/completions` protocol (streaming + tools), so a
  local runtime (Ollama / LM Studio / llama.cpp / vLLM) drops in by pointing
  `OpenAI(base_url=..., api_key=...)` at localhost. Add `LLM_BASE_URL` /
  `LLM_API_KEY` config knobs.
- **Automatic failover (later):** a wrapper that catches API errors and retries
  against a second (local) client.
- **Open decisions to brainstorm:** which runtime, which small model,
  manual-swap-only vs. automatic failover.
- **Caveat:** small local models emit less-reliable tool calls
  (`open_url` / `save_note` / `send_message` / `get_weather` / `get_time` /
  camera tools) — the brain is tool-heavy, so validate this early.

## Operational: recovering a wedged robot daemon

Diagnosed 2026-07-15 on the wireless Reachy Mini (`reachy-mini.local`,
`192.168.0.50`). Symptom: the daemon is up, `/api/motors/status` says
`enabled`, every SDK/REST motion command returns success — and the robot never
moves. Even the daemon's own `POST /api/move/play/wake_up` doesn't move it.
"Went to sleep, won't wake up" presents the same way.

**Recovery:** `curl -X POST http://reachy-mini.local:8000/api/daemon/restart`
(or the Restart button on the dashboard at that address). The boot sequence
re-inits the motor bus and wakes the robot. If the API restart fails,
power-cycle.

**Diagnostic ladder (fast → slow):**
1. `ping reachy-mini.local` + `curl .../api/daemon/status` — daemon reachable?
2. `data/reachy.log` — body warnings? (`make_robot` silently degrades to
   `NullBody` on connect failure; `ReconnectingBody` logs per-motion failures.)
3. Measured move test: GET `/api/state/present_head_pose`, POST
   `/api/move/play/wake_up`, GET pose again — objective, no eyes needed.
4. Robot logs stream over `ws://reachy-mini.local:8000/logs/ws/daemon`;
   dashboard page `/logs`.

**Gotchas:**
- Constructing `ReachyMini(media_backend="no_media")` **releases the daemon's
  media**, killing any live robot camera/mic stream held by another client —
  never open a second no-media connection while a media client is active (this
  is why `ReconnectingBody` is seeded with the initial body).
- On daemon 1.8.0 the WebRTC signalling port 8443 only opens after
  `POST /api/media/acquire`, and the SDK media client needs it during
  construction — `make_robot` RESTs the acquire first.
- Robot WebRTC frames are read-only numpy arrays (cv2 can't draw on them in
  place; first frame takes ~2 s).
- `backend_status.ready:false` / `last_alive:null` in `/api/daemon/status` is
  **not** diagnostic on daemon 1.8.0 — it stays false even when motion works
  (the 1.8.4 SDK source suggests otherwise; version-mismatch trap).
- USB-C to the Mac is power-only on the wireless model — all control is over
  WiFi.
- SDK 1.8.4 vs daemon 1.8.0 mismatch is the standing suspect for the wedge;
  update via the dashboard System Updates to rule it out.

## Status: hardware-gap source of truth

Trust `docs/architecture.md` → "Known gaps" over the top-level `CLAUDE.md` for
hardware-gap status; the `CLAUDE.md` prose has lagged reality at times. As of
mid-2026: Phase 4a/4b/4c have shipped (on-robot media via `--source robot`,
`ROBOT_HOST` network mode + `ReconnectingBody`, barge-in). Open per
architecture.md: on-robot `say` output still renders Mac-side, far-field mic
gain untuned, robot-media hot-recovery, and Phase 4d (autostart + offline
degradation) not started.

Camera tools `look()` / `selfie()` shipped 2026-07-14 (`perception/vision.py`,
`run`-only injected closures). `look_around()` — turn the base left/right,
capture at each, describe surroundings — added 2026-07-23 on branch
`camera-poke-emotions` (spec/plan under `docs/superpowers/`); still needs the
on-robot smoke test per `docs/testing.md`. Monocular 3D reconstruction was
considered and deferred (single RGB cam; would need a head-sweep + multi-view
model as a standalone `reconstruct` command).

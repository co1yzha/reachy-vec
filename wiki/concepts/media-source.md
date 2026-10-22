---
title: Media source (auto / robot / mac)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [media-source, robot]
sources: [raw/docs/architecture.md, raw/docs/pipelines.md]
---

`MEDIA_SOURCE` / `reachy-vec run --source` selects whose camera/mic/speaker the
loop uses:

| Value | Source |
|---|---|
| `mac` | The [[mac-brain]]'s own camera/mic/speaker (default when no robot media) |
| `robot` | The [[reachy-mini]]'s camera/mic/speaker over the SDK (`RobotCamera`, `RobotAudioSource`, `RobotAudioSink`) — robot audio out needs the [[qwen3-tts]] backend |
| `auto` | Robot when the daemon offers media, else Mac |

- **Phase 4a** shipped on-robot media; `auto` falls back to the Mac when the
  daemon offers no media.
- Still open: on-robot `say` output (renders Mac-side) and far-field mic gain.
- A `no_media` SDK connection **releases** the daemon's media — which is why
  `ReconnectingBody` is seeded with the initial body.


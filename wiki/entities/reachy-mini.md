---
title: Reachy Mini (robot)
created: 2026-10-19
updated: 2026-10-19
type: entity
tags: [robot, media-source]
sources: [raw/docs/readme.md, raw/docs/architecture.md]
---

The Reachy Mini is the robot "body" — the [[mac-brain]] does the thinking. It is **wireless**: the USB-C port is
power-only, and the whole robot is controlled over WiFi via the SDK.

- **Onboard:** Raspberry Pi 5 + the reachy-mini daemon (dashboard at `:8000`).
- **Network:** joins a hotspot at first boot, then appears on the LAN as
  `reachy-mini.local` (mDNS). Enterprise WiFi (eduroam etc.) usually rejects
  embedded devices — use a phone/Mac hotspot or personal router.
- **Role today:** a thin body for motion; since **Phase 4a** its camera/mic/
  speaker can also drive the loop on-robot (`--source robot`). See
  [[media-source]] and [[roadmap-phases]].
- **Gotchas:** a wedged onboard daemon makes every command "succeed" while the
  head stays still — restart via `POST /api/daemon/restart` or power-cycle.
  One media client at a time; a `no_media` SDK connection releases the daemon's
  media and kills another client's live stream. ^[raw/docs/readme.md]


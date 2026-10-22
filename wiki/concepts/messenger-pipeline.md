---
title: Messenger pipeline (relays)
created: 2026-10-19
updated: 2026-10-19
type: concept
tags: [messaging, state-machine]
sources: [raw/docs/pipelines.md]
---

Phase 3. Lets the robot relay a message to an absent teammate.

1. Sender (recognized) says "tell Bob …" → the model calls `send_message` (see [[tools]]) → a
   row in the `messages` table with `delivered_at=""`. The recipient must be
   enrolled (case-insensitive name match).
2. When the recipient is next recognized, right after the greeting the
   [[oracle-loop]] speaks each pending message **oldest-first** ("By the way,
   Alice left you a message: …") and stamps `delivered_at`.


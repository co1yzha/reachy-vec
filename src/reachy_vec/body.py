"""Robot I/O over WiFi: audio in/out, camera frames, motion primitives.

Motion primitives: greet (head-turn + antenna wiggle), idle, listen pose,
nod, droop. Falls back to the mujoco simulator when no robot_host is set.
"""

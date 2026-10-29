"""Speaker volume on request ("speak up", "too loud").

Two backends behind one small shape (get() -> 0-100, set(level)): the robot
daemon's REST volume (when replies play on the robot) and macOS output volume
via osascript (when they play on the Mac). VolumeControl turns spoken intents
into steps and reports the result in words the LLM can relay.
"""

import json
import logging
import subprocess
import urllib.request
from typing import Protocol

logger = logging.getLogger(__name__)

STEP = 15
FLOOR = 15   # never mute: a silent robot can't confirm it heard "louder"
QUIET = 35
CEILING = 100

_UP = ("up", "louder", "higher", "increase", "raise")
_DOWN = ("down", "quieter", "lower", "softer", "decrease", "reduce")
_MAX = ("max", "maximum", "loudest", "full")
_QUIET = ("quiet", "min", "minimum", "quietest", "low")


class Volume(Protocol):
    def get(self) -> int: ...
    def set(self, level: int) -> None: ...


def _http_json(url: str, *, method: str = "GET", body=None, timeout: float = 3.0):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read() or b"{}")


class RobotVolume:
    """The daemon's /api/volume endpoints (setting one also plays a short
    test sound on the robot)."""

    def __init__(self, base_url: str, http=_http_json):
        self._base = base_url.rstrip("/")
        self._http = http

    def get(self) -> int:
        _, data = self._http(f"{self._base}/volume/current")
        return int(data["volume"])

    def set(self, level: int) -> None:
        self._http(f"{self._base}/volume/set", method="POST", body={"volume": level})


def _osascript(script: str) -> str:
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                          check=True, timeout=5).stdout.strip()


class MacVolume:
    def __init__(self, run=_osascript):
        self._run = run

    def get(self) -> int:
        return int(self._run("output volume of (get volume settings)"))

    def set(self, level: int) -> None:
        self._run(f"set volume output volume {level}")


class VolumeControl:
    def __init__(self, volume: Volume):
        self._volume = volume

    def change(self, change: str) -> str:
        """change: up/down/max/quiet (or synonyms) or a level 0-100 -> a
        short report for the LLM to say."""
        word = (change or "").strip().lower().rstrip("%")
        try:
            current = self._volume.get()
            if word.isdigit():
                target = int(word)
            elif word in _UP:
                if current >= CEILING:
                    return "Already at the loudest I go."
                target = current + STEP
            elif word in _DOWN:
                if current <= FLOOR:
                    return "Already at the quietest I go without going silent."
                target = current - STEP
            elif word in _MAX:
                target = CEILING
            elif word in _QUIET:
                target = QUIET
            else:
                return "Ask whether they want it louder or quieter."
            target = max(FLOOR, min(CEILING, target))
            self._volume.set(target)
        except Exception as exc:
            logger.warning("volume change %r failed: %s", change, exc)
            return "I couldn't change my volume - the speaker isn't answering."
        logger.info("volume %d -> %d (%s)", current, target, change)
        return f"Volume changed from {current} to {target} out of 100."

# Phase 4b — ROBOT_HOST + body resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the declared-but-unused `ROBOT_HOST` real (connect to a remote Reachy Mini over the network) and let robot **motions** survive a transient daemon/WiFi drop by reconnecting, degrading to a body-less mode with one spoken notice only when it truly gives up.

**Architecture:** Two self-contained pieces in `body/robot.py`. (1) `make_robot` reads `settings.robot_host`/`robot_port` and connects in `connection_mode="network"` when a host is set, else the SDK default (`"auto"`, local daemon). (2) A `ReconnectingBody` wraps a body behind the same `Body.perform()` protocol; it catches the `ConnectionError`/`TimeoutError` the SDK raises on a lost connection, rebuilds a fresh body-only connection on the next motion, and after N consecutive failures marks itself dead (no-op motions) and announces once. Robot **media** (camera/mic/speaker) keeps its 4a soft-degrade — `get_frame()`/`get_audio_sample()` return `None`/silence on a hiccup, so the robot simply goes quiet until the stream recovers. Live media→Mac hot-swap is explicitly out of scope (its own later phase).

**Tech Stack:** Python 3.12, `reachy-mini` SDK (`ReachyMini(host, port, connection_mode)`, `mini.client.is_connected()`), pytest, ruff, typer.

## Global Constraints

- Python **3.12+**; no new third-party dependency.
- **Every heavy dependency behind a Protocol with a test fake.** `ReconnectingBody` and `make_robot` are tested with the existing `FakeMini` / injected `connect` (no SDK, no hardware).
- Config knobs go in `src/reachy_vec/config.py` with the `REACHY_VEC_` prefix.
- Reconnection logic lives in the body adapter only — **no new branches in `brain/oracle.py`**; the loop keeps calling `body.perform(...)` unchanged.
- **Out of scope (deferred):** live media→Mac hot-swap. When robot media dies, the camera/mic just yield `None`/silence (already true from 4a); do NOT add live device re-wiring here.
- After changes: `uv run ruff check src tests` and `uv run pytest -q` must both pass.

---

### Task 1: Wire `ROBOT_HOST` / `ROBOT_PORT` into `make_robot`

**Files:**
- Modify: `src/reachy_vec/config.py` (add `robot_port`)
- Modify: `src/reachy_vec/body/robot.py` (`make_robot` reads host/port; add `from reachy_vec.config import settings`)
- Test: `tests/test_body.py`

**Interfaces:**
- Consumes: `settings.robot_host` (existing, `str | None`), `settings.robot_port` (new).
- Produces: `make_robot(with_media=False, connect=None)` now passes `host=`, `port=`, `connection_mode="network"` to the SDK constructor when `robot_host` is set; when unset, passes neither (SDK default `connection_mode="auto"`, local daemon). Signature unchanged; `connect` stays injectable and now receives those kwargs.

- [ ] **Step 1: Add the `robot_port` config knob**

In `src/reachy_vec/config.py`, under the `# Robot` line (near `robot_host`):

```python
    robot_host: str | None = None
    robot_port: int = 8000  # daemon port; used with robot_host in network mode
```

- [ ] **Step 2: Write the failing tests**

```python
# add to tests/test_body.py (FakeMini already defined in this file)
from reachy_vec.config import settings as _settings


def test_make_robot_uses_network_mode_when_robot_host_set(monkeypatch):
    monkeypatch.setattr(_settings, "robot_host", "reachy.local")
    monkeypatch.setattr(_settings, "robot_port", 8123)
    captured = {}

    def connect(**kw):
        captured.update(kw)
        return FakeMini()

    make_robot(with_media=False, connect=connect)
    assert captured["connection_mode"] == "network"
    assert captured["host"] == "reachy.local"
    assert captured["port"] == 8123


def test_make_robot_local_when_no_robot_host(monkeypatch):
    monkeypatch.setattr(_settings, "robot_host", None)
    captured = {}

    def connect(**kw):
        captured.update(kw)
        return FakeMini()

    make_robot(with_media=False, connect=connect)
    assert "connection_mode" not in captured  # SDK default 'auto'
    assert "host" not in captured
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -k make_robot -v`
Expected: FAIL — `KeyError: 'connection_mode'` (host/port not passed yet)

- [ ] **Step 4: Implement in `body/robot.py`**

Add the import at the top of the file (below `import logging`):

```python
from reachy_vec.config import settings
```

In `make_robot`, replace the single `mini = connect(media_backend=backend)` line with a kwargs build:

```python
        backend = "default" if with_media else "no_media"
        kwargs = {"media_backend": backend}
        if settings.robot_host:
            kwargs.update(
                host=settings.robot_host,
                port=settings.robot_port,
                connection_mode="network",
            )
        mini = connect(**kwargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -v`
Expected: PASS (existing body tests + 2 new)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/config.py src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: wire ROBOT_HOST/ROBOT_PORT — connect to a remote robot in network mode"
```

---

### Task 2: `ReconnectingBody`

**Files:**
- Modify: `src/reachy_vec/body/robot.py`
- Test: `tests/test_body.py`

**Interfaces:**
- Consumes: a `connect_body: Callable[[], Body]` thunk that returns a fresh `Body` or raises `ConnectionError`/`TimeoutError`; an optional `announce: Callable[[str], None]`.
- Produces: `class ReconnectingBody:` `__init__(self, connect_body, max_attempts=3, announce=None)`, `perform(self, motion)` — satisfies the `Body` protocol. Lazily connects on first motion; on a lost-connection error it drops the inner body and reconnects on the next motion; after `max_attempts` consecutive failures it goes permanently dead for the session (motions become no-ops) and calls `announce(...)` exactly once. A successful motion resets the failure counter, so transient blips recover silently.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_body.py


class FlakyBody:
    """RobotBody stand-in: raises ConnectionError for the first `fail` performs."""

    def __init__(self, fail=0):
        self._fail = fail
        self.motions: list[str] = []

    def perform(self, motion):
        if self._fail > 0:
            self._fail -= 1
            raise ConnectionError("Lost connection with the server.")
        self.motions.append(motion)


def test_reconnecting_body_recovers_silently_from_a_blip():
    from reachy_vec.body.robot import ReconnectingBody

    # first inner body fails once; next connect yields a healthy body
    bodies = iter([FlakyBody(fail=1), FlakyBody(fail=0)])
    said: list[str] = []
    body = ReconnectingBody(
        connect_body=lambda: next(bodies), max_attempts=3, announce=said.append
    )
    body.perform("greet")  # inner #1 raises -> dropped, no announce
    body.perform("nod")    # reconnects to inner #2 -> records "nod"
    assert said == []      # transient blip stays silent


def test_reconnecting_body_gives_up_and_announces_once():
    from reachy_vec.body.robot import ReconnectingBody

    def always_fails():
        return FlakyBody(fail=99)

    said: list[str] = []
    body = ReconnectingBody(
        connect_body=always_fails, max_attempts=3, announce=said.append
    )
    for _ in range(5):
        body.perform("nod")  # never raises out
    assert len(said) == 1              # announced exactly once
    assert "body" in said[0].lower()


def test_reconnecting_body_is_noop_after_death():
    from reachy_vec.body.robot import ReconnectingBody

    healthy = FlakyBody(fail=0)
    bodies = iter([FlakyBody(fail=99)] * 3 + [healthy])
    body = ReconnectingBody(connect_body=lambda: next(bodies), max_attempts=3)
    for _ in range(10):
        body.perform("nod")
    assert healthy.motions == []  # dead body never reaches a later healthy connection
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -k reconnecting -v`
Expected: FAIL — `ImportError: cannot import name 'ReconnectingBody'`

- [ ] **Step 3: Implement `ReconnectingBody` in `body/robot.py`**

Add after `RobotBody` (and add `from collections.abc import Callable` to the imports):

```python
class ReconnectingBody:
    """Wraps a Body; rebuilds its connection after a transient drop, and
    degrades to a silent no-op (announcing once) after max_attempts failures.

    Media is NOT re-acquired here (camera/mic soft-degrade independently);
    this only keeps motions alive across a daemon/WiFi blip.
    """

    def __init__(
        self,
        connect_body: "Callable[[], Body]",
        max_attempts: int = 3,
        announce: "Callable[[str], None] | None" = None,
    ):
        self._connect_body = connect_body
        self._max_attempts = max_attempts
        self._announce = announce or (lambda _msg: None)
        self._inner: Body | None = None
        self._failures = 0
        self._dead = False

    def perform(self, motion: str) -> None:
        if self._dead:
            return
        try:
            if self._inner is None:
                self._inner = self._connect_body()
            self._inner.perform(motion)
            self._failures = 0
        except (ConnectionError, TimeoutError) as exc:
            self._inner = None
            self._failures += 1
            logger.warning(
                "Body command %r failed (%s); reconnect attempt %d/%d.",
                motion, exc, self._failures, self._max_attempts,
            )
            if self._failures >= self._max_attempts:
                self._dead = True
                self._announce(
                    "I've lost connection to my body, but I can still hear you."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -v`
Expected: PASS (all body tests + 3 new)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: ReconnectingBody — motions survive a daemon/WiFi blip"
```

---

### Task 3: Wire reconnection into `run.py` + config switches

**Files:**
- Modify: `src/reachy_vec/config.py` (add `robot_reconnect`, `body_reconnect_attempts`)
- Modify: `src/reachy_vec/cli/run.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ReconnectingBody` (Task 2), `make_robot` (Task 1), the already-built `speaker` (for `announce`).
- Produces:
  - `config.py`: `robot_reconnect: bool = True`, `body_reconnect_attempts: int = 3`.
  - `cli/run.py`: `wrap_reconnect(body, connect_body, announce) -> Body` — pure helper: returns a `ReconnectingBody` when `settings.robot_reconnect` and `body` is a `RobotBody`, else returns `body` unchanged (a `NullBody` needs no reconnection). Wired in `run()` after the speaker exists, so `announce=speaker.speak`; the loop is constructed with the wrapped body.

- [ ] **Step 1: Add config knobs**

In `src/reachy_vec/config.py`, under the robot settings:

```python
    robot_reconnect: bool = True  # rebuild the body connection after a transient drop
    body_reconnect_attempts: int = 3  # consecutive motion failures before giving up
```

- [ ] **Step 2: Write the failing tests**

```python
# add to tests/test_cli.py
from reachy_vec.body.robot import NullBody, ReconnectingBody, RobotBody
from reachy_vec.cli.run import wrap_reconnect


class _StubMini:
    def goto_target(self, **kw):
        pass

    def goto_sleep(self):
        pass

    def wake_up(self):
        pass


def test_wrap_reconnect_wraps_a_robot_body(monkeypatch):
    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", True)
    body = RobotBody(_StubMini())
    wrapped = wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None)
    assert isinstance(wrapped, ReconnectingBody)


def test_wrap_reconnect_leaves_nullbody_alone(monkeypatch):
    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", True)
    body = NullBody()
    assert wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None) is body


def test_wrap_reconnect_disabled_returns_body_unchanged(monkeypatch):
    monkeypatch.setattr("reachy_vec.cli.run.settings.robot_reconnect", False)
    body = RobotBody(_StubMini())
    assert wrap_reconnect(body, connect_body=lambda: body, announce=lambda m: None) is body
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k wrap_reconnect -v`
Expected: FAIL — `ImportError: cannot import name 'wrap_reconnect'`

- [ ] **Step 4: Implement `wrap_reconnect` + wire it in `run.py`**

Add the helper near `resolve_media_source` in `cli/run.py`:

```python
def wrap_reconnect(body, connect_body, announce):
    """Wrap a RobotBody so motions survive a daemon/WiFi drop; pass others through."""
    from reachy_vec.body.robot import ReconnectingBody, RobotBody

    if settings.robot_reconnect and isinstance(body, RobotBody):
        return ReconnectingBody(
            connect_body=connect_body,
            max_attempts=settings.body_reconnect_attempts,
            announce=announce,
        )
    return body
```

In `run()`, after `speaker = make_speaker(...)` is created and before building the `OracleLoop`, wrap the body (reconnection rebuilds a **body-only** connection — no media re-acquire, matching the deferred-media scope):

```python
    body = wrap_reconnect(
        body,
        connect_body=lambda: make_robot(with_media=False)[0],
        announce=speaker.speak,
    )
```

The `OracleLoop(..., body=body, ...)` line already uses `body`, so no further change there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing + 3 new)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/config.py src/reachy_vec/cli/run.py tests/test_cli.py
git commit -m "feat: run wraps the robot body in ReconnectingBody (config-gated)"
```

---

### Task 4: Config + docs

**Files:**
- Modify: `docs/configuration.md`, `docs/architecture.md`, `docs/testing.md`, `.env.example`

**Interfaces:** none (documentation only). Fold into one commit.

- [ ] **Step 1: `docs/configuration.md`** — in the **Environment** table, replace the "reserved, not yet consumed" `ROBOT_HOST` note with the now-wired behavior, and add rows:

```markdown
| `ROBOT_HOST` | unset | remote Reachy Mini address; when set, `make_robot` connects in `connection_mode="network"` to `ROBOT_HOST:ROBOT_PORT`. Unset = local daemon (SDK `auto`) |
| `ROBOT_PORT` | `8000` | daemon port, used with `ROBOT_HOST` |
| `ROBOT_RECONNECT` | `true` | rebuild the body connection after a transient motion drop; `false` = a single drop degrades straight to body-less |
| `BODY_RECONNECT_ATTEMPTS` | `3` | consecutive motion failures before the robot gives up on its body for the session (speaks one notice, keeps listening) |
```

- [ ] **Step 2: `docs/architecture.md`** — in "Known gaps", mark gap #2 done:

```markdown
2. **`ROBOT_HOST` + body resilience — done in Phase 4b.** `make_robot` now
   connects to a remote robot (`connection_mode="network"`) when `ROBOT_HOST`
   is set, and a `ReconnectingBody` keeps motions alive across a transient
   daemon/WiFi drop, degrading to a spoken body-less mode only after
   `BODY_RECONNECT_ATTEMPTS`. Still open: **robot media** does not hot-recover
   — on a stream drop the camera/mic go quiet (soft-degrade) until it returns;
   live media→Mac fallback is a later phase.
```

- [ ] **Step 3: `docs/testing.md`** — add smoke rows under the media-source section:

```markdown
**Body resilience (Phase 4b).** With the robot connected, pull the daemon
briefly mid-visit (`pkill -f reachy-mini-daemon`, then relaunch within a
motion or two): motions skip during the gap and resume silently on
reconnect. Kill it for good: after `BODY_RECONNECT_ATTEMPTS` the robot says
it has lost its body once and keeps answering (body-less). Remote robot:
set `REACHY_VEC_ROBOT_HOST=<addr>` and confirm it connects over the network.
```

- [ ] **Step 4: `.env.example`** — add under the common overrides:

```bash
# REACHY_VEC_ROBOT_HOST=reachy-mini.local   # remote robot; unset = local daemon
# REACHY_VEC_ROBOT_PORT=8000
```

- [ ] **Step 5: Full suite + lint, then commit**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass

```bash
git add -A
git commit -m "docs: ROBOT_HOST wired + body reconnection (Phase 4b)"
```

---

## Self-Review

**Spec coverage** (against 4b in the Phase 4 spec):
- "Wire the declared knob: `host`/`connection_mode` when `ROBOT_HOST` set, else auto" → Task 1. ✓
- "A reconnecting body wrapper; persistently-down link degrades to `NullBody`, surfaced in log + one spoken notice" → Task 2 (+ wiring Task 3). ✓
- "No new state-machine branches" → all logic in the body adapter; `oracle.py` untouched. ✓
- "Media loss handled the same way: fall back to Mac devices" → **intentionally descoped** per the user's decision; media keeps its 4a soft-degrade. Documented in Global Constraints, Task 4 gap note, and below.

**Deviation from spec (approved by the user):** live media→Mac hot-swap is deferred to its own phase. Rationale captured with the user: the camera/mic/speaker consumers capture the media handle at startup, so a live swap needs an indirection/re-wiring layer that doesn't exist; body reconnection is self-contained and delivers most of the resilience value, while robot media already fails soft (None frame / silence). Recorded in `docs/architecture.md` gap #2 and this plan.

**Placeholder scan:** none — every code step shows complete code; every test step shows assertions; every run step shows command + expected result.

**Type consistency:** `make_robot(with_media=False, connect=None) -> (Body, media|None)` unchanged from Phase 4a and reused as the reconnect thunk in Task 3. `ReconnectingBody(connect_body, max_attempts, announce)` defined in Task 2, consumed in Task 3 via `wrap_reconnect`. `wrap_reconnect(body, connect_body, announce) -> Body` defined and tested in Task 3. `FakeMini` (from Phase 4a's `tests/test_body.py`) reused in Task 1; `FlakyBody`/`_StubMini` are new local fakes. Config names `robot_port`, `robot_reconnect`, `body_reconnect_attempts` match between `config.py` and their call sites.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-phase4b-robot-host-resilience.md`.

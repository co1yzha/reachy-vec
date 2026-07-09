# Phase 4c — Barge-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person interrupt Reachy mid-reply by talking over it — the current sentence stops, the in-flight LLM stream is abandoned (partial reply kept in history so the conversation stays coherent), and the interrupting utterance becomes the next turn.

**Architecture:** Implements the approved [phase-2c spec](../specs/2026-07-08-phase2c-voice-bargein-design.md) (only its cloned-voice half shipped). Four seams, each with a test fake: (1) `Speaker` gains `stop()` to halt the current sentence; (2) a `SpeechInterrupted` exception + partial-reply handling in `ChatBrain`; (3) a `BargeInMonitor` thread that watches the mic (reusing the `AudioSource` seam from 4a and silero-VAD) and fires after sustained speech; (4) `OracleLoop` arms the monitor per reply, guards each streamed sentence, and skips the nod on interrupt. A `barge_in_factory=None` constructor arg keeps the feature off in tests and lets `run.py` wire it from config.

**Tech Stack:** Python 3.12, silero-vad, sounddevice, `subprocess.Popen`, threading, pytest, ruff, typer.

## Global Constraints

- Python **3.12+**; no new third-party dependency.
- **Every heavy dependency behind a Protocol with a test fake.** Tests must run with no real threads, mic, or model: `BargeInMonitor` takes an injectable `spawn` (run inline in tests) and `is_speech` (skip silero); the Oracle uses a scripted `FakeBargeInMonitor`.
- **`SpeechInterrupted` never escapes `respond`** — it is caught inside `ChatBrain`; the Oracle learns of the interrupt from `monitor.fired`, not an exception.
- **Backward compatible:** with `barge_in_factory=None` (the default, and every existing test), `on_sentence` is `speaker.speak` exactly as today. The whole existing suite must stay green.
- No new branches in the retrieval/tool logic — interruption only short-circuits sentence emission and the tool round-trip.
- After changes: `uv run ruff check src tests` and `uv run pytest -q` must both pass.

## Known limitations (carried from the phase-2c spec)

- No echo cancellation — a loud TV/nearby conversation can false-trigger; tune `BARGE_IN_MIN_SPEECH_S` up. Matters more with the robot's speaker near its mic (Phase 4a).
- On-robot audio out (`RobotAudioSink`) stop is best-effort: playback is one pushed buffer per sentence, so `stop()` prevents the *next* sentence rather than cutting the current one mid-buffer. The `say` and local-`sounddevice` backends stop mid-sentence.

---

### Task 1: `Speaker.stop()` across backends

**Files:**
- Modify: `src/reachy_vec/audio/speak.py`
- Modify: `tests/conftest.py` (`FakeSpeaker.stop`)
- Test: `tests/test_speak.py` (update the two `SaySpeaker` tests to the Popen form; add stop tests)

**Interfaces:**
- Produces:
  - `Speaker` protocol gains `def stop(self) -> None: ...` (interrupt current playback; no-op when idle).
  - `SaySpeaker(popen=subprocess.Popen)` — `speak` launches `say` via `Popen` and waits; `stop()` terminates a running process. (Replaces the old `run=` injection.)
  - `QwenTTSSpeaker(..., stop=None)` — `stop` defaults to `sounddevice.stop`; `QwenTTSSpeaker.stop()` calls it (halts local playback). The robot-sink path's `stop()` is a no-op (see Known limitations).
  - `FakeSpeaker.stop()` records a `stopped` count.

- [ ] **Step 1: Update `FakeSpeaker` in `tests/conftest.py`**

```python
class FakeSpeaker:
    """Records spoken lines and stop() calls."""

    def __init__(self):
        self.spoken: list[str] = []
        self.stopped = 0

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped += 1
```

- [ ] **Step 2: Rewrite the `SaySpeaker` tests and add stop tests in `tests/test_speak.py`**

Replace `test_say_speaker_invokes_say` and `test_say_speaker_skips_empty_text` with:

```python
class _FakeProc:
    def __init__(self):
        self.terminated = False
        self._done = False

    def wait(self):
        self._done = True

    def poll(self):
        return 0 if self._done else None

    def terminate(self):
        self.terminated = True


def test_say_speaker_invokes_say():
    cmds = []

    def popen(cmd):
        cmds.append(cmd)
        return _FakeProc()

    SaySpeaker(popen=popen).speak("hello team")
    assert cmds == [["say", "hello team"]]


def test_say_speaker_skips_empty_text():
    cmds = []
    SaySpeaker(popen=lambda cmd: cmds.append(cmd) or _FakeProc()).speak("   ")
    assert cmds == []


def test_say_speaker_stop_terminates_running_process():
    proc = _FakeProc()
    speaker = SaySpeaker(popen=lambda cmd: proc)
    speaker.speak("a long sentence")  # _FakeProc.wait() marks it done immediately
    proc._done = False                # pretend it's still playing
    speaker.stop()
    assert proc.terminated is True


def test_say_speaker_stop_is_safe_when_idle():
    SaySpeaker(popen=lambda cmd: _FakeProc()).stop()  # must not raise
```

Add near the qwen tests:

```python
def test_qwen_speaker_stop_calls_injected_stop():
    stops = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: ("AUDIO", 24000),
        play=lambda audio, sr: None,
        stop=lambda: stops.append(True),
    )
    speaker.stop()
    assert stops == [True]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_speak.py -v`
Expected: FAIL — `SaySpeaker` has no `popen`/`stop`, `QwenTTSSpeaker` has no `stop`.

- [ ] **Step 4: Implement in `speak.py`**

`Speaker` protocol:

```python
class Speaker(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
```

`SaySpeaker`:

```python
class SaySpeaker:
    """macOS `say` via Popen so a reply can be interrupted mid-sentence."""

    def __init__(self, popen=subprocess.Popen):
        self._popen = popen
        self._proc = None

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._proc = self._popen(["say", text])
        self._proc.wait()

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
```

`QwenTTSSpeaker`: add a `stop` param and method (default `sounddevice.stop`, imported lazily):

```python
    def __init__(
        self,
        sample_path: Path,
        sample_text: str | None = None,
        model_id: str = QWEN_TTS_DEFAULT_MODEL,
        generate=None,
        play=None,
        stop=None,
    ):
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_id = model_id
        self._generate = generate
        self._play = play or _play_blocking
        self._stop = stop

    def stop(self) -> None:
        stop = self._stop
        if stop is None:
            import sounddevice as sd

            stop = sd.stop
        stop()
```

Add a `stop()` to `RobotAudioSink`? No — the sink is a `play` callable, not a `Speaker`. The `QwenTTSSpeaker` wrapping a robot sink keeps `self._stop=None` → default `sd.stop()` is a harmless no-op for pushed audio (nothing playing via sounddevice). Leave as-is; documented under Known limitations.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_speak.py -v`
Expected: PASS (updated + new)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/speak.py tests/test_speak.py tests/conftest.py
git commit -m "feat: Speaker.stop() — interruptible say/qwen playback"
```

---

### Task 2: `SpeechInterrupted` + partial-reply handling in `ChatBrain`

**Files:**
- Modify: `src/reachy_vec/brain/chat.py`
- Test: `tests/test_chat_brain.py`

**Interfaces:**
- Produces:
  - `class SpeechInterrupted(Exception)` in `chat.py`.
  - `_complete_streaming` catches `SpeechInterrupted` raised from `on_sentence`, closes the stream, and returns a `_StreamedMessage(content, {}, interrupted=True)`.
  - `_StreamedMessage.__init__(self, content, tool_calls, interrupted=False)` with `self.interrupted`.
  - `respond` skips the tool round-trip when interrupted, appends the partial reply to history with a trailing `" -- (interrupted)"` marker, still increments `_exchanges`, and returns the partial text.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_chat_brain.py — check existing imports/fixtures at the top first
from reachy_vec.brain.chat import SpeechInterrupted


def test_interrupt_keeps_partial_reply_and_skips_tools(make_brain):
    # make_brain is the existing fixture; if the file builds ChatBrain inline,
    # mirror that setup here instead.
    brain = make_brain(reply="One sentence. Two sentence. Three sentence.")
    spoken = []

    def on_sentence(text):
        spoken.append(text)
        if len(spoken) == 1:  # interrupt right after the first sentence
            raise SpeechInterrupted()

    text = brain.respond("hi", on_sentence=on_sentence)
    assert spoken == ["One sentence."]           # only the first was spoken
    assert text.startswith("One sentence.")      # partial returned
    assert brain._history[-1]["content"].endswith(" -- (interrupted)")
    assert brain._exchanges == 1
```

Note: if `tests/test_chat_brain.py` has no `make_brain` fixture, build the brain the same way the existing tests do (Store + `FakeEmbedder` + `FakeLLMClient(reply=...)`), and use `FakeLLMClient` — its streaming path splits the reply into sentences via the shared `_message_to_stream` helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_brain.py -k interrupt -v`
Expected: FAIL — `ImportError: cannot import name 'SpeechInterrupted'`

- [ ] **Step 3: Implement in `chat.py`**

Add the exception near the top (after the imports):

```python
class SpeechInterrupted(Exception):
    """Raised from on_sentence when the user barges in; caught in ChatBrain."""
```

Wrap the streaming loop in `_complete_streaming`:

```python
    def _complete_streaming(self, on_sentence: Callable[[str], None]):
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": PERSONALITY}, *self._history],
            tools=TOOLS,
            stream=True,
            **self._llm_kwargs,
        )
        content, buffer = "", ""
        tool_calls: dict[int, dict] = {}
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                for tc in getattr(delta, "tool_calls", None) or []:
                    entry = tool_calls.setdefault(
                        tc.index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if getattr(tc.function, "name", None):
                        entry["name"] = tc.function.name
                    if getattr(tc.function, "arguments", None):
                        entry["arguments"] += tc.function.arguments
                if getattr(delta, "content", None):
                    content += delta.content
                    if not tool_calls:  # don't speak alongside pending tool calls
                        buffer += delta.content
                        buffer = _flush_sentences(buffer, on_sentence)
            if not tool_calls and buffer.strip():
                on_sentence(buffer.strip())
        except SpeechInterrupted:
            if hasattr(stream, "close"):
                stream.close()
            return _StreamedMessage(content, {}, interrupted=True)
        return _StreamedMessage(content, tool_calls)
```

Update `_StreamedMessage`:

```python
class _StreamedMessage:
    def __init__(self, content: str, tool_calls: dict[int, dict], interrupted: bool = False):
        self.content = content or None
        self.interrupted = interrupted
        self.tool_calls = [
            _ToolCall(entry["id"] or f"call_{index}", entry["name"], entry["arguments"])
            for index, entry in sorted(tool_calls.items())
        ] or None
```

Update `respond` (the block after the first `_complete`):

```python
        message = self._complete(on_sentence)
        interrupted = getattr(message, "interrupted", False)
        if not interrupted and getattr(message, "tool_calls", None):
            self._history.append(_assistant_tool_message(message))
            for call in message.tool_calls:
                self._history.append(self._execute_tool(call))
            message = self._complete(on_sentence)
            interrupted = getattr(message, "interrupted", False)
        text = (message.content or "").strip()
        self._history.append(
            {
                "role": "assistant",
                "content": f"{text} -- (interrupted)" if interrupted else text,
            }
        )
        self._exchanges += 1
        self._trim()
        logger.info("reply to %s: %r%s", self._turn.name or "user", text,
                    " (interrupted)" if interrupted else "")
        return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_brain.py -v`
Expected: PASS (existing + new)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/chat.py tests/test_chat_brain.py
git commit -m "feat: SpeechInterrupted — keep partial reply, skip tools on barge-in"
```

---

### Task 3: `BargeInMonitor`

**Files:**
- Modify: `src/reachy_vec/audio/listen.py`
- Test: `tests/test_listen.py`

**Interfaces:**
- Consumes: an `AudioSource` (Task-2a seam), `SAMPLE_RATE`, `CHUNK_S`.
- Produces: `class BargeInMonitor:` `__init__(self, source, min_speech_s=0.7, sample_rate=SAMPLE_RATE, is_speech=None, spawn=None)`; `start(self, on_fire)` (resets state, launches the watch); `stop(self)` (signals the watch to end); attributes `fired: bool` and `broken: bool`. The watch reads frames, counts consecutive speech chunks, and on reaching `min_speech_s / CHUNK_S` sets `fired=True` and calls `on_fire()`. `spawn` defaults to a daemon thread; tests inject `spawn=lambda fn: fn()` to run inline. `is_speech` defaults to a silero-VAD gate; tests inject a stub.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_listen.py (ScriptedSource already defined in this file)
from reachy_vec.audio.listen import BargeInMonitor  # noqa: E402


def _inline(fn):
    fn()  # run the watch synchronously; returns None (no thread)


def test_barge_in_fires_after_sustained_speech():
    speech = np.ones(512, dtype=np.float32)
    src = ScriptedSource([speech] * 5)
    fired = []
    mon = BargeInMonitor(
        src, min_speech_s=0.096, is_speech=lambda f: bool(f[0]), spawn=_inline
    )  # 0.096 / 0.032 = 3 chunks
    mon.start(on_fire=lambda: fired.append(True))
    assert mon.fired is True
    assert fired == [True]


def test_barge_in_ignores_brief_speech():
    speech, silence = np.ones(512, dtype=np.float32), np.zeros(512, dtype=np.float32)
    src = ScriptedSource([speech, silence, speech, silence])  # never 3 in a row
    mon = BargeInMonitor(
        src, min_speech_s=0.096, is_speech=lambda f: bool(f[0]), spawn=_inline
    )
    mon.start(on_fire=lambda: None)
    assert mon.fired is False


def test_barge_in_survives_a_broken_source():
    class Boom:
        def frames(self, chunk_samples):
            raise RuntimeError("mic gone")
            yield  # pragma: no cover

    mon = BargeInMonitor(Boom(), is_speech=lambda f: True, spawn=_inline)
    mon.start(on_fire=lambda: None)  # must not raise
    assert mon.broken is True
    assert mon.fired is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_listen.py -k barge -v`
Expected: FAIL — `ImportError: cannot import name 'BargeInMonitor'`

- [ ] **Step 3: Implement `BargeInMonitor` in `listen.py`**

Add `import threading` at the top, then the class near `MicSource`:

```python
class BargeInMonitor:
    """Watch the mic while Reachy speaks; fire after sustained speech.

    Shares the AudioSource seam and silero-VAD. The ~min_speech_s hysteresis
    is what stops Reachy's own voice (heard through the mic) from triggering.
    """

    def __init__(self, source, min_speech_s=0.7, sample_rate=SAMPLE_RATE, is_speech=None, spawn=None):
        self._source = source
        self._min_chunks = max(1, int(min_speech_s / CHUNK_S))
        self._sample_rate = sample_rate
        self._is_speech = is_speech
        self._spawn = spawn or _daemon_spawn
        self.fired = False
        self.broken = False
        self._stop = False
        self._on_fire = None
        self._vad = None

    def start(self, on_fire) -> None:
        self.fired = False
        self._stop = False
        self._on_fire = on_fire
        self._spawn(self._watch)

    def stop(self) -> None:
        self._stop = True

    def _watch(self) -> None:
        try:
            is_speech = self._is_speech or self._silero_is_speech()
            chunk_samples = int(self._sample_rate * CHUNK_S)
            run = 0
            frame_iter = self._source.frames(chunk_samples)
            try:
                for frame in frame_iter:
                    if self._stop:
                        return
                    if is_speech(frame):
                        run += 1
                        if run >= self._min_chunks:
                            self.fired = True
                            if self._on_fire:
                                self._on_fire()
                            return
                    else:
                        run = 0
            finally:
                frame_iter.close()
        except Exception:
            logger.exception("barge-in monitor crashed; disabling for the session")
            self.broken = True

    def _silero_is_speech(self):
        import torch
        from silero_vad import load_silero_vad

        if self._vad is None:
            self._vad = load_silero_vad()

        def is_speech(frame):
            return self._vad(torch.from_numpy(frame), self._sample_rate).item() > 0.5

        return is_speech


def _daemon_spawn(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_listen.py -v`
Expected: PASS (existing + 3 new)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/listen.py tests/test_listen.py
git commit -m "feat: BargeInMonitor — fire on sustained speech over the mic"
```

---

### Task 4: Oracle wiring

**Files:**
- Modify: `src/reachy_vec/brain/oracle.py`
- Modify: `tests/conftest.py` (`FakeBargeInMonitor`)
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `SpeechInterrupted` (Task 2); a monitor object with `start(on_fire)`, `stop()`, `fired`.
- Produces:
  - `OracleLoop.__init__(..., barge_in_factory=None)` — a zero-arg callable returning a fresh monitor per reply; `None` = feature off (today's behavior, `on_sentence=speaker.speak`).
  - Per reply: build a monitor, `start(on_fire=self._speaker.stop)`, pass a guarded `on_sentence` that raises `SpeechInterrupted` when `monitor.fired`, and in a `finally` call `monitor.stop()`. If `monitor.fired`, skip the `nod` and loop straight back to listening.
  - `tests/conftest.py`: `FakeBargeInMonitor(fire_after=None)` — a scripted monitor whose guarded `on_sentence` fires after N spoken sentences.

- [ ] **Step 1: Add `FakeBargeInMonitor` to `tests/conftest.py`**

```python
class FakeBargeInMonitor:
    """Scripted barge-in: fires (sets .fired) after `fire_after` on_fire arms.

    The Oracle calls start(on_fire) then speaks; the guarded on_sentence checks
    .fired before each sentence. Here we flip .fired on the Nth started reply's
    first fired-check by having the test set fire_after / calling trip()."""

    def __init__(self, fire_on_sentence=None):
        self.fire_on_sentence = fire_on_sentence  # 1-based sentence index to fire before
        self.fired = False
        self.started = 0
        self.stopped = 0
        self._seen = 0

    def start(self, on_fire):
        self.started += 1
        self._seen = 0
        self.fired = False
        self._on_fire = on_fire

    def stop(self):
        self.stopped += 1

    def trip(self):
        """Simulate the monitor thread firing."""
        self.fired = True
        self._on_fire()
```

- [ ] **Step 2: Write the failing test**

```python
# add to tests/test_oracle.py — mirror the existing OracleLoop construction helper
from reachy_vec.brain.chat import SpeechInterrupted
from tests.conftest import FakeBargeInMonitor


def test_barge_in_stops_speaker_and_skips_nod(oracle_env):
    """A tripped monitor stops the speaker and the loop skips the nod."""
    monitor = FakeBargeInMonitor()
    speaker = FakeSpeaker()

    class InterruptingBrain:
        def begin_conversation(self, *a):
            pass

        def end_conversation(self):
            pass

        def respond(self, question, identity=None, on_sentence=None):
            monitor.trip()                 # user starts talking
            try:
                on_sentence("half a sen")  # guarded -> should raise
            except SpeechInterrupted:
                return "half a sen"        # ChatBrain would swallow it
            return "full reply"

    # Build the loop with barge_in_factory=lambda: monitor and the brain above,
    # one recognized face then silence (reuse the file's fixture/helpers).
    loop = build_oracle(
        brain=InterruptingBrain(),
        speaker=speaker,
        barge_in_factory=lambda: monitor,
        sights=[recognized_face, None],
        utterances=["question", None],
    )
    loop.run_once()
    assert monitor.started == 1
    assert monitor.stopped == 1
    assert speaker.stopped == 1              # on_fire called speaker.stop
    assert "nod" not in [m for m in loop_body_motions(loop)]  # nod skipped
```

Note: adapt `build_oracle` / `recognized_face` / `loop_body_motions` to the helpers already in `tests/test_oracle.py`. If the file constructs `OracleLoop(...)` inline per test, do the same and assert on the injected `FakeBody().motions`.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -k barge -v`
Expected: FAIL — `OracleLoop.__init__` has no `barge_in_factory`.

- [ ] **Step 4: Implement in `oracle.py`**

Add the import and constructor arg:

```python
from reachy_vec.brain.chat import SpeechInterrupted
```

```python
        speaker_id=None,
        voice_passive_cap: int = 10,
        barge_in_factory=None,
    ):
        ...
        self._barge_in_factory = barge_in_factory
```

Replace the reply block inside `_converse`'s loop:

```python
            voice_obs = self._identify_voice(utterance.audio)
            monitor = self._barge_in_factory() if self._barge_in_factory else None
            if monitor is not None:
                monitor.start(on_fire=self._speaker.stop)

                def on_sentence(text, _m=monitor):
                    if _m.fired:
                        raise SpeechInterrupted()
                    self._speaker.speak(text)
            else:
                on_sentence = self._speaker.speak
            try:
                self._brain.respond(
                    utterance.text,
                    identity=fuse(face_obs, voice_obs),
                    on_sentence=on_sentence,
                )
            except Exception:
                logger.exception("brain.respond failed")
                self._speaker.speak(APOLOGY)
            finally:
                if monitor is not None:
                    monitor.stop()
            if monitor is not None and monitor.fired:
                self._maybe_bank_voice(face_obs, voice_obs, utterance.audio)
                continue  # user is already talking; skip the nod, listen again
            self._body.perform("nod")
            self._maybe_bank_voice(face_obs, voice_obs, utterance.audio)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: PASS (existing + new)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/brain/oracle.py tests/test_oracle.py tests/conftest.py
git commit -m "feat: Oracle arms BargeInMonitor per reply; skips nod on interrupt"
```

---

### Task 5: `run.py` wiring, config, docs

**Files:**
- Modify: `src/reachy_vec/config.py` (`barge_in`, `barge_in_min_speech_s`)
- Modify: `src/reachy_vec/cli/run.py`
- Modify: `docs/configuration.md`, `docs/architecture.md`, `docs/testing.md`, `.env.example`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `BargeInMonitor` (Task 3), the `audio_source`/`MicSource` selection from Phase 4a, `OracleLoop(barge_in_factory=)` (Task 4).
- Produces:
  - `config.py`: `barge_in: bool = True`, `barge_in_min_speech_s: float = 0.7`.
  - `cli/run.py`: `make_barge_in_factory(chosen, media) -> Callable | None` — returns `None` when `settings.barge_in` is false; otherwise a zero-arg factory building a `BargeInMonitor` over a fresh source (`RobotAudioSource(media)` for the robot world, else `MicSource()`), with `min_speech_s=settings.barge_in_min_speech_s`. Wired into `OracleLoop(barge_in_factory=...)`.

- [ ] **Step 1: Add config knobs**

In `config.py`, under `# Interaction`:

```python
    barge_in: bool = True  # allow talking over a reply to interrupt it
    barge_in_min_speech_s: float = 0.7  # sustained speech needed to fire (raise if false triggers)
```

- [ ] **Step 2: Write the failing tests**

```python
# add to tests/test_cli.py
from reachy_vec.cli.run import make_barge_in_factory


def test_barge_in_factory_none_when_disabled(monkeypatch):
    monkeypatch.setattr("reachy_vec.cli.run.settings.barge_in", False)
    assert make_barge_in_factory("mac", media=None) is None


def test_barge_in_factory_builds_monitor_when_enabled(monkeypatch):
    from reachy_vec.audio.listen import BargeInMonitor

    monkeypatch.setattr("reachy_vec.cli.run.settings.barge_in", True)
    factory = make_barge_in_factory("mac", media=None)
    assert isinstance(factory(), BargeInMonitor)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k barge -v`
Expected: FAIL — `ImportError: cannot import name 'make_barge_in_factory'`

- [ ] **Step 4: Implement in `run.py`**

Add near the other helpers:

```python
def make_barge_in_factory(chosen: str, media):
    """Zero-arg factory building a fresh BargeInMonitor per reply, or None."""
    if not settings.barge_in:
        return None
    from reachy_vec.audio.listen import BargeInMonitor, MicSource
    from reachy_vec.audio.sources import RobotAudioSource

    def factory():
        source = (
            RobotAudioSource(media, target_rate=settings.audio_input_rate)
            if chosen == "robot"
            else MicSource()
        )
        return BargeInMonitor(source, min_speech_s=settings.barge_in_min_speech_s)

    return factory
```

Wire it into the `OracleLoop(...)` construction:

```python
        speaker_id=speaker_id,
        voice_passive_cap=settings.voice_passive_cap,
        barge_in_factory=make_barge_in_factory(chosen, media),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing + 2 new)

- [ ] **Step 6: Docs**

- `docs/configuration.md` → new **Interaction pacing** rows: `BARGE_IN` (`true`) and `BARGE_IN_MIN_SPEECH_S` (`0.7`, "raise if a loud room false-triggers; no echo cancellation").
- `docs/architecture.md` → "Known gaps" gap #3: mark barge-in **done in Phase 4c**; note the on-robot-audio stop caveat.
- `docs/testing.md` → the phase-2c smoke rows: interrupt mid-answer → stops within a beat and answers the new question; brief "mm-hm" while it talks → no trigger; `REACHY_VEC_BARGE_IN=false` → old behavior.
- `.env.example` → commented `REACHY_VEC_BARGE_IN=true`.

- [ ] **Step 7: Full suite + lint, then commit**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass

```bash
git add -A
git commit -m "feat: wire barge-in into run (config-gated) + docs"
```

---

## Self-Review

**Spec coverage** (against the phase-2c spec):
- `Speaker.stop()`, `SaySpeaker` Popen, `QwenSpeaker` stop → Task 1. ✓
- `SpeechInterrupted`, `_complete_streaming` catch, partial reply + `-- (interrupted)` marker, skip tool round-trip, exchanges still increments → Task 2. ✓
- `BargeInMonitor` sustained-speech gate over the AudioSource + silero-VAD, `broken` flag → Task 3. ✓
- Oracle arms monitor (`on_fire=speaker.stop`), guarded `on_sentence` raises, disarm in finally, skip nod + listen again on interrupt, `barge_in_factory=None` off-switch → Task 4. ✓
- `BARGE_IN` / `BARGE_IN_MIN_SPEECH_S` config + run wiring → Task 5. ✓
- Mic sharing (monitor only while speaking, transcriber only while listening) → structural: the monitor is started per reply and stopped in `finally` before the next `listen_once`; documented. ✓

**Deviation from spec (documented):** the spec's chunked-playback `QwenSpeaker.stop()` that halts within ~50 ms is implemented as `sounddevice.stop()` (local playback) — simpler and sufficient for the current one-shot `_play_blocking`. On-robot `RobotAudioSink` stop is best-effort (per-sentence buffer), captured under Known limitations. The `.vscode/launch.json` F5 piece from the phase-2c spec is out of scope here (local-only, gitignored).

**Placeholder scan:** the only "adapt to existing helpers" notes are in Task 2 and Task 4 test steps, where the test must match `tests/test_chat_brain.py` / `tests/test_oracle.py`'s existing construction style — the executor reads those files and mirrors them. Every implementation step shows complete code.

**Type consistency:** `Speaker.stop()` (Task 1) is called by the Oracle's `on_fire=self._speaker.stop` (Task 4) and by `FakeSpeaker.stop` (Task 1). `SpeechInterrupted` defined in Task 2, imported in Task 4. `BargeInMonitor(source, min_speech_s, ..., is_speech, spawn)` (Task 3) built by `make_barge_in_factory` (Task 5) and faked by `FakeBargeInMonitor` (Task 4) — the Oracle only uses `.start(on_fire)`, `.stop()`, `.fired`, which both the real and fake provide. `barge_in_factory` name matches across `OracleLoop.__init__` (Task 4) and `run.py` (Task 5).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-phase4c-barge-in.md`.

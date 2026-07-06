# Phase 1b: Voice — Speak & Listen on Mac Audio

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Speaker` (text → Mac speaker, `say` backend) and `Transcriber` (Mac mic → VAD-segmented utterance → text via faster-whisper), both behind protocols with fakes.

**Architecture:** `audio/speak.py`: `Speaker` protocol; `SaySpeaker` shells out to macOS `say`; `make_speaker()` selects by `settings.tts_backend` (fish-speech/openvoice slots raise a clear not-implemented message for now). `audio/listen.py`: `Transcriber` protocol with `listen_once(timeout_s) -> str | None`; `MicTranscriber` records via sounddevice, segments with silero-VAD, transcribes with faster-whisper (models lazy-loaded).

**Tech Stack:** sounddevice, silero-vad, faster-whisper, macOS `say`, pytest.

**Depends on:** nothing. **Unblocks:** 1d.

## Global Constraints

- Python `>=3.12`; run everything through `uv run`.
- Tests never load whisper/VAD models or open the mic/speaker; real paths are covered by the manual smoke test.
- STT model size from `settings.stt_model` (default `"small"`); TTS backend from `settings.tts_backend`.
- Commit after every green test cycle; conventional-commit messages.

---

### Task 1: Speaker

**Files:**
- Modify: `src/reachy_vec/audio/speak.py` (docstring stub)
- Test: `tests/test_speak.py`

**Interfaces:**
- Produces: `Speaker` protocol with `speak(text: str) -> None`; `SaySpeaker(run=subprocess.run)`; `make_speaker() -> Speaker`. 1d consumes both.

- [ ] **Step 1: Write the failing test**

`tests/test_speak.py`:

```python
import pytest

from reachy_vec.audio.speak import SaySpeaker, make_speaker


def test_say_speaker_invokes_say():
    calls = []
    speaker = SaySpeaker(run=lambda cmd, **kw: calls.append(cmd))
    speaker.speak("hello team")
    assert calls == [["say", "hello team"]]


def test_say_speaker_skips_empty_text():
    calls = []
    SaySpeaker(run=lambda cmd, **kw: calls.append(cmd)).speak("   ")
    assert calls == []


def test_make_speaker_say_backend(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "say")
    assert isinstance(make_speaker(), SaySpeaker)


def test_make_speaker_unimplemented_backend_raises(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "fish-speech")
    with pytest.raises(NotImplementedError, match="fish-speech"):
        make_speaker()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speak.py -v`
Expected: FAIL with `ImportError: cannot import name 'SaySpeaker'`.

- [ ] **Step 3: Write the implementation**

Replace `src/reachy_vec/audio/speak.py` with:

```python
"""Text-to-speech behind a pluggable Speaker protocol.

Backends (settings.tts_backend): "say" (macOS built-in, no cloning) now;
"fish-speech" (voice clone, primary per spec) and "openvoice" (fallback)
land in a later plan and use settings.voice_sample as the clone reference.
"""

import subprocess
from typing import Protocol

from reachy_vec.config import settings


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class SaySpeaker:
    """macOS `say` — dev/debug backend, blocks until speech finishes."""

    def __init__(self, run=subprocess.run):
        self._run = run

    def speak(self, text: str) -> None:
        text = text.strip()
        if text:
            self._run(["say", text], check=False)


def make_speaker() -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        return SaySpeaker()
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired yet - set REACHY_VEC_TTS_BACKEND=say"
    )
```

- [ ] **Step 4: Run tests, then commit**

Run: `uv run pytest -q` — expected: all PASS.

```bash
git add src/reachy_vec/audio/speak.py tests/test_speak.py
git commit -m "feat: Speaker protocol with macOS say backend"
```

---

### Task 2: Transcriber

**Files:**
- Modify: `src/reachy_vec/audio/listen.py` (docstring stub)
- Modify: `tests/conftest.py` (add fakes for 1d reuse)
- Test: `tests/test_listen.py`

**Interfaces:**
- Produces: `Transcriber` protocol with `listen_once(timeout_s: float) -> str | None` (None = silence until timeout); `MicTranscriber(model_size: str, sample_rate: int = 16000)`; helper `collect_utterance(chunks, is_speech, max_silence_chunks) -> list | None` (pure, unit-tested). `FakeTranscriber` and `FakeSpeaker` added to `tests/conftest.py`. 1d consumes the protocol and fakes.

- [ ] **Step 1: Add dependencies**

```bash
uv add sounddevice silero-vad faster-whisper
```

(They move from the `perception` extra to main deps — `run` needs them at runtime. Remove `faster-whisper` and `sounddevice` from the `perception` extra in `pyproject.toml` in this commit.)

- [ ] **Step 2: Write the failing test**

Append to `tests/conftest.py`:

```python
class FakeSpeaker:
    """Records spoken lines."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeTranscriber:
    """Returns scripted utterances, then None (silence)."""

    def __init__(self, utterances: list[str]):
        self._it = iter(utterances)

    def listen_once(self, timeout_s: float) -> str | None:
        return next(self._it, None)
```

`tests/test_listen.py`:

```python
from reachy_vec.audio.listen import collect_utterance

from tests.conftest import FakeTranscriber

SPEECH, SILENCE = "s", "."


def run_collect(pattern: str, max_silence: int = 2):
    chunks = list(pattern)
    return collect_utterance(
        iter(chunks), is_speech=lambda c: c == SPEECH, max_silence_chunks=max_silence
    )


def test_collects_speech_and_stops_after_trailing_silence():
    assert run_collect("..sss..x") == ["s", "s", "s", ".", "."]


def test_returns_none_when_no_speech_at_all():
    assert run_collect("......") is None


def test_short_pause_inside_utterance_is_kept():
    assert run_collect("ss.ss..", max_silence=2) == ["s", "s", ".", "s", "s", ".", "."]


def test_fake_transcriber_scripts_then_silence():
    t = FakeTranscriber(["hello"])
    assert t.listen_once(5) == "hello"
    assert t.listen_once(5) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_listen.py -v`
Expected: FAIL with `ImportError: cannot import name 'collect_utterance'`.

- [ ] **Step 4: Write the implementation**

Replace `src/reachy_vec/audio/listen.py` with:

```python
"""Mic capture -> VAD-segmented utterance -> text (faster-whisper).

collect_utterance is pure and unit-tested; MicTranscriber does the device
and model work and is covered by the manual smoke test only.
"""

import logging
from typing import Callable, Iterator, Protocol

import numpy as np

from reachy_vec.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_S = 0.032  # silero-vad native frame for 16 kHz (512 samples)


class Transcriber(Protocol):
    def listen_once(self, timeout_s: float) -> str | None: ...


def collect_utterance(chunks: Iterator, is_speech: Callable, max_silence_chunks: int):
    """Accumulate chunks from first speech until max_silence_chunks of quiet.

    Returns the collected chunk list (speech plus trailing/inner silence),
    or None if the iterator ends before any speech occurs.
    """
    collected: list = []
    silence_run = 0
    for chunk in chunks:
        if is_speech(chunk):
            collected.append(chunk)
            silence_run = 0
        elif collected:
            collected.append(chunk)
            silence_run += 1
            if silence_run >= max_silence_chunks:
                break
    return collected or None


class MicTranscriber:
    """Blocks on the default input device; lazy-loads VAD + whisper models."""

    def __init__(self, model_size: str | None = None, sample_rate: int = SAMPLE_RATE):
        self._model_size = model_size or settings.stt_model
        self._sample_rate = sample_rate
        self._vad = None
        self._whisper = None

    def _load(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(self._model_size, compute_type="int8")

    def listen_once(self, timeout_s: float) -> str | None:
        import sounddevice as sd
        import torch

        self._load()
        chunk_samples = int(self._sample_rate * CHUNK_S)
        max_chunks = int(timeout_s / CHUNK_S)
        max_silence = int(0.8 / CHUNK_S)  # 0.8 s of quiet ends the utterance

        def frames() -> Iterator[np.ndarray]:
            with sd.InputStream(
                samplerate=self._sample_rate, channels=1, dtype="float32"
            ) as stream:
                for _ in range(max_chunks):
                    data, _overflow = stream.read(chunk_samples)
                    yield data[:, 0].copy()

        def is_speech(chunk: np.ndarray) -> bool:
            prob = self._vad(torch.from_numpy(chunk), self._sample_rate).item()
            return prob > 0.5

        collected = collect_utterance(frames(), is_speech, max_silence)
        if collected is None:
            return None
        audio = np.concatenate(collected)
        segments, _info = self._whisper.transcribe(audio, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("heard: %r", text)
        return text or None
```

- [ ] **Step 5: Run tests, then commit**

Run: `uv run pytest -q` — expected: all PASS.

```bash
git add pyproject.toml uv.lock src/reachy_vec/audio/listen.py tests/conftest.py tests/test_listen.py
git commit -m "feat: Transcriber with silero-VAD segmentation and faster-whisper STT"
```

- [ ] **Step 6: Manual smoke test (speak + echo; needs mic permission)**

```bash
uv run python -c "
from reachy_vec.audio.speak import make_speaker
from reachy_vec.audio.listen import MicTranscriber
s = make_speaker()
s.speak('Say something after the beep.')
t = MicTranscriber()
print('listening 10s...'); text = t.listen_once(10)
s.speak(f'I heard: {text}' if text else 'I heard nothing.')
print('transcript:', text)
"
```

Expected: first run downloads whisper-small (~500 MB) and the VAD model; then your sentence is transcribed and spoken back. macOS will prompt for microphone permission for your terminal — accept it.

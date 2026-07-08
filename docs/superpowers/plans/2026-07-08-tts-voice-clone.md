# Qwen3-TTS Voice-Clone Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The robot speaks in a cloned voice (from a short local WAV of the user) via a new `qwen-tts` backend behind the existing `Speaker` protocol.

**Architecture:** A new `QwenTTSSpeaker` class in `src/reachy_vec/audio/speak.py` lazy-loads Qwen3-TTS 0.6B through mlx-audio on first `speak()`, synthesizes each sentence conditioned on `settings.voice_sample`, and plays it blocking through sounddevice — same contract as `SaySpeaker`, so `OracleLoop`/`ChatBrain` are untouched. `make_speaker()` selects it when `settings.tts_backend == "qwen-tts"` and fails fast if the voice sample is missing.

**Tech Stack:** mlx-audio (MLX, Apple silicon), model `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`, sounddevice (already a dependency), pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-tts-voice-clone-design.md`

## Global Constraints

- Heavy imports (`mlx_audio`, `numpy` for playback) deferred inside methods/functions — `import reachy_vec` and the test suite must stay fast (repo convention).
- No model download, network, or audio device in the test suite — tests inject fake `generate`/`play` callables.
- Default backend stays `say`; existing tests and CI behavior unchanged.
- Synthesis failure mid-conversation is logged and the sentence skipped — never crashes the Oracle loop.
- New config knobs go in `config.py` (`REACHY_VEC_` env prefix), not module constants.
- Run `uv run pytest -q` and `uv run ruff check src tests` before every commit; both must pass.

---

### Task 1: `QwenTTSSpeaker` class

**Files:**
- Modify: `src/reachy_vec/audio/speak.py`
- Test: `tests/test_speak.py`

**Interfaces:**
- Consumes: nothing new (stdlib + existing module).
- Produces: `QwenTTSSpeaker(sample_path: Path, sample_text: str | None = None, model_id: str = QWEN_TTS_DEFAULT_MODEL, generate=None, play=None)` with method `speak(text: str) -> None`. Test seams: `generate(text: str) -> tuple[audio, int]` (waveform, sample rate) and `play(audio, sample_rate: int) -> None`. Module constant `QWEN_TTS_DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"`. Task 2 wires this into `make_speaker()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speak.py`:

```python
def test_qwen_speaker_synthesizes_and_plays():
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: (f"AUDIO<{text}>", 24000),
        play=lambda audio, sr: played.append((audio, sr)),
    )
    speaker.speak("hello team")
    assert played == [("AUDIO<hello team>", 24000)]


def test_qwen_speaker_skips_empty_text():
    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=lambda text: ("AUDIO", 24000),
        play=lambda audio, sr: played.append((audio, sr)),
    )
    speaker.speak("   ")
    assert played == []


def test_qwen_speaker_logs_and_skips_on_synthesis_error(caplog):
    def boom(text):
        raise RuntimeError("mlx exploded")

    played = []
    speaker = QwenTTSSpeaker(
        sample_path=Path("me.wav"),
        generate=boom,
        play=lambda audio, sr: played.append((audio, sr)),
    )
    with caplog.at_level(logging.ERROR):
        speaker.speak("hello")  # must not raise
    assert played == []
    assert "TTS synthesis failed" in caplog.text
```

And extend the imports at the top of the file:

```python
import logging
from pathlib import Path

import pytest

from reachy_vec.audio.speak import QwenTTSSpeaker, SaySpeaker, make_speaker
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speak.py -q`
Expected: ImportError — `cannot import name 'QwenTTSSpeaker'`

- [ ] **Step 3: Implement `QwenTTSSpeaker`**

In `src/reachy_vec/audio/speak.py`, add below the imports (`import logging` and `from pathlib import Path` join the existing imports; `logger = logging.getLogger(__name__)` after them, matching `listen.py`):

```python
QWEN_TTS_DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


def _play_blocking(audio, sample_rate: int) -> None:
    import numpy as np
    import sounddevice as sd

    sd.play(np.asarray(audio, dtype=np.float32), sample_rate)
    sd.wait()


class QwenTTSSpeaker:
    """Qwen3-TTS via mlx-audio — clones the voice in `sample_path`.

    Model + clone conditioning load lazily on first speak() and stay cached.
    `generate` / `play` are injectable for tests (no model, no audio device).
    """

    def __init__(
        self,
        sample_path: Path,
        sample_text: str | None = None,
        model_id: str = QWEN_TTS_DEFAULT_MODEL,
        generate=None,
        play=None,
    ):
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_id = model_id
        self._generate = generate
        self._play = play or _play_blocking

    def _ensure_generate(self):
        if self._generate is None:
            import numpy as np
            from mlx_audio.tts.utils import load_model

            model = load_model(self._model_id)

            def generate(text: str):
                kwargs = {"text": text, "ref_audio": str(self._sample_path)}
                if self._sample_text:
                    kwargs["ref_text"] = self._sample_text
                results = list(model.generate(**kwargs))
                audio = np.concatenate([np.asarray(r.audio) for r in results])
                return audio, results[0].sample_rate

            self._generate = generate
        return self._generate

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            audio, sample_rate = self._ensure_generate()(text)
            self._play(audio, sample_rate)
        except Exception:
            logger.exception("TTS synthesis failed; skipping sentence")
```

Also update the module docstring to reflect reality:

```python
"""Text-to-speech behind a pluggable Speaker protocol.

Backends (settings.tts_backend): "say" (macOS built-in, no cloning) and
"qwen-tts" (Qwen3-TTS via mlx-audio, clones settings.voice_sample).
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speak.py -q`
Expected: all pass (existing 4 + new 3)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/speak.py tests/test_speak.py
git commit -m "feat: QwenTTSSpeaker - voice-clone TTS via mlx-audio"
```

---

### Task 2: Config knobs, `make_speaker` wiring, mlx-audio dependency

**Files:**
- Modify: `src/reachy_vec/config.py:29-30`
- Modify: `src/reachy_vec/audio/speak.py` (`make_speaker`)
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_speak.py`

**Interfaces:**
- Consumes: `QwenTTSSpeaker` and `QWEN_TTS_DEFAULT_MODEL` from Task 1.
- Produces: settings `tts_backend` (now `say | qwen-tts`), `tts_model: str`, `voice_sample: Path | None` (existing), `voice_sample_text: str | None`; `make_speaker()` returns `QwenTTSSpeaker` for `qwen-tts`, raises `ValueError` when `voice_sample` is missing/not a file.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speak.py`:

```python
def test_make_speaker_qwen_backend(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    assert isinstance(make_speaker(), QwenTTSSpeaker)


def test_make_speaker_qwen_requires_voice_sample(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", None)
    with pytest.raises(ValueError, match="REACHY_VEC_VOICE_SAMPLE"):
        make_speaker()


def test_make_speaker_qwen_rejects_missing_sample_file(monkeypatch, tmp_path):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr(
        "reachy_vec.audio.speak.settings.voice_sample", tmp_path / "nope.wav"
    )
    with pytest.raises(ValueError, match="REACHY_VEC_VOICE_SAMPLE"):
        make_speaker()
```

Also update the existing unimplemented-backend test — `fish-speech` is no longer the example:

```python
def test_make_speaker_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "kokoro")
    with pytest.raises(NotImplementedError, match="kokoro"):
        make_speaker()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speak.py -q`
Expected: the three new tests FAIL (`NotImplementedError: TTS backend 'qwen-tts'`); renamed test passes.

- [ ] **Step 3: Update config and `make_speaker`**

In `src/reachy_vec/config.py`, replace lines 29–30 with:

```python
    tts_backend: str = "say"  # say (macOS built-in) | qwen-tts (voice clone, local MLX)
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"  # mlx-audio model id
    voice_sample: Path | None = None  # ~10s clean WAV of the voice to clone (qwen-tts)
    voice_sample_text: str | None = None  # its transcript; omit -> auto-transcribed once
```

In `src/reachy_vec/audio/speak.py`, replace `make_speaker` with:

```python
def make_speaker() -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        return SaySpeaker()
    if backend == "qwen-tts":
        sample = settings.voice_sample
        if sample is None or not Path(sample).is_file():
            raise ValueError(
                "tts_backend=qwen-tts needs REACHY_VEC_VOICE_SAMPLE pointing to "
                "a short WAV of the voice to clone (see .env.example)"
            )
        return QwenTTSSpeaker(
            sample_path=Path(sample),
            sample_text=settings.voice_sample_text,
            model_id=settings.tts_model,
        )
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired - use 'say' or 'qwen-tts'"
    )
```

- [ ] **Step 4: Add the mlx-audio dependency**

Run: `uv add mlx-audio`
Expected: resolves and pins in `pyproject.toml`/`uv.lock`. Then confirm the suite still imports fast (mlx-audio must not be imported at module import time): `uv run pytest tests/test_speak.py -q`

- [ ] **Step 5: Run full tests to verify everything passes**

Run: `uv run pytest -q`
Expected: full suite PASSES

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/config.py src/reachy_vec/audio/speak.py tests/test_speak.py pyproject.toml uv.lock
git commit -m "feat: qwen-tts backend selectable via config, fail-fast on missing voice sample"
```

---

### Task 3: Docs, .env.example, smoke-test checklist

**Files:**
- Modify: `.env.example`
- Modify: `README.md` (setup/usage area — find the section that mentions TTS or `run`)
- Modify: `docs/testing.md` (manual smoke-test checklist)
- Modify: `docs/architecture.md` and `docs/pipelines.md` (wherever `say`/fish-speech/TTS is mentioned — `grep -rn -i "fish-speech\|tts" docs/*.md`)

**Interfaces:**
- Consumes: env var names and defaults exactly as defined in Task 2.
- Produces: user-facing docs; no code.

- [ ] **Step 1: Add opt-in lines to `.env.example`**

Append:

```bash
# Voice-clone TTS (optional; default backend is macOS `say`).
# Record ~10s of clean speech first, e.g.:
#   sox -d -r 24000 -c 1 data/voice_sample.wav trim 0 10
# First qwen-tts run downloads the model from Hugging Face (~1.5 GB).
#REACHY_VEC_TTS_BACKEND=qwen-tts
#REACHY_VEC_VOICE_SAMPLE=data/voice_sample.wav
# Optional transcript of the sample; skips a one-off Whisper auto-transcription.
#REACHY_VEC_VOICE_SAMPLE_TEXT="exact words spoken in the sample"
```

- [ ] **Step 2: Add a "Cloned voice" subsection to README**

Locate the usage/setup section (near where `reachy-vec run` is documented) and add:

```markdown
### Cloned voice (optional)

By default the robot uses the macOS `say` voice. To have it speak in a cloned
voice (fully local, Qwen3-TTS on MLX):

1. Record ~10 seconds of clean speech, e.g.
   `sox -d -r 24000 -c 1 data/voice_sample.wav trim 0 10`
   (or QuickTime → export WAV). Only clone voices with the speaker's consent.
2. In `.env`, set `REACHY_VEC_TTS_BACKEND=qwen-tts` and
   `REACHY_VEC_VOICE_SAMPLE=data/voice_sample.wav`. Optionally set
   `REACHY_VEC_VOICE_SAMPLE_TEXT` to the sample's transcript to skip a
   one-off auto-transcription.
3. `uv run reachy-vec run --preview` — the first run downloads the model
   (~1.5 GB); expect 1–3 s of synthesis per sentence.
```

- [ ] **Step 3: Update stale TTS references in docs**

Run `grep -rn -i "fish-speech\|openvoice" docs/architecture.md docs/pipelines.md README.md CLAUDE.md` and update each hit to name the implemented backends (`say`, `qwen-tts` / Qwen3-TTS 0.6B via mlx-audio) instead of fish-speech/openvoice. Do NOT edit files under `docs/superpowers/specs/` — those are historical records.

- [ ] **Step 4: Add the manual smoke test to `docs/testing.md`**

Append to the manual checklist:

```markdown
- **Cloned voice:** with `REACHY_VEC_TTS_BACKEND=qwen-tts` and a recorded
  `voice_sample`, run `uv run reachy-vec run --preview`, ask a question, and
  confirm (a) the reply is in the cloned voice, (b) per-sentence delay is
  acceptable (~1–3 s), (c) a conversation survives a synthesis hiccup
  (robot skips the sentence rather than crashing).
```

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q && uv run ruff check src tests
git add .env.example README.md docs/testing.md docs/architecture.md docs/pipelines.md CLAUDE.md
git commit -m "docs: voice-clone TTS setup (qwen-tts backend)"
```

---

## Verification (after all tasks)

1. `uv run pytest -q` and `uv run ruff check src tests` — both clean.
2. Manual smoke test from `docs/testing.md` (records real latency on this Mac). If 0.6B quality disappoints, set `REACHY_VEC_TTS_MODEL` to a 1.7B mlx-community variant — no code change needed.

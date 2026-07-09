# Phase 4a — On-robot media Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the perception + speech loop off the Mac's own devices and onto the Reachy Mini's camera, microphone, and speaker, without changing the Oracle state machine or its device-free test suite.

**Architecture:** The Reachy Mini SDK's `MediaManager` (`mini.media`) already exposes `get_frame()`, `get_audio_sample()`, and `push_audio_sample()`. We add three new Protocol implementations that wrap it — a `RobotCamera`, a `RobotAudioSource` behind a new `AudioSource` seam inside `audio/listen.py`, and a robot audio *sink* for `QwenTTSSpeaker` — plus one pure resampling helper (robot audio runs at 44.1/48 kHz; VAD/STT/ECAPA need 16 kHz mono). `run.py` picks Mac vs. robot devices from `MEDIA_SOURCE` / `--source`. Everything stays behind the existing `Camera` / `Speaker` protocols and a new `AudioSource` protocol, each with a test fake.

**Tech Stack:** Python 3.12, `reachy-mini` SDK, `scipy.signal.resample_poly` (already a transitive dependency — do NOT add a new one), numpy, sounddevice, pytest, ruff, typer.

## Global Constraints

- Python **3.12+** (`requires-python = ">=3.12"`).
- **No new third-party dependency.** Resampling uses `scipy` (already present via a transitive dep); confirm with `uv run python -c "import scipy"` before relying on it.
- **Heavy imports stay deferred** inside methods/functions (`sounddevice`, `scipy`, SDK) so `import reachy_vec` and the test suite stay fast — match the existing pattern in `audio/listen.py` and `audio/speak.py`.
- **Every heavy dependency sits behind a Protocol with a test fake.** New fakes go in `tests/conftest.py`. The full suite must run with no devices, models, or network.
- Config knobs go in `src/reachy_vec/config.py` with the `REACHY_VEC_` prefix — never as module constants.
- Audio contract everywhere in the pipeline: **mono float32 at 16 kHz** (`SAMPLE_RATE = 16000` in `audio/listen.py`).
- After changes: `uv run ruff check src tests` and `uv run pytest -q` must both pass (CI gates on both).

---

### Task 1: Resampling helper

**Files:**
- Create: `src/reachy_vec/audio/resample.py`
- Test: `tests/test_resample.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `to_mono_16k(data: np.ndarray, src_rate: int, target_rate: int = 16000) -> np.ndarray` — accepts mono `(n,)` or multi-channel `(n, channels)` float32, returns mono float32 at `target_rate`. Passthrough (same array values) when already mono at `target_rate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resample.py
import numpy as np

from reachy_vec.audio.resample import to_mono_16k


def test_passthrough_when_already_mono_16k():
    data = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = to_mono_16k(data, src_rate=16000)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, data)


def test_downmixes_stereo_to_mono():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)  # (2 frames, 2 ch)
    out = to_mono_16k(stereo, src_rate=16000)
    assert out.ndim == 1
    np.testing.assert_allclose(out, np.array([0.0, 0.5], dtype=np.float32))


def test_resamples_length_by_rate_ratio():
    data = np.zeros(48000, dtype=np.float32)  # 1 s at 48 kHz
    out = to_mono_16k(data, src_rate=48000)
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 1  # ~1 s at 16 kHz


def test_empty_input_returns_empty():
    out = to_mono_16k(np.zeros(0, dtype=np.float32), src_rate=48000)
    assert out.shape == (0,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reachy_vec.audio.resample'`

- [ ] **Step 3: Write the implementation**

```python
# src/reachy_vec/audio/resample.py
"""Turn arbitrary-rate, possibly multi-channel audio into mono float32 16 kHz.

Robot mic/speaker run at the device's native rate (typically 44.1/48 kHz);
silero-VAD, faster-whisper, and ECAPA all expect 16 kHz mono. scipy's
resample_poly is anti-aliased (unlike a raw np.interp), which matters when
downsampling for STT."""

import numpy as np

TARGET_RATE = 16000


def to_mono_16k(data: np.ndarray, src_rate: int, target_rate: int = TARGET_RATE) -> np.ndarray:
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:  # (frames, channels) -> mono
        audio = audio.mean(axis=1)
    if audio.size == 0 or src_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(target_rate))
    up, down = target_rate // g, src_rate // g
    return resample_poly(audio, up, down).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_resample.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/resample.py tests/test_resample.py
git commit -m "feat: to_mono_16k resampling helper for robot audio"
```

---

### Task 2: `AudioSource` seam + `MicSource` (refactor, behavior-preserving)

**Files:**
- Modify: `src/reachy_vec/audio/listen.py` (the `_AudioCapture` class and the `MicTranscriber` / `OpenAITranscriber` / `make_transcriber` constructors)
- Test: `tests/test_listen.py` (add cases; existing cases must keep passing)

**Interfaces:**
- Consumes: `collect_utterance`, `SAMPLE_RATE`, `CHUNK_S` (already in `listen.py`).
- Produces:
  - `class AudioSource(Protocol): def frames(self, chunk_samples: int) -> Iterator[np.ndarray]: ...` — yields mono float32 frames of length `chunk_samples` at 16 kHz, indefinitely.
  - `class MicSource:` with `__init__(self, sample_rate: int = SAMPLE_RATE)` and `frames(self, chunk_samples)` — the current sounddevice `InputStream` loop, extracted verbatim.
  - `_AudioCapture.__init__(self, source: "AudioSource | None" = None, sample_rate: int = SAMPLE_RATE)` — defaults to `MicSource(sample_rate)`, so all current callers behave identically.
  - `MicTranscriber.__init__(..., source=None)` and `OpenAITranscriber.__init__(..., source=None)` thread `source` to `super().__init__`.
  - `make_transcriber(client=None, initial_prompt=None, source=None)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_listen.py
from reachy_vec.audio.listen import MicSource, _AudioCapture  # noqa: E402


class ScriptedSource:
    """AudioSource fake: yields pre-baked frames, then stops (silence -> None)."""

    def __init__(self, frames):
        self._frames = frames

    def frames(self, chunk_samples):
        yield from self._frames


def test_capture_uses_injected_source():
    # three "speech" frames then two "silence" frames; VAD keyed on frame[0]
    speech = np.ones(512, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)
    cap = _AudioCapture(source=ScriptedSource([speech, speech, silence, silence]))
    cap._load_vad = lambda: None  # skip the real model
    cap._vad = None
    # is_speech is built inside _capture from self._vad; patch it via a stub vad:
    cap._vad = lambda frame, rate: type("P", (), {"item": lambda self: float(frame[0])})()
    out = cap._capture(timeout_s=5)
    assert out is not None
    assert len(out) == 512 * 4  # 3 speech + trailing silence run (max_silence=25 not hit -> all)


def test_default_source_is_mic():
    assert isinstance(_AudioCapture()._source, MicSource)
```

Note: `chunk_samples` at 16 kHz is `int(16000 * 0.032) = 512`, matching the frame length above.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_listen.py -v`
Expected: FAIL with `ImportError: cannot import name 'MicSource'`

- [ ] **Step 3: Refactor `listen.py`**

Add the protocol + `MicSource` above `_AudioCapture`:

```python
class AudioSource(Protocol):
    def frames(self, chunk_samples: int) -> Iterator[np.ndarray]:
        """Yield mono float32 frames of length chunk_samples at 16 kHz."""
        ...


class MicSource:
    """Default AudioSource: the Mac's default mic via sounddevice, 16 kHz mono."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._sample_rate = sample_rate

    def frames(self, chunk_samples: int) -> Iterator[np.ndarray]:
        import sounddevice as sd

        with sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype="float32"
        ) as stream:
            while True:
                data, _overflow = stream.read(chunk_samples)
                yield data[:, 0].copy()
```

Replace `_AudioCapture.__init__` and `_capture` with:

```python
    def __init__(self, source: "AudioSource | None" = None, sample_rate: int = SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._source = source or MicSource(sample_rate)
        self._vad = None

    def _capture(self, timeout_s: float) -> np.ndarray | None:
        """Record one VAD-segmented utterance; None if silence until timeout."""
        import torch

        self._load_vad()
        chunk_samples = int(self._sample_rate * CHUNK_S)
        max_chunks = int(timeout_s / CHUNK_S)
        max_silence = int(0.8 / CHUNK_S)  # 0.8 s of quiet ends the utterance

        frame_iter = self._source.frames(chunk_samples)

        def bounded() -> Iterator[np.ndarray]:
            for _ in range(max_chunks):
                try:
                    yield next(frame_iter)
                except StopIteration:
                    return

        def is_speech(chunk: np.ndarray) -> bool:
            prob = self._vad(torch.from_numpy(chunk), self._sample_rate).item()
            return prob > 0.5

        try:
            collected = collect_utterance(bounded(), is_speech, max_silence)
        finally:
            frame_iter.close()
        return np.concatenate(collected) if collected else None
```

Thread `source` through the two transcribers and the factory:

```python
class MicTranscriber(_AudioCapture):
    def __init__(self, model_size=None, sample_rate=SAMPLE_RATE, initial_prompt=None, source=None):
        super().__init__(source=source, sample_rate=sample_rate)
        self._model_size = model_size or settings.stt_model
        self._initial_prompt = initial_prompt
        self._whisper = None
```

```python
class OpenAITranscriber(_AudioCapture):
    def __init__(self, client, initial_prompt=None, source=None):
        super().__init__(source=source)
        self._client = client
        self._initial_prompt = initial_prompt
```

```python
def make_transcriber(client=None, initial_prompt=None, source=None) -> Transcriber:
    if settings.stt_backend == "openai":
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        return OpenAITranscriber(client, initial_prompt=initial_prompt, source=source)
    return MicTranscriber(initial_prompt=initial_prompt, source=source)
```

- [ ] **Step 4: Run the full listen suite to verify old + new pass**

Run: `uv run pytest tests/test_listen.py tests/test_transcriber_factory.py -v`
Expected: PASS (existing tests unchanged in behavior, 2 new pass)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/listen.py tests/test_listen.py
git commit -m "refactor: pluggable AudioSource seam; extract MicSource"
```

---

### Task 3: `RobotAudioSource`

**Files:**
- Create: `src/reachy_vec/audio/sources.py`
- Modify: `tests/conftest.py` (add `FakeMedia`)
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `to_mono_16k` (Task 1); `AudioSource` shape (Task 2); a `media` object with `get_input_audio_samplerate() -> int` and `get_audio_sample() -> np.ndarray | None`.
- Produces:
  - `tests/conftest.py`: `FakeMedia` with `get_frame`, `get_audio_sample`, `get_input_audio_samplerate`, `get_input_channels`, `get_output_audio_samplerate`, `push_audio_sample`, `play_sound`.
  - `class RobotAudioSource:` `__init__(self, media, target_rate: int = SAMPLE_RATE)`, `frames(self, chunk_samples)` — pulls `media.get_audio_sample()`, converts each to mono 16 kHz, buffers, and yields exactly `chunk_samples`-length frames; yields a silence frame when the daemon has no sample yet (keeps VAD cadence).

- [ ] **Step 1: Add `FakeMedia` to `tests/conftest.py`**

```python
class FakeMedia:
    """Stand-in for mini.media: scripted frames + audio samples, records output."""

    def __init__(self, frames=None, samples=None, in_rate=48000, in_channels=1, out_rate=48000):
        self._frames = iter(frames or [])
        self._samples = iter(samples or [])
        self._in_rate, self._in_channels, self._out_rate = in_rate, in_channels, out_rate
        self.pushed: list = []
        self.played: list = []

    def get_frame(self):
        return next(self._frames, None)

    def get_audio_sample(self):
        return next(self._samples, None)

    def get_input_audio_samplerate(self):
        return self._in_rate

    def get_input_channels(self):
        return self._in_channels

    def get_output_audio_samplerate(self):
        return self._out_rate

    def push_audio_sample(self, data):
        self.pushed.append(data)

    def play_sound(self, path):
        self.played.append(path)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_sources.py
import numpy as np

from reachy_vec.audio.sources import RobotAudioSource
from tests.conftest import FakeMedia


def _take(gen, n):
    return [next(gen) for _ in range(n)]


def test_yields_fixed_length_16k_frames_from_48k_samples():
    # one 4800-sample burst at 48 kHz -> 1600 samples at 16 kHz -> ~3 frames of 512
    media = FakeMedia(samples=[np.zeros(4800, dtype=np.float32)], in_rate=48000)
    src = RobotAudioSource(media)
    frames = _take(src.frames(512), 3)
    assert all(f.shape == (512,) for f in frames)
    assert all(f.dtype == np.float32 for f in frames)


def test_yields_silence_frame_when_no_sample_available():
    media = FakeMedia(samples=[], in_rate=16000)  # daemon has nothing yet
    src = RobotAudioSource(media)
    frame = next(src.frames(512))
    assert frame.shape == (512,)
    assert not frame.any()  # all zeros


def test_downmixes_stereo_robot_audio():
    stereo = np.ones((1024, 2), dtype=np.float32)  # 1024 stereo frames at 16 kHz
    media = FakeMedia(samples=[stereo], in_rate=16000, in_channels=2)
    src = RobotAudioSource(media)
    frame = next(src.frames(512))
    assert frame.shape == (512,)
    np.testing.assert_allclose(frame, np.ones(512, dtype=np.float32))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reachy_vec.audio.sources'`

- [ ] **Step 4: Write the implementation**

```python
# src/reachy_vec/audio/sources.py
"""AudioSource backed by the robot's microphone via the Reachy Mini SDK.

mini.media.get_audio_sample() returns variable-length float32 at the device's
native rate; we convert to mono 16 kHz and re-chunk to the fixed frame size the
VAD expects. See audio/listen.py for the AudioSource contract."""

from collections.abc import Iterator

import numpy as np

from reachy_vec.audio.listen import SAMPLE_RATE
from reachy_vec.audio.resample import to_mono_16k


class RobotAudioSource:
    """Pull the robot mic through mini.media; deliver mono 16 kHz frames."""

    def __init__(self, media, target_rate: int = SAMPLE_RATE):
        self._media = media
        self._target_rate = target_rate

    def frames(self, chunk_samples: int) -> Iterator[np.ndarray]:
        src_rate = self._media.get_input_audio_samplerate()
        buf = np.empty(0, dtype=np.float32)
        while True:
            sample = self._media.get_audio_sample()
            if sample is None:
                yield np.zeros(chunk_samples, dtype=np.float32)
                continue
            buf = np.concatenate([buf, to_mono_16k(sample, src_rate, self._target_rate)])
            while len(buf) >= chunk_samples:
                yield buf[:chunk_samples].copy()
                buf = buf[chunk_samples:]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sources.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/sources.py tests/test_sources.py tests/conftest.py
git commit -m "feat: RobotAudioSource — robot mic behind the AudioSource seam"
```

---

### Task 4: `RobotCamera`

**Files:**
- Modify: `src/reachy_vec/perception/camera.py`
- Test: `tests/test_camera.py` (create)

**Interfaces:**
- Consumes: `media.get_frame() -> np.ndarray | None` (BGR).
- Produces: `class RobotCamera:` `__init__(self, media)`, `read(self)` returns `media.get_frame()` — satisfies the existing `Camera` protocol, so `InsightFaceMatcher` and `PreviewSight` consume it unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_camera.py
import numpy as np

from reachy_vec.perception.camera import RobotCamera
from tests.conftest import FakeMedia


def test_robot_camera_reads_frames_from_media():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cam = RobotCamera(FakeMedia(frames=[frame]))
    assert cam.read() is frame
    assert cam.read() is None  # exhausted


def test_robot_camera_passes_through_none():
    cam = RobotCamera(FakeMedia(frames=[]))
    assert cam.read() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_camera.py -v`
Expected: FAIL with `ImportError: cannot import name 'RobotCamera'`

- [ ] **Step 3: Add `RobotCamera` to `camera.py`**

```python
class RobotCamera:
    """Frames from the robot's camera via the Reachy Mini SDK (mini.media)."""

    def __init__(self, media):
        self._media = media

    def read(self):
        return self._media.get_frame()  # BGR ndarray or None, same as WebcamCamera
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_camera.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/perception/camera.py tests/test_camera.py
git commit -m "feat: RobotCamera — robot camera behind the Camera protocol"
```

---

### Task 5: Robot TTS output sink

**Files:**
- Modify: `src/reachy_vec/audio/speak.py`
- Test: `tests/test_speak.py` (add cases)

**Interfaces:**
- Consumes: `to_mono_16k` (Task 1, reused for the reverse direction — resample TTS output *up* to the robot's output rate); `media` with `get_output_audio_samplerate()` and `push_audio_sample()`. `QwenTTSSpeaker` already accepts an injectable `play=(audio, sample_rate) -> None`.
- Produces:
  - `class RobotAudioSink:` `__init__(self, media)`, `__call__(self, audio, sample_rate)` — resamples `audio` from `sample_rate` to `media.get_output_audio_samplerate()` and calls `media.push_audio_sample(...)`.
  - `make_speaker(media=None) -> Speaker` — when `media` is given and backend is `qwen-tts`, construct `QwenTTSSpeaker(..., play=RobotAudioSink(media))`. `say` backend with `media` set logs a warning and returns a normal `SaySpeaker` (say plays on the Mac speaker; on-robot `say` rendering is deferred — see Known limitations).

Note: `to_mono_16k(audio, src_rate=sample_rate, target_rate=out_rate)` resamples in either direction; the name is about the *output* being mono, and TTS audio is already mono, so it's just a rate change.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_speak.py
import numpy as np

from reachy_vec.audio.speak import RobotAudioSink
from tests.conftest import FakeMedia


def test_robot_sink_pushes_resampled_audio_to_media():
    media = FakeMedia(out_rate=16000)  # no resample when rates match
    sink = RobotAudioSink(media)
    audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    sink(audio, sample_rate=16000)
    assert len(media.pushed) == 1
    np.testing.assert_allclose(media.pushed[0], audio)


def test_robot_sink_resamples_to_output_rate():
    media = FakeMedia(out_rate=48000)
    RobotAudioSink(media)(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert abs(len(media.pushed[0]) - 48000) <= 2  # upsampled ~3x


def test_make_speaker_qwen_uses_robot_sink_when_media_given(monkeypatch, tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "qwen-tts")
    monkeypatch.setattr("reachy_vec.audio.speak.settings.voice_sample", sample)
    speaker = make_speaker(media=FakeMedia())
    assert isinstance(speaker._play, RobotAudioSink)


def test_make_speaker_say_with_media_warns_and_stays_local(monkeypatch, caplog):
    import logging

    monkeypatch.setattr("reachy_vec.audio.speak.settings.tts_backend", "say")
    with caplog.at_level(logging.WARNING):
        speaker = make_speaker(media=FakeMedia())
    assert isinstance(speaker, SaySpeaker)
    assert "say" in caplog.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speak.py -v`
Expected: FAIL with `ImportError: cannot import name 'RobotAudioSink'`

- [ ] **Step 3: Implement in `speak.py`**

Add after `_play_blocking`:

```python
class RobotAudioSink:
    """Play TTS audio through the robot's speaker via mini.media.push_audio_sample."""

    def __init__(self, media):
        self._media = media

    def __call__(self, audio, sample_rate: int) -> None:
        import numpy as np

        from reachy_vec.audio.resample import to_mono_16k

        out_rate = self._media.get_output_audio_samplerate()
        data = to_mono_16k(np.asarray(audio, dtype=np.float32), sample_rate, out_rate)
        self._media.push_audio_sample(data)
```

Change `make_speaker` to accept `media`:

```python
def make_speaker(media=None) -> Speaker:
    backend = settings.tts_backend
    if backend == "say":
        if media is not None:
            logger.warning(
                "tts_backend='say' has no on-robot output yet; playing through the "
                "Mac speaker. Use 'qwen-tts' for robot audio."
            )
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
            play=RobotAudioSink(media) if media is not None else None,
        )
    raise NotImplementedError(
        f"TTS backend {backend!r} is not wired - use 'say' or 'qwen-tts'"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speak.py -v`
Expected: PASS (existing + 4 new)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/audio/speak.py tests/test_speak.py
git commit -m "feat: RobotAudioSink — route cloned-voice TTS to the robot speaker"
```

---

### Task 6: Shared robot connection with media acquired

**Files:**
- Modify: `src/reachy_vec/body/robot.py`
- Test: `tests/test_body.py` (add cases)

**Interfaces:**
- Consumes: the `reachy_mini.ReachyMini` SDK (`media_backend`, `acquire_media`, `release_media`, `.media`, `client.disconnect`).
- Produces:
  - `def make_robot(with_media: bool = False) -> tuple[Body, object | None]` — returns `(body, media)`. With media: connect `ReachyMini(media_backend="default")`, call `acquire_media()`, register an atexit that calls `release_media()` then `client.disconnect()`, return `(RobotBody(mini), mini.media)`. Without media: today's behavior, `(RobotBody(mini), None)`. On any failure: `(NullBody(), None)`, logged.
  - `make_body() -> Body` stays, now `return make_robot(with_media=False)[0]` (keeps existing callers/tests green).
  - `make_robot` takes an injectable `connect=` for tests (defaults to the real SDK connect), mirroring how the codebase injects `run`/`generate`/`play` elsewhere.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_body.py
from reachy_vec.body.robot import NullBody, RobotBody, make_robot


class FakeMini:
    def __init__(self):
        self.acquired = self.released = False
        self.media = object()

        class _Client:
            def disconnect(self_):
                pass

        self.client = _Client()

    def acquire_media(self):
        self.acquired = True

    def release_media(self):
        self.released = True


def test_make_robot_with_media_acquires_and_returns_media():
    mini = FakeMini()
    body, media = make_robot(with_media=True, connect=lambda **kw: mini)
    assert isinstance(body, RobotBody)
    assert media is mini.media
    assert mini.acquired is True


def test_make_robot_degrades_to_nullbody_on_connect_failure():
    def boom(**kw):
        raise RuntimeError("no daemon")

    body, media = make_robot(with_media=True, connect=boom)
    assert isinstance(body, NullBody)
    assert media is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_body.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_robot'`

- [ ] **Step 3: Refactor `robot.py`**

Replace `make_body` with `make_robot` + a thin `make_body`:

```python
def make_robot(with_media: bool = False, connect=None) -> tuple[Body, object | None]:
    """Connect to the daemon; optionally acquire camera+mic+speaker media.

    Returns (body, media). `media` is mini.media when with_media and the
    connection succeed, else None. Any failure degrades to (NullBody(), None).
    `connect` is injectable for tests.
    """
    try:
        import atexit

        if connect is None:
            from reachy_mini import ReachyMini

            def connect(**kw):
                return ReachyMini(**kw)

        backend = "default" if with_media else "no_media"
        mini = connect(media_backend=backend)
        if with_media:
            mini.acquire_media()

            def _cleanup():
                try:
                    mini.release_media()
                finally:
                    mini.client.disconnect()

            atexit.register(_cleanup)
            return RobotBody(mini), mini.media
        atexit.register(mini.client.disconnect)
        return RobotBody(mini), None
    except Exception as exc:  # daemon down, robot absent, etc.
        logger.warning("No robot/daemon available (%s); running body-less.", exc)
        return NullBody(), None


def make_body() -> Body:
    """Body only (no media); back-compat wrapper over make_robot."""
    return make_robot(with_media=False)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_body.py -v`
Expected: PASS (existing `make_body` tests + 2 new)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/reachy_vec/body/robot.py tests/test_body.py
git commit -m "feat: make_robot — one connection sharing body + media"
```

---

### Task 7: Media selection, `run.py` wiring, config, docs

**Files:**
- Modify: `src/reachy_vec/config.py` (add `media_source`, `audio_input_rate`)
- Modify: `src/reachy_vec/cli/run.py` (add `--source`, a `resolve_media_source` helper, and the robot-vs-Mac device selection)
- Modify: `docs/configuration.md`, `docs/testing.md`, `docs/architecture.md`, `.env.example`
- Test: `tests/test_cli.py` (add cases for `resolve_media_source`)

**Interfaces:**
- Consumes: `make_robot` (Task 6), `RobotCamera` (Task 4), `RobotAudioSource` (Task 3), `make_transcriber(..., source=)` (Task 2), `make_speaker(media=)` (Task 5).
- Produces:
  - `config.py`: `media_source: str = "auto"`, `audio_input_rate: int = 16000`.
  - `cli/run.py`: `resolve_media_source(requested: str, media_available: bool) -> str` returning `"robot"` or `"mac"` — pure, unit-tested. Rules: `requested == "robot"` → `"robot"` (even if unavailable — caller then hard-errors); `requested == "mac"` → `"mac"`; `requested == "auto"` → `"robot"` if `media_available` else `"mac"`.
  - `cli/run.py`: `run(preview=False, source: str | None = None)` — `--source robot|mac|auto` overrides `settings.media_source`.

- [ ] **Step 1: Add config knobs**

In `src/reachy_vec/config.py`, under `# Perception`:

```python
    media_source: str = "auto"  # auto | robot | mac — where camera/mic/speaker live
    audio_input_rate: int = 16000  # target rate fed to VAD/STT/ECAPA
```

- [ ] **Step 2: Write the failing test for the resolver**

```python
# add to tests/test_cli.py
from reachy_vec.cli.run import resolve_media_source


def test_resolve_auto_prefers_robot_when_available():
    assert resolve_media_source("auto", media_available=True) == "robot"


def test_resolve_auto_falls_back_to_mac():
    assert resolve_media_source("auto", media_available=False) == "mac"


def test_resolve_explicit_robot_is_honored_even_if_unavailable():
    assert resolve_media_source("robot", media_available=False) == "robot"


def test_resolve_explicit_mac():
    assert resolve_media_source("mac", media_available=True) == "mac"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k resolve -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_media_source'`

- [ ] **Step 4: Add the resolver + wiring to `run.py`**

Add the pure helper near the top of `cli/run.py`:

```python
def resolve_media_source(requested: str, media_available: bool) -> str:
    """Pick 'robot' or 'mac'. 'auto' -> robot when media is available, else mac."""
    if requested in ("robot", "mac"):
        return requested
    return "robot" if media_available else "mac"
```

In `run()`, add the option and replace the device-construction block. Connect the robot first (it decides media availability), then build camera / transcriber source / speaker accordingly:

```python
def run(
    preview: bool = typer.Option(False, "--preview", help="Show the webcam feed + face matches."),
    source: str = typer.Option(
        None, "--source", help="Media source: auto | robot | mac (default: REACHY_VEC_MEDIA_SOURCE)."
    ),
) -> None:
    ...
    from reachy_vec.audio.sources import RobotAudioSource
    from reachy_vec.body.robot import make_robot
    from reachy_vec.perception.camera import RobotCamera, WebcamCamera

    requested = source or settings.media_source
    # Acquire robot media only if the user might want it (auto/robot).
    want_media = requested in ("auto", "robot")
    body, media = make_robot(with_media=want_media)
    chosen = resolve_media_source(requested, media_available=media is not None)
    if chosen == "robot" and media is None:
        typer.echo("--source robot but no robot media available (is the daemon up "
                   "with media?).", err=True)
        raise typer.Exit(code=1)

    if chosen == "robot":
        camera = RobotCamera(media)
        audio_source = RobotAudioSource(media, target_rate=settings.audio_input_rate)
    else:
        camera = WebcamCamera(settings.camera_index)
        audio_source = None  # MicSource default
    if camera.read() is None:
        typer.echo(f"No camera frame from '{chosen}' source - check the device.", err=True)
        raise typer.Exit(code=1)

    ...
    transcriber = make_transcriber(client=client, initial_prompt=vocab_prompt, source=audio_source)
    ...
    speaker = make_speaker(media=media if chosen == "robot" else None)
    ...
    loop = OracleLoop(..., body=body, ...)  # use the already-connected body, not make_body()
```

Remove the old standalone `camera = WebcamCamera(...)`, `speaker = make_speaker()`, and `body=make_body()` lines — they're replaced above.

- [ ] **Step 5: Run the resolver tests + full CLI suite**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing + 4 new)

- [ ] **Step 6: Update docs**

- `docs/configuration.md` → **Perception** table: add `MEDIA_SOURCE` (`auto`) and `AUDIO_INPUT_RATE` (`16000`) rows; update the `ROBOT_HOST` note to say media is now wired via `--source`/`MEDIA_SOURCE` (host wiring itself is still Phase 4b).
- `docs/architecture.md` → in "Known gaps", mark gap #1 (on-robot media) as **done in Phase 4a** and note `say` on-robot output + `ROBOT_HOST` remain open (4b).
- `docs/testing.md` → add smoke rows under section 4: `uv run reachy-vec run --source robot` → robot's camera/mic/speaker drive the loop (needs the daemon running with media, `qwen-tts` backend for robot audio); `--source mac` reproduces today's desk behavior.
- `.env.example` → add commented `REACHY_VEC_MEDIA_SOURCE=auto`.

- [ ] **Step 7: Full suite + lint, then commit**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass

```bash
git add -A
git commit -m "feat: run --source selects robot vs Mac camera/mic/speaker"
```

---

### Task 8 (optional stretch): Orient to the speaker

**Files:**
- Modify: `src/reachy_vec/brain/oracle.py` (on greet, if a face bbox + robot media are present)
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: the insightface bbox already produced by `perception/face.py` (verify it's surfaced on the observation; if not, thread the bbox centre through), plus `mini.look_at_image(u, v)`.
- Produces: the head turns toward the greeted face. Ship only if the bbox is already available on the face observation; otherwise defer to its own plan (do not expand face.py's contract inside 4a).

- [ ] **Step 1:** Inspect the face observation returned by `InsightFaceMatcher.observe` for a bbox/centre. If absent, STOP — write a one-line note in the 4a completion summary that this needs its own task, and skip to committing 4a. If present, continue with a TDD cycle: fake body records a `look_at(u, v)` call on greet; wire `mini.look_at_image` behind a small `Body.look_at` no-op (NullBody ignores it). Commit separately.

---

## Self-Review

**Spec coverage** (against the 4a section of the Phase 4 spec):
- "Use `media_backend="default"`, acquire/release media" → Task 6. ✓
- `RobotCamera` wrapping `get_frame()` → Task 4. ✓
- `AudioSource` seam + `MicSource` + `RobotAudioSource` on `get_audio_sample()` → Tasks 2, 3. ✓
- Route `QwenTTSSpeaker` to `push_audio_sample()` → Task 5. ✓ (`say` on-robot rendering explicitly deferred — see below.)
- Explicit resampling to 16 kHz mono in / output rate out → Task 1, reused in Tasks 3 and 5. ✓
- `--source robot|mac` + `MEDIA_SOURCE` / `AUDIO_INPUT_RATE` config → Task 7. ✓
- Stretch "orient to the speaker" via `look_at_image` → Task 8 (guarded on the bbox already existing). ✓

**Deviations from spec (intentional, right-sizing):**
- `say`-on-robot rendering (`say -o` → `play_sound`) is **deferred within 4a**. Rationale: the spec names `qwen-tts` the recommended on-robot backend; the array path is clean and fully testable, while `say -o` file rendering adds a temp-file + format-negotiation branch with little value for the milestone. Task 5 makes `say + media` warn and stay Mac-local. Record this in the 4a summary; it is a small follow-up, not a gap in the milestone.

**Placeholder scan:** none — every code step shows complete code; every test step shows the assertions; every run step shows the command and expected result.

**Type consistency:** `to_mono_16k(data, src_rate, target_rate=16000)` used identically in Tasks 1/3/5. `AudioSource.frames(chunk_samples)` defined in Task 2, implemented by `RobotAudioSource` in Task 3, consumed by `_AudioCapture._capture` in Task 2. `make_robot(with_media, connect) -> (Body, media|None)` defined in Task 6, consumed in Task 7. `make_speaker(media=None)` and `make_transcriber(..., source=None)` signatures match between definition (Tasks 5, 2) and call sites (Task 7). `FakeMedia` (Task 3) exposes exactly the methods Tasks 3/4/5 call.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-phase4a-on-robot-media.md`.

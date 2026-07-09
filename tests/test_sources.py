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

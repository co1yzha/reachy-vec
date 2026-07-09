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

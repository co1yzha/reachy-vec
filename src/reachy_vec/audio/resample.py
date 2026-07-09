"""Turn arbitrary-rate, possibly multi-channel audio into mono float32 16 kHz.

Robot mic/speaker run at the device's native rate (typically 44.1/48 kHz);
silero-VAD, faster-whisper, and ECAPA all expect 16 kHz mono. scipy's
resample_poly is anti-aliased (unlike a raw np.interp), which matters when
downsampling for STT.
"""

import numpy as np

TARGET_RATE = 16000


def to_mono_16k(
    data: np.ndarray, src_rate: int, target_rate: int = TARGET_RATE
) -> np.ndarray:
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

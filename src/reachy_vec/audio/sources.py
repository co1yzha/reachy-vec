"""AudioSource backed by the robot's microphone via the Reachy Mini SDK.

mini.media.get_audio_sample() returns variable-length float32 at the device's
native rate; we convert to mono 16 kHz and re-chunk to the fixed frame size the
VAD expects. See audio/listen.py for the AudioSource contract.
"""

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
        # The daemon buffers mic audio while nobody reads it; whatever is
        # queued when a listener attaches is stale — typically the robot's
        # own speech, which would replay to the VAD as a phantom utterance.
        # Reads out-pace real-time capture, so this catches up quickly.
        for _ in range(1000):
            if self._media.get_audio_sample() is None:
                break
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

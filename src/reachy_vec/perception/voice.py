"""Speaker identification: ECAPA voice embeddings matched against the voices
table. Mirrors face.py: never guesses - below-threshold = unknown speaker,
borderline (within the shared margin under threshold) or too-short audio =
None ("can't tell", fusion falls back to face).
"""

import logging
from typing import Protocol

import numpy as np

from reachy_vec.config import settings
from reachy_vec.perception.face import BORDERLINE_MARGIN, Observation
from reachy_vec.store.db import Store

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpeakerIdentifier(Protocol):
    def identify(self, audio: np.ndarray | None) -> Observation | None: ...
    def embed(self, audio: np.ndarray | None) -> list[float] | None: ...


class EcapaSpeakerIdentifier:
    """speechbrain ECAPA-TDNN (spkrec-ecapa-voxceleb), 192-dim, lazy-loaded.

    A model-load failure degrades to "can't tell" permanently (logged once) -
    the robot keeps working face-only.
    """

    def __init__(
        self,
        store: Store,
        threshold: float | None = None,
        min_utterance_s: float | None = None,
    ):
        self._store = store
        self._threshold = threshold if threshold is not None else settings.voice_threshold
        self._min_samples = int(
            SAMPLE_RATE
            * (
                min_utterance_s
                if min_utterance_s is not None
                else settings.voice_min_utterance_s
            )
        )
        self._model = None
        self._broken = False

    def _load_model(self):
        from speechbrain.inference.speaker import EncoderClassifier

        return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def _load(self):
        if self._model is None and not self._broken:
            try:
                self._model = self._load_model()
            except Exception:
                logger.exception("ECAPA load failed - voice ID disabled")
                self._broken = True

    def embed(self, audio: np.ndarray | None) -> list[float] | None:
        if audio is None or len(audio) < self._min_samples:
            return None
        self._load()
        if self._model is None:
            return None
        import torch

        with torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(audio).unsqueeze(0))
        return emb.squeeze().tolist()

    def identify(self, audio: np.ndarray | None) -> Observation | None:
        vector = self.embed(audio)
        if vector is None:
            return None
        match = self._store.match_voice(vector)
        if match is None:
            return None  # nobody has a voice profile yet
        person_id, name, score = match
        if score >= self._threshold:
            return Observation(person_id=person_id, name=name, score=score)
        if score >= self._threshold - BORDERLINE_MARGIN:
            return None
        return Observation(person_id=None, name=None, score=score)

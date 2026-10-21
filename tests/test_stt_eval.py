import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "stt_eval", Path(__file__).resolve().parent.parent / "scripts" / "stt_eval.py"
)
stt_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stt_eval)


def test_wer_counts_substitutions_insertions_deletions():
    assert stt_eval.wer("what is the latest demo", "what is the latest demo") == 0.0
    assert stt_eval.wer("what is the latest demo", "what is a latest demo") == 0.2   # sub
    assert stt_eval.wer("tell Hannah", "tell Hannah now") == 0.5                   # insert
    assert stt_eval.wer("tell Hannah now", "tell now") == 1 / 3                     # delete
    assert stt_eval.wer("Hello, World!", "hello world") == 0.0                      # punctuation


def test_mix_at_snr_hits_the_target_ratio():
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 0.1, 16000).astype(np.float32)
    noise = rng.normal(0, 0.3, 8000).astype(np.float32)        # shorter: gets looped
    mixed = stt_eval.mix_at_snr(speech, noise, 10.0)
    added = mixed - speech
    snr = 10 * np.log10(np.mean(speech**2) / np.mean(added**2))
    assert abs(snr - 10.0) < 0.1 and mixed.shape == speech.shape


def test_mix_clean_is_unchanged():
    speech = np.ones(10, dtype=np.float32) * 0.1
    assert stt_eval.mix_at_snr(speech, np.ones(5, dtype=np.float32), None) is speech

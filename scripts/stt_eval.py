"""Compare speech-to-text backends on synthetic clips, with background chatter.

    uv run --no-sync python scripts/stt_eval.py [--backends local,parakeet]

Clips are spoken by several macOS `say` voices (varied accents); "babble" is
other sentences in other voices mixed underneath at a given signal-to-noise
ratio, standing in for people talking in the background. Per backend and noise
level it reports word error rate, wake-word hits (clips that start with
"Reachy"), false wakes (clips that don't), name accuracy, and seconds per clip.
Synthetic speech is cleaner than a real room - treat the numbers as a ranking,
then confirm on real recordings.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
VOICES = ["Daniel", "Samantha", "Karen", "Moira", "Rishi", "Tessa", "Aman", "Tara"]
NAMES = ["Yang", "David", "Hannah", "Shaun"]

# (text, starts with the wake word?)
CLIPS = [
    ("Reachy, what is the latest demo in the hub?", True),
    ("Hey Reachy, tell Hannah the review moved to three.", True),
    ("Reachy, any demos about food mapping?", True),
    ("Reachy, remind David about the meeting tomorrow.", True),
    ("Reachy, what's the weather like outside?", True),
    ("Reachy, send a message to Shaun saying the build passed.", True),
    ("Reachy, how are you today?", True),
    ("Reachy, ask Yang if the dashboard is ready.", True),
    ("I was reaching for the coffee when the call started.", False),
    ("We should be ready to present the demo by Friday.", False),
    ("Can you pass me that cable over there?", False),
    ("The satellite data came in late again this week.", False),
]
BABBLE = [
    "So anyway the meeting ran over and nobody had lunch.",
    "I think the train was delayed by about twenty minutes.",
    "Have you seen the new coffee machine on the second floor?",
    "We need to sort out the budget before the end of the month.",
]
SNRS_DB = [None, 10.0, 5.0]  # None = clean


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level edit distance / reference length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return float(bool(hyp))
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / len(ref)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float | None) -> np.ndarray:
    """Speech plus noise scaled to snr_db (noise looped/trimmed to fit)."""
    if snr_db is None:
        return speech
    noise = np.resize(noise, speech.shape)
    p_speech = float(np.mean(speech**2)) or 1e-12
    p_noise = float(np.mean(noise**2)) or 1e-12
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = speech + gain * noise
    peak = float(np.max(np.abs(mixed))) or 1.0
    return (mixed / peak * 0.9).astype(np.float32) if peak > 0.99 else mixed.astype(np.float32)


def say(text: str, voice: str, workdir: Path) -> np.ndarray:
    import soundfile as sf

    path = workdir / f"{abs(hash((text, voice)))}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(path), f"--data-format=LEF32@{SAMPLE_RATE}",
                    text], check=True)
    audio, _ = sf.read(path, dtype="float32")
    return np.pad(audio, (SAMPLE_RATE // 4, SAMPLE_RATE // 4))  # a little lead-in/out


def make_backend(name: str):
    """(transcribe(audio) -> text, warm-up seconds) for one backend."""
    t0 = time.perf_counter()
    if name == "parakeet":
        from reachy_vec.audio.listen import ParakeetTranscriber

        stt = ParakeetTranscriber()
        stt.warm_up()
        fn = stt._transcribe
    elif name == "local":
        from faster_whisper import WhisperModel

        from reachy_vec.config import settings

        model = WhisperModel(settings.stt_model, compute_type="int8")

        def fn(audio):
            segments, _ = model.transcribe(audio, language="en",
                                           hotwords=settings.stt_hotwords or None)
            return " ".join(s.text.strip() for s in segments)

        fn(np.zeros(SAMPLE_RATE, dtype=np.float32))
    else:
        raise SystemExit(f"unknown backend {name!r} (local | parakeet)")
    return fn, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", default="local,parakeet")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from reachy_vec.audio.listen import normalize_rms
    from reachy_vec.brain.oracle import split_wake
    from reachy_vec.config import settings

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        print("synthesizing clips...")
        clips = [(text, wake, voice, say(text, voice, workdir))
                 for text, wake in CLIPS for voice in VOICES[:4]]
        babble = np.concatenate(
            [say(t, v, workdir) for t, v in zip(BABBLE, VOICES[4:], strict=True)]
        )
        for backend in args.backends.split(","):
            fn, warm = make_backend(backend)
            print(f"\n== {backend}  (load + warm-up {warm:.1f}s)")
            print(f"{'noise':>8} {'WER':>6} {'wake hits':>10} {'false wakes':>12} "
                  f"{'names':>7} {'s/clip':>7}")
            for snr in SNRS_DB:
                errs, hits, wakes, false, negs, names_ok, names_n, secs = [], 0, 0, 0, 0, 0, 0, []
                for text, wake, _voice, audio in clips:
                    noisy = mix_at_snr(audio, babble, snr)
                    t0 = time.perf_counter()
                    heard = fn(normalize_rms(noisy, settings.stt_target_rms))
                    secs.append(time.perf_counter() - t0)
                    rest = split_wake(heard, settings.wake_words)
                    # score the words after the wake word against the words after "Reachy"
                    ref = split_wake(text, settings.wake_words) if wake else text
                    errs.append(wer(ref or text, rest if (wake and rest is not None) else heard))
                    if wake:
                        wakes += 1
                        hits += rest is not None
                    else:
                        negs += 1
                        false += rest is not None
                    for name in NAMES:
                        if name in text:
                            names_n += 1
                            names_ok += name.lower() in heard.lower()
                label = "clean" if snr is None else f"{snr:.0f} dB"
                print(f"{label:>8} {np.mean(errs):6.1%} {hits:>4}/{wakes:<5} {false:>6}/{negs:<5} "
                      f"{names_ok:>3}/{names_n:<3} {np.median(secs):7.2f}")


if __name__ == "__main__":
    main()

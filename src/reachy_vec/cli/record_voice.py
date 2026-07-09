"""Record a reference voice sample for the qwen-tts clone backend."""

import time
from pathlib import Path

import typer

from reachy_vec.audio.record import SAMPLE_RATE, record_sample, write_wav
from reachy_vec.config import settings

DURATION_S = 10.0

# A phonetically varied line; because we know it, we can emit the transcript
# directly (VOICE_SAMPLE_TEXT) and skip a Whisper pass at synthesis time.
PROMPT_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while five wizards "
    "vex the jolly queen on a bright summer morning."
)


def record_voice(
    out: str = typer.Option(
        "data/voice_sample.wav", help="Where to save the recorded WAV."
    ),
) -> None:
    """Record a ~10 s voice sample to clone with the qwen-tts backend."""
    out = Path(out)
    typer.echo("Read this aloud clearly, at a natural pace:\n")
    typer.echo(f"    {PROMPT_SENTENCE}\n")
    for n in (3, 2, 1):
        typer.echo(f"Recording in {n}...")
        time.sleep(1.0)
    typer.echo(f"Recording for {DURATION_S:.0f}s - go!")

    audio = record_sample(DURATION_S, SAMPLE_RATE)
    if audio is None or not len(audio):
        typer.echo("No audio captured - check the mic and retry.", err=True)
        raise typer.Exit(code=1)

    write_wav(out, audio, SAMPLE_RATE)
    typer.echo(f"\nSaved {DURATION_S:.0f}s sample to {out}\n")
    typer.echo("Add these lines to your .env to speak in this voice:\n")
    typer.echo("REACHY_VEC_TTS_BACKEND=qwen-tts")
    typer.echo(f"REACHY_VEC_VOICE_SAMPLE={out}")
    typer.echo(f'REACHY_VEC_VOICE_SAMPLE_TEXT="{PROMPT_SENTENCE}"')

    # Nudge: if a sample path is already configured elsewhere, say so.
    if settings.voice_sample and Path(settings.voice_sample) != out:
        typer.echo(
            f"\nNote: REACHY_VEC_VOICE_SAMPLE is currently {settings.voice_sample}; "
            "update it to the path above (or re-run with --out to match).",
        )

# Cloning your voice

Give the robot your voice. By default it speaks with the macOS `say` voice;
with a ~10-second recording of you, the `qwen-tts` backend makes it speak any
answer in your own voice instead. Everything runs locally on the Mac — the
recording and the model never leave the machine.

This takes about five minutes. You do it once; the robot reuses the sample
from then on.

## What you need

- The `customise-speech-qwen3-tts` branch checked out (that's where the
  `record-voice` command lives until it merges to `main`).
- A working microphone and a quiet room.
- Internet for the first run only — the voice model (~1.5 GB) downloads once
  and is cached afterwards.

## Step 1 — Record your sample

```bash
uv run reachy-vec record-voice
```

The command prints a sentence, counts down `3… 2… 1…`, then records for ten
seconds. Read the sentence aloud at a natural, relaxed pace — the way you'd
actually talk to the robot, not a news-anchor voice. When it finishes it
saves `data/voice_sample.wav` and prints three lines for your `.env`.

The first time, macOS will ask for microphone permission — allow it, then
re-run the command.

## Step 2 — Point the robot at your voice

Open `.env` (copy it from `.env.example` if you don't have one yet) and paste
the three lines the command printed. They look like this:

```bash
REACHY_VEC_TTS_BACKEND=qwen-tts
REACHY_VEC_VOICE_SAMPLE=data/voice_sample.wav
REACHY_VEC_VOICE_SAMPLE_TEXT="The quick brown fox jumps over the lazy dog while five wizards vex the jolly queen on a bright summer morning."
```

`REACHY_VEC_VOICE_SAMPLE_TEXT` must match the words you actually recorded —
that's how the model separates *how you sound* from *what you said*. If you
used `record-voice`, the printed sentence already matches, so just paste it.
(You can also delete that line entirely; the model will transcribe the sample
itself on first use, just a little slower.)

## Step 3 — Hear it

```bash
uv run reachy-vec chat
```

Type anything and the reply comes back in your cloned voice. `chat` is the
quickest check — no camera or robot needed. The **first** reply pauses while
the model downloads and warms up; after that, expect roughly 1–3 seconds per
sentence.

When you're happy, run the full experience:

```bash
uv run reachy-vec run --preview
```

## Getting a good clone

The recording matters far more than any setting. For a natural result:

- **Quiet room, no background noise.** Fans, music, and echo all leak into
  the clone.
- **Speak naturally.** Match the tone you want the robot to use. A stiff,
  over-enunciated sample produces a stiff robot.
- **Consistent volume**, mouth a steady distance from the mic.
- **Re-record freely.** `record-voice` overwrites the file each time — the
  fastest fix for a disappointing clone is a cleaner take.

## If the voice isn't good enough

Try a larger, higher-fidelity model by adding one line to `.env` (it's slower
per sentence but sounds better):

```bash
REACHY_VEC_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
```

## Going back to the default voice

Comment out or remove `REACHY_VEC_TTS_BACKEND=qwen-tts` (or set it back to
`say`). The robot returns to the macOS voice immediately; your sample stays on
disk for next time.

## A note on privacy and consent

The sample and the model stay entirely on the Mac — nothing is uploaded. Clone
only your own voice, or someone else's with their explicit permission. The WAV
under `data/` is git-ignored and never committed.

## Current limitations

- **Playback is through the Mac speaker, not the robot.** Synthesis runs
  locally and plays on the Mac; on-robot audio output isn't wired yet (see
  [architecture.md → Known gaps](architecture.md#known-gaps--toward-a-real-robot-deploy)).
- **You can't interrupt it.** Barge-in (talking over a reply) is specced but
  not implemented, so a long cloned-voice answer plays to the end.

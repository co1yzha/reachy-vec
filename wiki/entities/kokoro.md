---
title: Kokoro (Kokoro-82M)
created: 2026-10-22
updated: 2026-10-22
type: entity
tags: [tts]
sources: [raw/docs/pipelines.md, raw/docs/configuration.md]
confidence: high
---

Kokoro-82M is a preset-voice text-to-speech model (no cloning) running via mlx-audio on the local Mac. It is the current active TTS backend for reachy-vec, selected by `REACHY_VEC_TTS_BACKEND=kokoro`.

## Key facts

- **Model:** Kokoro-82M (82 million parameters)
- **Speed:** ~25× faster than real time
- **Voice cloning:** No — uses preset voices selected by `REACHY_VEC_KOKORO_VOICE`
- **Accent prefixes:** `a…` = US English, `b…` = British English
- **Install:** `uv pip install --python .venv/bin/python "misaki[en]"` (provides the text-to-phoneme step)
- **Config:** `REACHY_VEC_TTS_BACKEND=kokoro`, `REACHY_VEC_KOKORO_VOICE=bf_alice`, `REACHY_VEC_TTS_SPEED=1`

## Preset voices (selected by `REACHY_VEC_KOKORO_VOICE`)

| Voice | Accent | Description |
|---|---|---|
| `af_heart` | US female | Warm, expressive, very natural-sounding |
| `af_emma` | US female | Slightly warmer, conversational |
| `am_adam` | US male | Deep, clear, good for narration |
| `am_michael` | US male | Deeper, slightly gravelly, characterful |
| `am_puck` | US male | Playful, slightly higher-pitched |
| `bf_alice` | UK female | Crisp, articulate, BBC-presenter clarity |
| `bf_emma` | UK female | Warm, conversational, real-person feel |
| `bf_sarah` | UK female | Polished, clear |
| `bf_thomas` | UK male | Mid-range, natural, pleasant |
| `bm_george` | UK male | British male (configured in .env) |

## Where it runs

Local on the Mac via mlx-audio. When `--source robot` is used, audio plays through the robot's speaker (unlike `say` which plays on the Mac).

## Known issues

- mlx-audio 0.4.4 has a length bug that crashes ~1 in 6 sentences; Reachy patches this internally.
- `TTS_SPEED` acts as a real tempo change (pitch is kept, unlike the clone backend).

## Related

- [[tts-backends]] — comparison of all TTS backends
- [[qwen3-tts]] — the alternative voice-clone backend
- [[barge-in]] — interruption behaviour (on-robot stop is best-effort)
- [[media-source]] — `--source robot` needed for on-robot playback

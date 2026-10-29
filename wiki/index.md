# Wiki Index

> Content catalog. Every wiki page listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: 2026-10-22 | Total pages: 24

## Entities

- [[bge-small-en]] — `BAAI/bge-small-en-v1.5` (384-dim, normalized; local via sentence-transformers)
- [[ecapa-tdnn]] — speechbrain ECAPA-TDNN `spkrec-ecapa-voxceleb` (192-dim, local) in
- [[faster-whisper]] — faster-whisper `base.en` (int8, English-only, local) is the **default** STT
- [[gpt-5-mini]] — `gpt-5-mini` (OpenAI API, cloud) is the reasoning engine. It does the
- [[insightface]] — insightface `buffalo_s` (512-dim, local via onnxruntime) in `perception/face.py`
- [[lancedb-store]] — A single **embedded LanceDB** database at `data/lancedb` holds all vectors.
- [[mac-brain]] — The **brain runs on a Mac (Apple silicon)**; the robot is a thin body. All
- [[qwen3-tts]] — Qwen3-TTS 0.6B (via mlx-audio, fully local) is the **voice-clone** TTS
- [[reachy-mini]] — The Reachy Mini is the robot "body". It is **wireless**: the USB-C port is
- [[silero-vad]] — silero-vad (16 kHz, 512-sample frames, local) in `audio/listen.py:_AudioCapture`
- [[kokoro]] — Kokoro-82M (via mlx-audio, preset voices, no cloning, ~25x faster than real time) is the **preset** TTS backend. Voices prefixed `a…` = US, `b…` = UK — e.g. `af_heart`, `bf_alice`, `bf_emma`, `am_michael`, `bm_george`. Needs `misaki[en]`.

## Concepts

- [[configuration]] — Layered settings: env vars > repo .env > Reachy profile .env; robot knobs and DATA_DIR.
- [[barge-in]] — Phase 4c. Talking over a reply interrupts it, gated by `BARGE_IN` (on by
- [[tools]] — All in `brain/chat.py`. Tool calls cost one extra LLM round-trip (sentence
- [[identity-rule]] — The design-spec rule: **never guess.** Every utterance is voice-identified
- [[media-source]] — `MEDIA_SOURCE` / `reachy-vec run --source` selects whose camera/mic/speaker the
- [[memory-pipeline]] — Phase 2. Memories are per-person notes in the `memories` table.
- [[messenger-pipeline]] — Phase 3. Lets the robot relay a message to an absent teammate.
- [[oracle-loop]] — `brain/oracle.py:OracleLoop` is a **synchronous state machine** — all
- [[protocol-pattern]] — The core pattern of reachy-vec: **every heavy dependency sits behind a small
- [[rag-pipeline]] — `brain/chat.py:ChatBrain.respond` — one turn:
- [[roadmap-phases]] — The project is built in phases (design specs in `docs/superpowers/specs/`):

## Comparisons

- [[stt-backends]] — Three STT backends, selected by `REACHY_VEC_STT_BACKEND`:
- [[tts-backends]] — Three TTS backends, selected by `REACHY_VEC_TTS_BACKEND`:

## Queries

_No filed queries yet — substantial answers get filed here._

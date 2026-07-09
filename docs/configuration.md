# Configuration reference

All settings live in `src/reachy_vec/config.py` (`pydantic-settings`).
Set them as environment variables with the `REACHY_VEC_` prefix or in
`.env` at the repo root (e.g. `REACHY_VEC_LLM_MODEL=gpt-4o`).
Unknown keys in `.env` are ignored.

## Secrets (no prefix)

| Variable | Used by | Needed for |
|---|---|---|
| `OPENAI_API_KEY` | openai SDK directly | `chat`, `run` (LLM), `STT_BACKEND=openai` |
| `MONGODB_URI` | `cli/sync.py` | `sync-mongo` only |

## Models

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `gpt-5-mini` | OpenAI chat model for answers, tools, and memory distillation |
| `LLM_REASONING_EFFORT` | `minimal` | Reasoning effort for gpt-5* models (`minimal`/`low`/`medium`/`high`); ignored for others |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | must stay 384-dim (`EMBEDDING_DIM`); changing models requires re-ingesting everything (docs and memories share the space) |
| `EMBEDDING_QUERY_PREFIX` | `Represent this sentence for searching relevant passages: ` | BGE query instruction, applied to search queries only (never documents); set empty when using a non-BGE embedding model |
| `STT_BACKEND` | `local` | `local` (faster-whisper) or `openai` (`gpt-4o-transcribe`; more accurate, ~1 s slower) |
| `STT_MODEL` | `base.en` | faster-whisper size for the local backend; `small.en` = more accurate, slower |
| `TTS_BACKEND` | `say` | `say` (macOS built-in) or `qwen-tts` (voice clone, local MLX) |
| `TTS_MODEL` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | mlx-audio model id for `qwen-tts`; swap for a 1.7B variant if clone quality disappoints |
| `VOICE_SAMPLE` | unset | ~10 s clean WAV of the voice to clone; required for `qwen-tts` |
| `VOICE_SAMPLE_TEXT` | unset | transcript of the sample; omit → auto-transcribed once via Whisper on first synthesis |

## Perception

| Setting | Default | Notes |
|---|---|---|
| `FACE_THRESHOLD` | `0.45` | cosine gate for recognition; lower (e.g. 0.40) if known people show as unknown; within 0.05 below = ignored as "no face" |
| `CAMERA_INDEX` | `0` | which webcam OpenCV opens |

## Voice ID (Phase 2b)

| Setting | Default | Notes |
|---|---|---|
| `VOICE_THRESHOLD` | `0.30` | cosine gate for speaker recognition (ECAPA scores run lower than face scores); within 0.05 below = "can't tell", fusion falls back to face |
| `VOICE_MIN_UTTERANCE_S` | `1.0` | utterances shorter than this aren't voice-identified |
| `VOICE_PASSIVE_CAP` | `10` | max passively-banked voice embeddings kept per person (oldest evicted; explicit enrollment rows never pruned) |

## Interaction pacing

| Setting | Default | Notes |
|---|---|---|
| `GREET_COOLDOWN_S` | `7200` | spoken "Hi <name>!" at most every 2 h per person; within cooldown = silent acknowledge |
| `SILENCE_TIMEOUT_S` | `30` | quiet time that ends a conversation (triggers memory distillation) |
| `IDLE_SLEEP_S` | `300` | no faces for this long → sleep motion; wakes on the next face |

## Environment

| Setting | Default | Notes |
|---|---|---|
| `ROBOT_HOST` | unset | **reserved, not yet consumed.** `body/robot.py:make_body` currently always connects to the local daemon (`ReachyMini(media_backend="no_media")`) and falls back to a logging `NullBody` if none is reachable. Point a remote robot at the brain by running the daemon against it; this knob will wire an explicit address later |
| `DATA_DIR` | `data` | holds `lancedb/` (all tables), `faces/` (enrollment JPEGs), `reachy.log` (transcript log — privacy-relevant, gitignored) |
| `WEATHER_LAT` / `WEATHER_LON` | `53.4084` / `-2.9916` | lab location for `get_weather` (Liverpool, UK; Open-Meteo, no key) |

## Not configurable (code constants)

Chunk size (1000 chars, `store/ingestion.py`), retrieval k (5 docs / 3
memories, `brain/chat.py`), history window (20 messages), memory-duplicate
threshold (0.97), VAD sensitivity (0.5) and end-of-utterance quiet (0.8 s,
`audio/listen.py`), borderline margin shared by faces and voices (0.05,
`perception/face.py`), unknown-face stable polls (3), insightface model
(`buffalo_s`), ECAPA model (`spkrec-ecapa-voxceleb`, `perception/voice.py`).
Promote one to `config.py` if you find yourself tuning it.

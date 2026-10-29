# Configuration reference

All settings live in `src/reachy_vec/config.py` (`pydantic-settings`).
Set them as environment variables with the `REACHY_VEC_` prefix. Unknown
keys are ignored.

## Where values come from (highest priority first)

1. **Real environment variables** exported by the shell/CI
2. **`.env` at the repo root** — per-machine overrides (gitignored)
3. **The Reachy profile's `.env`** — `~/.hermes/profiles/reachy/.env` —
   the deployment's shared home for secrets and defaults (loaded only if it
   exists). Put `OPENAI_API_KEY`, `MONGODB_URI`, `REACHY_VEC_ROBOT_HOST` etc.
   there once and both the agent and the robot read them; keep the repo
   `.env` for machine-specific overrides only.

`env_file` layering is verified by `tests/test_config.py::test_env_file_layering`.

## Secrets (no prefix)

| Variable | Used by | Needed for |
|---|---|---|
| `OPENAI_API_KEY` | openai SDK directly | `chat`, `run` (LLM when `LLM_BASE_URL` is unset), `STT_BACKEND=openai` |
| `MONGODB_URI` | `cli/sync.py` | `sync-mongo` only |
| `TAVILY_API_KEY` | `brain/chat.py` (`web_search` tool) | only when `WEB_SEARCH=true`; free tier is limited — the tool tells the user when credits run out (Tavily 432/433) |

## Models

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `gpt-5-mini` | chat model for answers, tools, and memory distillation; for the local server use `Qwen3.6-35B-A3B-4bit` |
| `LLM_BASE_URL` | unset | OpenAI-compatible server; `http://127.0.0.1:8001/v1` = local oMLX. Unset = OpenAI cloud |
| `LLM_API_KEY` | unset | key for that server (`REACHY_VEC_LLM_API_KEY`, prefixed); set it in `.env`, never commit or log it |
| `LLM_DISABLE_THINKING` | `true` | local only: sends `enable_thinking=False` so replies start fast |
| `LLM_TIMEOUT_S` | `30` | request timeout for the local client |
| `LLM_REASONING_EFFORT` | `minimal` | Reasoning effort for gpt-5* models (`minimal`/`low`/`medium`/`high`); ignored for others |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | must stay 384-dim (`EMBEDDING_DIM`); changing models requires re-ingesting everything (docs and memories share the space) |
| `EMBEDDING_QUERY_PREFIX` | `Represent this sentence for searching relevant passages: ` | BGE query instruction, applied to search queries only (never documents); set empty when using a non-BGE embedding model |
| `STT_BACKEND` | `local` | `local` (faster-whisper), `parakeet` (NVIDIA Parakeet TDT via parakeet-mlx on the Apple GPU; install with `uv pip install --python .venv/bin/python parakeet-mlx`), or `openai` (`gpt-4o-transcribe`; ~1 s network latency). Compare them with `scripts/stt_eval.py`. **Current active backend: `parakeet`.** |
| `PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v2` | model for `STT_BACKEND=parakeet`; downloaded on first use |
| `STT_MODEL` | `base.en` | faster-whisper size for the local backend; `small.en` = more accurate, slower |
| `TTS_BACKEND` | `say` | `say` (macOS built-in), `qwen-tts` (voice clone, local MLX), or `kokoro` (preset voices, local MLX, ~25x faster than real time). Kokoro preset voices: prefix `a…` = US, `b…` = UK — e.g. `af_heart`, `bf_alice`, `bf_emma`, `am_michael`, `bm_george`. Needs `uv pip install --python .venv/bin/python "misaki[en]"`. `TTS_SPEED` is a real tempo change (pitch kept). Reachy patches an mlx-audio 0.4.4 length bug that crashed ~1 in 6 sentences. **Current active backend: `kokoro`, voice: `bf_alice`.** |
| `PERSONALITY` | `scouse` | `plain` (original wording), `british` (dry, understated wit) or `scouse` (dry wit plus a light Liverpool touch: "boss", "made up", "ta"; neutral on Reds vs Blues). Adds a style paragraph with example exchanges to the system prompt and rewords the fixed lines (greetings, wake lines, apology, fillers). Defined in `brain/persona.py` |
| `KOKORO_VOICE` | `af_heart` | voice for `TTS_BACKEND=kokoro` (Kokoro-82M via mlx-audio: preset voices, no cloning, ~25x faster than real time). Prefix sets the accent: `a…` US, `b…` UK — e.g. `af_heart`, `bf_emma`, `am_michael`, `bm_george`. Needs `uv pip install --python .venv/bin/python "misaki[en]"`. `TTS_SPEED` is a real tempo change here (pitch kept). Reachy patches an mlx-audio 0.4.4 length bug that crashed ~1 in 6 sentences |
| `TTS_MODEL` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | mlx-audio model id for `qwen-tts`; swap for a 1.7B variant if clone quality disappoints |
| `VOICE_SAMPLE` | unset | ~10 s clean WAV of the voice to clone; required for `qwen-tts` |
| `VOICE_SAMPLE_TEXT` | unset | transcript of the sample; omit → auto-transcribed once via Whisper on first synthesis |

## Perception

| Setting | Default | Notes |
|---|---|---|
| `CAMERA_INDEX` | `0` | which webcam OpenCV opens (Mac source only; vision is disabled for now) |
| `MEDIA_SOURCE` | `auto` | where the camera/mic/speaker live: `robot` (Reachy Mini via the SDK), `mac` (this machine's devices), or `auto` (robot if the daemon offers media, else Mac). Override per-run with `reachy-vec run --source robot\|mac\|auto`. Robot audio out needs `TTS_BACKEND=qwen-tts` (`say` still plays on the Mac) |
| `AUDIO_INPUT_RATE` | `16000` | target sample rate fed to VAD/STT/ECAPA; the robot mic source resamples its native rate down to this |

## Voice ID (Phase 2b)

| Setting | Default | Notes |
|---|---|---|
| `VOICE_THRESHOLD` | `0.30` | cosine gate for speaker recognition (within 0.05 below = "can't tell", answered anonymously and not greeted); voice is the only identity signal |
| `VOICE_MIN_UTTERANCE_S` | `1.0` | utterances shorter than this aren't voice-identified |
| `VOICE_PASSIVE_CAP` | `10` | max passively-banked voice embeddings kept per person (oldest evicted; explicit enrollment rows never pruned) |
| `ENROLL_VOICE_PHRASES` | `2` | spoken sentences recorded when a stranger enrolls their voice |

## Interaction pacing

| Setting | Default | Notes |
|---|---|---|
| `GREET_COOLDOWN_S` | `7200` | spoken "Hi <name>!" at most every 2 h per person; within cooldown = silent acknowledge |
| `SILENCE_TIMEOUT_S` | `30` | quiet time that ends a conversation (triggers memory distillation) |
| `WAKE_WORDS` | 20+ spellings: `reachy`, `reachie`, `reechy`, `reachi`, `reache`, `reachee`, `reechie`, `reechi`, `rechy`, `rachie`, `richie`, `ritchie`, `ritchy`, `riachi`, `riyachi`, `ryachi`, `haruchi`, `wake up`, `hey robot` | whole words that start a conversation; words after it are the first question. Mishearings only — everyday words like "reach", "reaching", "ready", "rich" are left out on purpose (they'd wake the robot constantly; spelling-similarity matching was measured and rejected for the same reason). Add misses you see in `data/reachy.log`. Env form: JSON list. **Current: 20+ entries including "wake up" and "hey robot".** |
| `STT_HOTWORDS` | `Reachy` | biases local whisper toward spelling the wake word correctly; empty = off (clear it if noise causes false wakes) |
| `RAG_K` / `RAG_MIN_SCORE` | `3` / `0.62` | at most this many retrieved chunks per turn, each scoring ≥ the floor (bge-small: small talk 0.45–0.61, demo questions 0.66–0.78), so chit-chat sends no demo text |
| `WAKE_LISTEN_S` | `5` | idle listen window; loops forever between windows |
| `IDLE_SLEEP_S` | `300` | not woken for this long → sleep motion; wakes on the next "Reachy" |
| `BARGE_IN` | `true` | let someone talk over a reply to interrupt it; the current sentence stops and their utterance becomes the next turn. `false` = reply always plays to the end |
| `DASHBOARD` | `false` | `run` also serves the dashboard on localhost and opens its Live page (same as `run --dashboard`); skipped if the port is taken |
| `DASHBOARD_PORT` | `8400` | port for that dashboard |
| `ACK_ENABLED` | `true` | speak a short filler ("Mm.") the moment a question is heard, while the LLM reads the prompt; it hides the wait rather than removing it. Skipped for a question asked with the wake word right after a greeting |
| `ACK_FILLERS` | unset (= the `PERSONALITY` style's fillers) | the fillers, picked at random; pre-generated at startup with `qwen-tts` so they play instantly (JSON list in env) |
| `BARGE_IN_MIN_SPEECH_S` | `0.7` | sustained speech needed to trigger an interrupt; raise if a loud room false-triggers (no echo cancellation — the robot's own speaker near its mic is the usual culprit) |

## Tools

| Setting | Default | Notes |
|---|---|---|
| `OPEN_URL_TRIGGER_WORDS` | `["open","show","pull up","bring up","launch","display","load up"]` | `open_url` only runs if the user's turn contains one of these as a whole word, so merely mentioning a demo never pops a browser. JSON list |
| `WEB_SEARCH` | `false` | enable the Tavily `web_search` tool; also needs `TAVILY_API_KEY`. Off by default to protect a limited credit budget. When enabled, the system prompt restricts it to genuine current/live-info questions; out-of-credits (Tavily 432/433) degrades to a spoken notice |

### Hermes (`ask_hermes`)

| Setting | Default | Notes |
|---|---|---|
| `HERMES_ENABLED` | `false` | register `ask_hermes`; also needs the `hermes` binary on PATH (otherwise `run` warns and leaves it off) |
| `HERMES_BIN` | `hermes` | binary name/path |
| `HERMES_PROFILE` | `reachy` | always run under this profile (empty is rejected) |
| `HERMES_TOOLSETS` | `web,browser,memory,skills,session_search,todo,cronjob,kanban` | read-only + scheduling toolsets passed via `-t` |
| `HERMES_WAIT_S` | `20` | how long the robot waits before handing off; a slower task for a recognised speaker is delivered later as a message from Hermes |
| `HERMES_TASK_TIMEOUT_S` | `600` | hard cap per task |
| `HERMES_TRANSPORT` | `cli` | `gateway` = send tasks to the running Hermes gateway's API server (no per-task startup; ~0.7 s once warm, ~13 s for the first run after a gateway restart, so Reachy warms it in the background at startup; per-teammate Hermes memory via `X-Hermes-Session-Key: reachy:<person_id>`). Falls back to `cli` if the key is missing, the gateway is down, or its API toolsets exceed `HERMES_TOOLSETS`. `cli` = one-shot `hermes -p reachy -z` per task. **Current active transport: `gateway`.** |
| `HERMES_GATEWAY_URL` | `http://127.0.0.1:8642` | gateway API server; Reachy calls `/p/<HERMES_PROFILE>/v1/runs` |
| `HERMES_API_KEY` | — | the profile's `API_SERVER_KEY`; read automatically from `~/.hermes/profiles/reachy/.env`, so it lives in one place |

**Gateway setup (one-off, outside this repo):** the gateway multiplexes profiles, so the shared listener is enabled in the **default** profile (`API_SERVER_ENABLED=true` + its own `API_SERVER_KEY` in `~/.hermes/.env`) and the `reachy` profile needs its own key (`~/.hermes/profiles/reachy/.env`). There is no per-request toolset flag: set `platform_toolsets.api_server` to the read-only + scheduling list in **both** profiles (`hermes [-p reachy] config set platform_toolsets.api_server "[web, browser, memory, skills, session_search, todo, cronjob, kanban]"`), then `hermes gateway restart`. Runs that need a tool approval are stopped, never approved.

Hermes runs with its working directory at `data/hermes_cwd`, so it does not load the repo's agent rules.

## Environment

| Setting | Default | Notes |
|---|---|---|
| `ROBOT_HOST` | unset | remote Reachy Mini address; when set, `make_robot` connects in `connection_mode="network"` to `ROBOT_HOST:ROBOT_PORT`. Unset = local daemon (SDK `auto`). Falls back to a logging `NullBody` if none is reachable |
| `ROBOT_PORT` | `8000` | daemon port, used with `ROBOT_HOST` |
| `ROBOT_RECONNECT` | `true` | rebuild the body connection after a transient motion drop; `false` = a single drop degrades straight to body-less. Robot media (camera/mic) does not hot-recover either way — it soft-degrades (blank frame / silence) until the stream returns |
| `BODY_RECONNECT_ATTEMPTS` | `3` | consecutive motion failures before the robot gives up on its body for the session (speaks one notice, keeps listening) |
| `DATA_DIR` | `data` | holds `lancedb/` (all tables), `hermes_cwd/` (Hermes working dir), `reachy.log` (transcript log — privacy-relevant, gitignored) |
| `WEATHER_LAT` / `WEATHER_LON` | `53.4084` / `-2.9916` | lab location for `get_weather` (Liverpool, UK; Open-Meteo, no key) |

## Not configurable (code constants)

Chunk size (1000 chars, `store/ingestion.py`), retrieval k (5 docs / 3
memories, `brain/chat.py`), history window (20 messages), memory-duplicate
threshold (0.97), VAD sensitivity (0.5) and end-of-utterance quiet (0.8 s,
`audio/listen.py`), borderline margin for voices (0.05), ECAPA model (`spkrec-ecapa-voxceleb`, `perception/voice.py`).
Promote one to `config.py` if you find yourself tuning it.

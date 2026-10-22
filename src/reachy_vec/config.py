"""Settings for the brain, loaded from environment / .env (layered).

Priority, highest first:
1. Real environment variables (exported by the shell/CI)
2. ``.env`` at the repo root (per-machine overrides)
3. The Reachy profile's ``.env`` — ``~/.hermes/profiles/reachy/.env`` —
   the deployment's shared home for secrets and defaults (loaded only if
   it exists)

Environment variables use the REACHY_VEC_ prefix, e.g. REACHY_VEC_ROBOT_HOST.
OPENAI_API_KEY is read by the openai SDK directly.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deployment-level config home: shared secrets/defaults for this robot brain
# (e.g. OPENAI_API_KEY, REACHY_VEC_ROBOT_HOST). The repo .env overrides it.
REACHY_PROFILE_ENV = Path.home() / ".hermes" / "profiles" / "reachy" / ".env"


def _env_files() -> tuple[str, ...]:
    """Layered env files; the LAST entry wins (see tests/test_config.py)."""
    files: list[str] = []
    if REACHY_PROFILE_ENV.exists():
        files.append(str(REACHY_PROFILE_ENV))
    files.append(".env")
    return tuple(files)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REACHY_VEC_", env_file=_env_files(), extra="ignore"
    )

    # Robot (wireless Reachy Mini on the local network); None = simulator/headless
    robot_host: str | None = None
    robot_port: int = 8000  # daemon port; used with robot_host in network mode
    robot_reconnect: bool = True  # rebuild the body connection after a transient drop
    body_reconnect_attempts: int = 3  # consecutive motion failures before giving up

    # Models
    llm_model: str = "gpt-5-mini"
    llm_reasoning_effort: str = "minimal"  # gpt-5* only; keeps time-to-first-sentence low
    llm_base_url: str | None = None  # local OpenAI-compatible server (oMLX); None = OpenAI cloud
    llm_api_key: str | None = None  # key for that server; never logged
    llm_disable_thinking: bool = True  # local only: send enable_thinking=False
    llm_timeout_s: float = 30.0  # local client request timeout
    stt_model: str = "base.en"  # faster-whisper size; english-only = fastest + most accurate for EN
    stt_backend: str = "local"  # local (faster-whisper) | parakeet (parakeet-mlx) | openai
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v2"  # stt_backend=parakeet
    stt_hotwords: str = "Reachy"  # bias local whisper toward the wake word ("Reachy"
    # was heard as "I'm reaching."); empty = off. Clear it if noise causes false wakes
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )  # BGE query instruction; set empty to disable for non-BGE models
    rag_k: int = 3  # max retrieved doc chunks per turn
    rag_min_score: float = 0.62  # cosine floor for retrieved chunks; small talk scores
    # ~0.45-0.61 and demo questions ~0.66-0.78 on bge-small, so chit-chat gets none
    tts_backend: str = "say"  # say (macOS) | qwen-tts (voice clone) | kokoro (fast presets)
    kokoro_voice: str = "af_heart"  # tts_backend=kokoro; prefix a=US b=UK, e.g. bf_emma, am_michael
    tts_model: str = ""  # mlx-audio model id; empty -> QWEN_TTS_DEFAULT_MODEL (0.6B)
    # tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"  # mlx-audio model id

    # Vision (look/selfie/look_around tools; currently not wired into run); None => llm_model
    vision_model: str | None = None
    camera_tools: bool = True  # look / selfie / look_around on the robot camera
    vision_image_max_px: int = 1024  # downscale long edge before encoding, to cut vision tokens
    look_around_settle_s: float = 0.8  # pause after a base turn before capturing,
    # so the WebRTC frame catches up to the new heading (first frame lags ~2s)

    voice_sample: Path | None = None  # ~10s clean WAV of the voice to clone (qwen-tts)
    voice_sample_text: str | None = None  # its transcript; omit -> auto-transcribed once
    tts_target_rms: float = 0.16  # boost quieter TTS output to this level before
    # playback (clone output tracks the sample's level); a soft limiter tames
    # peaks so the sustained body reaches the target without clipping. 0 = off
    tts_presence_db: float = 4.0  # high-shelf lift above tts_presence_hz for
    # clarity/intelligibility; the clone's brightness tracks the sample. 0 = off
    tts_presence_hz: float = 3200.0  # presence shelf corner frequency
    tts_speed: float = 1.0  # >1 plays the cloned voice faster AND higher (tape-style, no
    # processing cost) - a livelier delivery; ~1.05-1.12, beyond that it sounds sped up

    # Perception
    camera_index: int = 0
    media_source: str = "auto"  # auto | robot | mac — where camera/mic/speaker live
    mic_source: str = "auto"  # auto (follow media_source) | mac (Mac default input, e.g.
    # a USB mic, even with the robot speaker) | robot
    audio_input_rate: int = 16000  # target rate fed to VAD/STT/ECAPA
    min_utterance_rms: float = 0.005  # discard quieter captures (STT hallucinates on
    # near-silence); raise if phantom demo names persist, lower if real speech drops

    # Camera-poke reaction: frame suddenly dark = someone covered the lens
    occlusion_dark_floor: float = 28.0  # mean brightness (0-255) below = dark
    occlusion_drop_ratio: float = 0.3   # ...and below this fraction of baseline
    occlusion_polls: int = 2            # consecutive dark polls to fire
    occlusion_cooldown_s: float = 10.0  # min seconds between reactions

    utterance_max_silence_s: float = 1.2  # quiet that ends an utterance; raise if
    # long sentences get split at natural pauses, lower if turn-taking feels slow
    utterance_max_s: float = 30.0  # hard cap on one utterance once speech starts
    stt_target_rms: float = 0.05  # boost quiet captures to this level before STT
    # (far-field mic runs quiet; low level hurts accuracy); 0 = off

    # Voice ID - the only identity signal (face recognition was removed)
    voice_threshold: float = 0.30  # below = unknown; within 0.05 under = "can't tell"
    voice_min_utterance_s: float = 1.0  # shorter audio -> can't tell
    voice_passive_cap: int = 10  # max passively-banked embeddings per person

    # Interaction
    greet_cooldown_s: float = 7200.0  # full spoken greeting at most every 2h
    silence_timeout_s: float = 30.0   # end conversation after this much quiet
    idle_sleep_s: float = 300.0       # not called for this long -> robot sleeps
    wake_words: list[str] = [
        "reachy", "reachie", "reechy", "reachi", "reache", "reachee", "reechie", "reechi",
        "rechy", "rachie", "richie", "ritchie", "ritchy", "riachi", "riyachi", "ryachi",
        "haruchi",
    ]
    # mishearings of "Reachy" that are not everyday words - "reach", "reaching",
    # "ready", "rich" are deliberately absent (they'd wake the robot constantly)
    # transcript words that start a conversation (incl. common STT mishearings);
    # env form: REACHY_VEC_WAKE_WORDS='["reachy","richie"]'
    wake_listen_s: float = 5.0  # idle listen window; loops forever between windows
    enroll_auto_offer: bool = False  # offer enrollment unasked to unknown voices; off =
    # assistant first - people say "Reachy, remember me" when they want it
    enroll_voice_phrases: int = 2  # sentences recorded when enrolling a new voice
    post_speech_settle_s: float = 0.3  # mic opens this long after the robot stops
    # talking, so the echo tail of its own voice isn't heard as an utterance
    barge_in: bool = True  # allow talking over a reply to interrupt it
    barge_in_mode: str = "keyword"  # keyword: only "Reachy"/"stop"/"wait"... interrupts
    # (the mic hearing the robot's own voice can't) | speech: any sustained speech
    barge_in_min_speech_s: float = 0.7  # sustained speech needed to fire (raise if false triggers)
    speech_wobble: bool = True  # audio-synced head sway while the robot speaker plays
    dashboard: bool = False  # run also serves the dashboard and opens its Live page
    dashboard_port: int = 8400  # (localhost only; same as `reachy-vec dashboard`)
    ack_enabled: bool = True  # speak a short filler while the LLM reads the prompt
    ack_fillers: list[str] = ["Mm.", "Right.", "Okay.", "Let me see."]
    # pre-generated at startup (qwen-tts), so they play instantly

    # Weather (Open-Meteo, no API key); default: Liverpool, UK
    weather_lat: float = 53.4084
    weather_lon: float = -2.9916

    # Tools
    open_url_trigger_words: list[str] = [
        "open", "show", "pull up", "bring up", "launch", "display", "load up", "play",
    ]  # open_url runs only if the user's turn contains one of these (whole words)
    web_search: bool = False  # enable the Tavily web_search tool (needs TAVILY_API_KEY)

    # Hermes agent (ask_hermes tool: delegates spoken tasks to the local Hermes CLI)
    hermes_enabled: bool = False
    hermes_bin: str = "hermes"
    hermes_profile: str = "reachy"
    hermes_toolsets: str = "web,browser,memory,skills,session_search,todo,cronjob,kanban"
    hermes_wait_s: float = 20.0  # how long to wait before handing off to background
    hermes_wait_anonymous_s: float = 60.0  # unrecognised speakers get no late delivery,
    # so wait longer and answer in the same turn
    hermes_task_timeout_s: float = 600.0
    hermes_transport: str = "cli"  # cli (one-shot `hermes -z`) | gateway (running API server)
    hermes_gateway_url: str = "http://127.0.0.1:8642"
    # the reachy profile's API_SERVER_KEY - read straight from
    # ~/.hermes/profiles/reachy/.env (loaded above), so it lives in one place
    hermes_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REACHY_VEC_HERMES_API_KEY", "API_SERVER_KEY"),
    )

    # Storage
    data_dir: Path = Path("data")

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"


settings = Settings()

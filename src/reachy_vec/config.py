"""Settings for the brain, loaded from environment / .env.

Environment variables use the REACHY_VEC_ prefix, e.g. REACHY_VEC_ROBOT_HOST.
OPENAI_API_KEY is read by the openai SDK directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REACHY_VEC_", env_file=".env", extra="ignore"
    )

    # Robot (wireless Reachy Mini on the local network); None = simulator/headless
    robot_host: str | None = None
    robot_port: int = 8000  # daemon port; used with robot_host in network mode
    robot_reconnect: bool = True  # rebuild the body connection after a transient drop
    body_reconnect_attempts: int = 3  # consecutive motion failures before giving up

    # Models
    llm_model: str = "gpt-5-mini"
    llm_reasoning_effort: str = "minimal"  # gpt-5* only; keeps time-to-first-sentence low
    stt_model: str = "base.en"  # faster-whisper size; english-only = fastest + most accurate for EN
    stt_backend: str = "local"  # local (faster-whisper) | openai (gpt-4o-transcribe)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )  # BGE query instruction; set empty to disable for non-BGE models
    tts_backend: str = "say"  # say (macOS built-in) | qwen-tts (voice clone, local MLX)
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"  # mlx-audio model id
    # tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"  # mlx-audio model id

    voice_sample: Path | None = None  # ~10s clean WAV of the voice to clone (qwen-tts)
    voice_sample_text: str | None = None  # its transcript; omit -> auto-transcribed once

    # Perception
    face_threshold: float = 0.45  # cosine similarity; below = unknown
    camera_index: int = 0
    media_source: str = "auto"  # auto | robot | mac — where camera/mic/speaker live
    audio_input_rate: int = 16000  # target rate fed to VAD/STT/ECAPA

    # Voice ID (Phase 2b) - ECAPA cosine scores run lower than face scores
    voice_threshold: float = 0.30  # below = unknown; within 0.05 under = "can't tell"
    voice_min_utterance_s: float = 1.0  # shorter audio -> can't tell
    voice_passive_cap: int = 10  # max passively-banked embeddings per person

    # Interaction
    greet_cooldown_s: float = 7200.0  # full spoken greeting at most every 2h
    silence_timeout_s: float = 30.0   # end conversation after this much quiet
    idle_sleep_s: float = 300.0       # no faces for this long -> robot sleeps

    # Weather (Open-Meteo, no API key); default: Liverpool, UK
    weather_lat: float = 53.4084
    weather_lon: float = -2.9916

    # Storage
    data_dir: Path = Path("data")

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"


settings = Settings()

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

    # Models
    llm_model: str = "gpt-4o"
    stt_model: str = "small"  # faster-whisper size
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    tts_backend: str = "say"  # say (works today) | fish-speech (planned primary) | openvoice
    voice_sample: Path | None = None  # reference audio for voice cloning

    # Perception
    face_threshold: float = 0.45  # cosine similarity; below = unknown
    camera_index: int = 0

    # Storage
    data_dir: Path = Path("data")

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"


settings = Settings()

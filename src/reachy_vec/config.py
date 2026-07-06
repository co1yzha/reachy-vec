"""Settings for the brain, loaded from environment / .env.

Environment variables use the REACHY_VEC_ prefix, e.g. REACHY_VEC_ROBOT_HOST.
ANTHROPIC_API_KEY is read by the anthropic SDK directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REACHY_VEC_", env_file=".env")

    # Robot (wireless Reachy Mini on the local network); None = simulator/headless
    robot_host: str | None = None

    # Models
    llm_model: str = "claude-sonnet-5"
    stt_model: str = "small"  # faster-whisper size
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Storage
    data_dir: Path = Path("data")

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"


settings = Settings()

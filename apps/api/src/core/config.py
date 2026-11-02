from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(ROOT_ENV_PATH)),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    APP_NAME: str = "chronorig-api"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Database & Redis (Safe local defaults for development & pytest)
    DATABASE_ASYNC_URL: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://chronorig:chronorig_dev_pwd@localhost:5432/chronorig"
    )
    DATABASE_SYNC_URL: PostgresDsn = PostgresDsn(
        "postgresql+psycopg://chronorig:chronorig_dev_pwd@localhost:5432/chronorig"
    )
    REDIS_URL: RedisDsn = RedisDsn("redis://localhost:6379/0")

    # 1. Meta Model API ($150 Hackathon Credits)
    META_MODEL_API_KEY: SecretStr = Field(
        default=SecretStr("replace_with_meta_model_api_key_min_20_chars"),
        min_length=16,
    )
    META_MODEL_BASE_URL: str = "https://api.meta.ai/v1"
    SAM_MODEL_ID: str = "sam-3.1"
    MUSE_SPARK_MODEL_ID: str = "muse-spark-1.3"
    MUSE_IMAGE_MODEL_ID: str = "muse-image"
    MUSE_WONDER_MODEL_ID: str = "muse-wonder-1.0"
    MUSE_VOICE_MODEL_ID: str = "muse-voice-transcribe-1.0"

    # 2. Partner Open-Weight Inference (Baseten $100 + 2M TPM)
    BASETEN_API_KEY: SecretStr = Field(
        default=SecretStr("replace_with_baseten_api_key_min_20_chars"),
        min_length=16,
    )
    BASETEN_BASE_URL: str = "https://inference.baseten.co/v1"
    MUSE_SPARK_OPEN_MODEL_ID: str = "meta-models/muse-spark-open-1026"
    MUSE_GLIMMER_MODEL_ID: str = "meta-models/muse-glimmer-30b"

    # Security & Limits
    WEB_STUDIO_INTERNAL_URL: str = "http://localhost:3000"
    JWT_SECRET_KEY: SecretStr = Field(
        default=SecretStr("chronorig_dev_jwt_secret_key_minimum_32_bytes_long"),
        min_length=32,
    )
    JWT_ALGORITHM: str = "HS256"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    MAX_VIDEO_DURATION_SEC: float = 15.0
    MAX_VIDEO_SIZE_BYTES: int = 50 * 1024 * 1024

    # Object Storage
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_BUCKET_NAME: str = "chronorig-artifacts"
    S3_ACCESS_KEY_ID: SecretStr = SecretStr("minioadmin")
    S3_SECRET_ACCESS_KEY: SecretStr = SecretStr("minioadmin")

    @field_validator("DATABASE_ASYNC_URL")
    @classmethod
    def ensure_asyncpg_driver(cls, v: PostgresDsn) -> PostgresDsn:
        if not str(v).startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_ASYNC_URL must use postgresql+asyncpg:// driver")
        return v

    @model_validator(mode="after")
    def enforce_production_secrets(self) -> Settings:
        if self.ENVIRONMENT == "production":
            if self.META_MODEL_API_KEY.get_secret_value().startswith("replace_with"):
                raise ValueError("Production requires a real META_MODEL_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

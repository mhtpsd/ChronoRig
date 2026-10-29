from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    ENVIRONMENT: Literal["development", "staging", "production", "test"]
    APP_NAME: str = "chronorig-api"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Database & Redis (Fail fast if missing)
    DATABASE_ASYNC_URL: PostgresDsn
    DATABASE_SYNC_URL: PostgresDsn
    REDIS_URL: RedisDsn

    # Meta Model API Configuration
    META_MODEL_API_KEY: SecretStr = Field(..., min_length=20)
    META_MODEL_BASE_URL: str = "https://api.meta.ai/v1"
    SAM_MODEL_ID: str = "sam-3.1"
    MUSE_SPARK_MODEL_ID: str = "muse-spark-1.3"
    MUSE_IMAGE_MODEL_ID: str = "muse-image-1.0"

    # Security & Limits
    JWT_SECRET_KEY: SecretStr = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    MAX_VIDEO_DURATION_SEC: float = 15.0
    MAX_VIDEO_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    # Object Storage
    S3_ENDPOINT_URL: str
    S3_BUCKET_NAME: str
    S3_ACCESS_KEY_ID: SecretStr
    S3_SECRET_ACCESS_KEY: SecretStr

    @field_validator("DATABASE_ASYNC_URL")
    @classmethod
    def ensure_asyncpg_driver(cls, v: PostgresDsn) -> PostgresDsn:
        if not str(v).startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_ASYNC_URL must use postgresql+asyncpg:// driver")
        return v


@lru_cache
def get_settings() -> Settings:
    # Instantiating Settings() immediately crashes startup if any required var is missing/invalid
    return Settings()


settings = get_settings()

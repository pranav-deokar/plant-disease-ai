"""
Central configuration — reads from environment variables.
All secrets MUST be set via .env in development or secret manager in production.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "Plant Disease AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development | staging | production

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24          # 24h
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ALGORITHM: str = "HS256"
    API_KEY_HEADER: str = "X-API-Key"

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost"]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str                           # postgresql+asyncpg://...
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # ── MongoDB (disease knowledge base) ─────────────────────────────────────
    MONGODB_URL: str
    MONGODB_DB_NAME: str = "plant_disease_kb"

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    # ── Object Storage ────────────────────────────────────────────────────────
    S3_BUCKET_NAME: str = "plant-disease-images"
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    # For local dev, point to MinIO
    S3_ENDPOINT_URL: str = ""

    # ── ML Model ──────────────────────────────────────────────────────────────
    MODEL_DIR: Path = Path("/app/models")
    PRIMARY_MODEL_NAME: str = "mobilenet_v3_plantvillage"
    PRIMARY_MODEL_VERSION: str = "v1.0.0"

    FALLBACK_MODEL_NAME: str | None = None
    MODEL_CONFIDENCE_THRESHOLD: float = 0.65
    GRAD_CAM_LAYER: str = "features.8"         # layer name for Grad-CAM
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    IMAGE_TARGET_SIZE: tuple = (380, 380)      # EfficientNet-B4 input

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── MLflow ────────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "plant_disease_detection"

    # ── Monitoring ────────────────────────────────────────────────────────────
    SENTRY_DSN: str = ""
    LOG_LEVEL: str = "INFO"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

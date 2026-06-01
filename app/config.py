from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_callback_secret: str = Field(default="")
    model_callback_timeout_seconds: float = Field(default=30, gt=0)

    # MinIO / S3
    minio_endpoint: str = Field(default="http://minio:9000")
    minio_access_key: str = Field(default="")
    minio_secret_key: str = Field(default="")
    minio_use_ssl: bool = Field(default=False)

    # 파이프라인 Python 인터프리터 (venv 사용 시 절대경로 지정)
    pipeline_python: str = Field(default="python3")


def get_settings() -> Settings:
    return Settings()

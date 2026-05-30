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


def get_settings() -> Settings:
    return Settings()

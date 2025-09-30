from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)
    workers: int = Field(default=4)

    secret_key: str = Field(default="dev-secret-change-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    database_url: str = Field(default="postgresql://insight_user:insight_password@localhost:5432/insight_db")
    redis_url: str = Field(default="redis://localhost:6379/0")

    model_name: str = Field(default="distilbert-base-uncased")
    model_path: str = Field(default="./data/models/sentiment_model_improved")
    max_sequence_length: int = Field(default=512)
    batch_size: int = Field(default=32)

    newsapi_key: Optional[str] = Field(default=None)
    alpha_vantage_key: Optional[str] = Field(default=None)
    
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
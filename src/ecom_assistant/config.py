"""Validated environment-backed application settings."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration; secrets are never included in repr output."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = "E-Commerce Shopping Assistant"
    app_env: Literal["development", "test", "production"] = "development"
    llm_mode: Literal["mock", "openai"] = "mock"
    openai_api_key: SecretStr | None = None
    database_url: str = "sqlite:///./data/ecom_assistant.db"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    catalog_seed_path: Path = Path(__file__).resolve().parents[2] / "data" / "catalog.csv"
    faq_source_path: Path = Path(__file__).resolve().parents[2] / "data" / "faq_documents.json"
    faq_persist_directory: Path = Path("data/chroma")
    faq_collection_name: str = "product_faq"
    faq_embedding_provider: Literal["local_hash", "openai"] = "local_hash"
    faq_embedding_model: str = "text-embedding-3-small"
    faq_chat_model: str = "gpt-4o-mini"
    cart_proposal_ttl_seconds: int = 300
    conversation_context_ttl_seconds: int = 3600

    @field_validator("app_name")
    @classmethod
    def app_name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("APP_NAME must not be blank")
        return value.strip()

    @field_validator("database_url")
    @classmethod
    def database_must_use_sqlite(cls, value: str) -> str:
        if not value.startswith("sqlite:"):
            raise ValueError("DATABASE_URL must use SQLite in this phase")
        return value

    @field_validator("cart_proposal_ttl_seconds")
    @classmethod
    def proposal_ttl_must_be_bounded(cls, value: int) -> int:
        if not 30 <= value <= 3600:
            raise ValueError("CART_PROPOSAL_TTL_SECONDS must be between 30 and 3600")
        return value

    @field_validator("conversation_context_ttl_seconds")
    @classmethod
    def conversation_ttl_must_be_bounded(cls, value: int) -> int:
        if not 60 <= value <= 86400:
            raise ValueError("CONVERSATION_CONTEXT_TTL_SECONDS must be between 60 and 86400")
        return value

    @model_validator(mode="after")
    def openai_mode_requires_key(self) -> "Settings":
        key_is_required = self.llm_mode == "openai" or self.faq_embedding_provider == "openai"
        if key_is_required and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the configured OpenAI provider")
        if self.openai_api_key and not self.openai_api_key.get_secret_value().strip():
            if key_is_required:
                raise ValueError("OPENAI_API_KEY must not be blank for the configured OpenAI provider")
            self.openai_api_key = None
        return self
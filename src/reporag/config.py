from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Database Settings
    database_url: str = "sqlite+aiosqlite:///reporag.db"

    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = Field(default=SecretStr("password"))

    # Qdrant Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # LLM Settings
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Auth Settings
    secret_key: SecretStr = Field(default=SecretStr("super-secret-key"))
    jwt_secret_key: SecretStr = Field(default=SecretStr("super-jwt-key"))
    google_client_secret: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.app_env == "production":
            errors = []
            if not self.secret_key or self.secret_key.get_secret_value() in (
                "super-secret-key",
                "",
                "change-me",
            ):
                errors.append("SECRET_KEY must be set to a secure value in production.")
            if not self.jwt_secret_key or self.jwt_secret_key.get_secret_value() in (
                "super-jwt-key",
                "",
                "change-me",
            ):
                errors.append(
                    "JWT_SECRET_KEY must be set to a secure value in production."
                )

            if self.llm_provider == "openai" and not self.openai_api_key:
                errors.append(
                    "OPENAI_API_KEY must be set in production when using OpenAI."
                )
            elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
                errors.append(
                    "ANTHROPIC_API_KEY must be set in production when using Anthropic."
                )

            if errors:
                raise ValueError(
                    "Production configuration missing required secrets:\n"
                    + "\n".join(errors)
                )
        return self


settings = Settings()

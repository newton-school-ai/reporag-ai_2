"""RepoRAG configuration module.

Centralized, type-safe application settings loaded from environment variables
(and an optional ``.env`` file) via Pydantic Settings. Importing ``settings``
gives every module a single validated source of configuration -- no scattered
``os.getenv`` calls.

Secrets are stored as :class:`~pydantic.SecretStr` so they are never printed or
serialized by accident. In production (``APP_ENV=production``) the required
secrets must be provided explicitly; otherwise importing this module raises a
``ValidationError`` at startup. In development and test the placeholder defaults
are allowed so the app imports with zero configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder values shipped in .env.example. Treated as "not set" so a copied
# but unedited .env can never satisfy the production secret requirements.
_PLACEHOLDER_SECRETS = {
    "",
    "change-me",
    "change-me-to-a-random-string",
    "sk-your-key-here",
    "sk-ant-your-key-here",
    "your-google-client-secret",
}


def _is_unset(secret: SecretStr) -> bool:
    """Return True if a secret is empty or still a documented placeholder."""
    return secret.get_secret_value().strip() in _PLACEHOLDER_SECRETS


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Field names map to upper-case environment variables (case-insensitive), so
    ``app_port`` is read from ``APP_PORT``. Every variable is documented in
    ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_debug: bool = True
    app_port: int = 8000
    app_host: str = "0.0.0.0"
    secret_key: SecretStr = SecretStr("change-me")

    # --- Database ---
    database_url: str = "sqlite:///./reporag.db"

    # --- Neo4j (knowledge graph) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("reporag123")

    # --- Qdrant (vector store) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_code: str = "reporag_code"
    qdrant_collection_docs: str = "reporag_docs"

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-sonnet-4-20250514"

    # --- Embedding models ---
    code_embedding_model: str = "microsoft/unixcoder-base"
    doc_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Auth: Google OAuth ---
    google_client_id: str = ""
    google_client_secret: SecretStr = SecretStr("")
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # --- Auth: JWT ---
    jwt_secret_key: SecretStr = SecretStr("change-me")
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # --- Rate limiting ---
    rate_limit_per_minute: int = 60

    # --- Retrieval ---
    vector_search_top_k: int = 20
    bm25_search_top_k: int = 20
    rerank_top_k: int = 10
    rrf_constant: int = 60

    # --- Ingestion ---
    max_repo_size_mb: int = 500
    clone_depth: int = 1
    chunk_max_tokens: int = 512
    supported_languages: str = "python,javascript,typescript"
    extension_map: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }

    # --- Feature flags ---
    enable_graph_retrieval: bool = True
    enable_reranker: bool = True
    enable_agentic_planner: bool = True

    # ----- Convenience accessors -----

    @property
    def is_production(self) -> bool:
        """True when running with ``APP_ENV=production``."""
        return self.app_env == "production"

    @property
    def supported_languages_list(self) -> list[str]:
        """``supported_languages`` parsed into a clean list of language names."""
        return [
            lang.strip() for lang in self.supported_languages.split(",") if lang.strip()
        ]

    @property
    def active_llm_api_key(self) -> SecretStr:
        """API key for the currently selected ``llm_provider``."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    # ----- Validation -----

    @model_validator(mode="after")
    def _require_secrets_in_production(self) -> Settings:
        """Fail fast if production is missing secrets it must not run without.

        In non-production environments the placeholder defaults are allowed so
        the app imports with zero configuration. In production, any unset secret
        raises a ``ValidationError`` at import time.
        """
        if self.app_env != "production":
            return self

        missing: list[str] = []
        if _is_unset(self.secret_key):
            missing.append("SECRET_KEY")
        if _is_unset(self.jwt_secret_key):
            missing.append("JWT_SECRET_KEY")
        if _is_unset(self.active_llm_api_key):
            missing.append(f"{self.llm_provider.upper()}_API_KEY")

        if missing:
            raise ValueError(
                "Missing required production secrets: "
                + ", ".join(missing)
                + ". Set them in the environment or .env before starting "
                "with APP_ENV=production."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated application settings singleton."""
    return Settings()


# Module-level singleton: ``from reporag.config import settings``.
settings = get_settings()

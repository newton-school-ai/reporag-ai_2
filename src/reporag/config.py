"""RepoRAG configuration module.

Loads settings from .env file using Pydantic Settings.
All configuration is centralized here -- no scattered os.getenv calls.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_env: str = "development"
    app_debug: bool = True
    app_port: int = 8000
    app_host: str = "0.0.0.0"
    secret_key: SecretStr = SecretStr("change-me")

    # Database
    database_url: str = "sqlite:///./reporag.db"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("reporag123")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_code: str = "reporag_code"
    qdrant_collection_docs: str = "reporag_docs"

    # LLM
    llm_provider: str = "openai"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Embedding models
    code_embedding_model: str = "microsoft/unixcoder-base"
    doc_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: SecretStr = SecretStr("")
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # JWT
    jwt_secret_key: SecretStr = SecretStr("change-me")
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Retrieval
    vector_search_top_k: int = 20
    bm25_search_top_k: int = 20
    rerank_top_k: int = 10
    rrf_constant: int = 60

    # Ingestion
    max_repo_size_mb: int = 500
    clone_depth: int = 1
    supported_languages: str = "python,javascript,typescript"

    # Feature flags
    enable_graph_retrieval: bool = True
    enable_reranker: bool = True
    enable_agentic_planner: bool = True


settings = Settings()

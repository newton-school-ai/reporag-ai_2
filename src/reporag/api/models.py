"""Pydantic schemas and data models for RepoRAG API.

Defines request and response structures for repository ingestion,
querying, citations, and system health checks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Repository models
# ---------------------------------------------------------------------------


class RepoIngestRequest(BaseModel):
    """Request payload to trigger asynchronous repository ingestion."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(
        ...,
        min_length=1,
        description="Remote Git repository HTTPS/SSH URL or local path.",
        examples=["https://github.com/fastapi/fastapi.git"],
    )
    branch: str | None = Field(
        None,
        description="Optional Git branch or tag name to clone.",
        examples=["main"],
    )
    shallow: bool = Field(
        True,
        description="Whether to perform a shallow clone (depth=1).",
    )


class RepoIngestResponse(BaseModel):
    """Response returned immediately when ingestion is queued."""

    repo_id: int = Field(..., description="Unique database ID of the repository.")
    name: str = Field(..., description="Inferred repository name.")
    url: str = Field(..., description="Repository URL.")
    status: str = Field(..., description="Current status (e.g. queued, processing).")
    message: str = Field(..., description="Human-readable status message.")


class RepositoryResponse(BaseModel):
    """Details of an ingested repository."""

    id: int = Field(..., description="Unique repository identifier.")
    name: str = Field(..., description="Repository name.")
    url: str = Field(..., description="Repository URL.")
    status: str = Field(
        ...,
        description="Current repository status (queued, cloning, processing, ready, failed).",
    )
    created_at: datetime | None = Field(
        None, description="Timestamp when record was created."
    )
    updated_at: datetime | None = Field(
        None, description="Timestamp when record was last updated."
    )


class RepositoryListResponse(BaseModel):
    """List of repositories with total count."""

    repositories: list[RepositoryResponse] = Field(
        default_factory=list, description="List of repositories."
    )
    total: int = Field(0, description="Total number of matching repositories.")


# ---------------------------------------------------------------------------
# Query & Citation models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request payload to query an ingested repository."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        description="Natural language question about the codebase.",
        examples=["How is authentication handled in this application?"],
    )
    repo_id: int | str = Field(
        ...,
        description="Target repository identifier (integer ID or repository name).",
        examples=[1],
    )
    top_k: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of context chunks to retrieve.",
    )
    include_citations: bool = Field(
        True,
        description="Whether to extract and validate line-level citations.",
    )
    strategy: str | None = Field(
        None,
        description="Optional retrieval strategy override ('hybrid', 'vector', 'bm25', 'graph').",
    )


class CitationResponse(BaseModel):
    """Line-level citation referencing retrieved source context."""

    file_path: str = Field(..., description="Path to the cited file.")
    start_line: int | None = Field(None, description="Cited start line number.")
    end_line: int | None = Field(None, description="Cited end line number.")
    snippet: str = Field("", description="Source code snippet matching the citation.")
    valid: bool | None = Field(
        None, description="True if citation is verified against retrieved context."
    )
    raw: str = Field("", description="Raw citation marker string from LLM answer.")


class QueryMetadata(BaseModel):
    """Execution metadata and performance diagnostics for a query."""

    model: str = Field("", description="LLM model name used for generation.")
    provider: str = Field("", description="LLM provider ('openai' or 'anthropic').")
    latency_seconds: float = Field(
        0.0, description="End-to-end execution time in seconds."
    )
    coverage: float = Field(
        0.0, description="Citation coverage score (0.0 to 1.0) of claims."
    )
    valid_citations: int = Field(0, description="Number of valid citations found.")
    invalid_citations: int = Field(0, description="Number of invalid citations found.")
    chunks_retrieved: int = Field(0, description="Number of code/doc chunks retrieved.")
    error: str | None = Field(
        None, description="Error message if generation or retrieval degraded."
    )


class QueryResponse(BaseModel):
    """Response payload containing generated answer, citations, and metadata."""

    answer: str = Field(..., description="Generated answer text.")
    citations: list[CitationResponse] = Field(
        default_factory=list, description="List of line-level citations."
    )
    metadata: QueryMetadata = Field(
        default_factory=QueryMetadata, description="Query execution metadata."
    )


# ---------------------------------------------------------------------------
# Health check models
# ---------------------------------------------------------------------------


class ComponentStatus(BaseModel):
    """Health status of an individual subsystem."""

    status: str = Field(
        ...,
        description="Status string ('healthy', 'unhealthy', 'degraded', 'not_configured').",
    )
    details: str | dict[str, Any] | None = Field(
        None, description="Optional diagnostic details or error message."
    )


class HealthResponse(BaseModel):
    """Aggregated health status of the RepoRAG pipeline and services."""

    status: str = Field(
        ...,
        description="Overall system health status ('healthy', 'degraded', 'unhealthy').",
    )
    components: dict[str, ComponentStatus | dict[str, Any]] = Field(
        ..., description="Map of component names to their health status."
    )
    version: str = Field("0.1.0", description="API version.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when health check was performed.",
    )

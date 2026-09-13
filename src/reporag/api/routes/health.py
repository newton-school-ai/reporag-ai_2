"""Health check endpoint.

GET /api/v1/health - Returns status of each pipeline component
(Neo4j, Qdrant, LLM, database).

Why
---
RepoRAG depends on three external services and an LLM provider. When an
answer comes back empty the first question is always "which dependency is
down?", and a single {"status": "ok"} cannot answer it. This endpoint
probes each component independently and reports them side by side.

Design
------
* Probe, do not trust config: A configured Qdrant URL proves nothing;
  the check performs a real round trip. The exception is the LLM, where a
  live call would cost money and seconds on every poll -- that one is
  reported from configuration only, and says so.
* Short timeouts: Every probe is bounded so a hung dependency cannot
  hang the health check that is meant to reveal it.
* Degraded, not down: A failing component yields HTTP 200 with
  status: "degraded". The API process itself is alive and can still
  serve the endpoints that do not need that component -- reporting 503
  would tell a load balancer to remove a node that is still useful.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.config import settings
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Reported by the health endpoint and used as the OpenAPI version in
# reporag.api.main, so the two can never disagree.
API_VERSION = "0.1.0"

# Every probe is bounded so a hung dependency cannot hang the health check.
_PROBE_TIMEOUT_SECONDS = 2.0

ComponentStatus = Literal["ok", "error", "not_configured"]


class ComponentHealth(BaseModel):
    """Health of a single dependency.

    Attributes:
        status: "ok" when reachable, "error" when the probe failed,
            "not_configured" when the component has no usable settings.
        detail: Human-readable explanation, including the failure reason.
    """

    status: ComponentStatus
    detail: str = ""


class HealthResponse(BaseModel):
    """Aggregate health of the pipeline.

    Attributes:
        status: "ok" when every component is healthy, otherwise "degraded".
        version: The running API version.
        environment: The active APP_ENV.
        components: Per-component health, keyed by component name.
    """

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)


async def _check_database(session: AsyncSession) -> ComponentHealth:
    """Probe the application database with a trivial round trip."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Database health probe failed: %s", exc)
        return ComponentHealth(status="error", detail=str(exc))
    dialect = session.bind.dialect.name if session.bind else "unknown"
    return ComponentHealth(status="ok", detail=f"connected ({dialect})")


def _check_neo4j() -> ComponentHealth:
    """Probe Neo4j with a bounded connectivity check.

    Deliberately uses the driver directly rather than GraphStore, whose
    constructor retries three times with exponential backoff -- roughly six
    seconds of sleeping that a health check must not pay for.
    """
    if not settings.neo4j_uri:
        return ComponentHealth(status="not_configured", detail="NEO4J_URI is unset")

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            connection_timeout=_PROBE_TIMEOUT_SECONDS,
        )
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
    except Exception as exc:
        logger.warning("Neo4j health probe failed: %s", exc)
        return ComponentHealth(status="error", detail=str(exc))

    return ComponentHealth(status="ok", detail=f"connected ({settings.neo4j_uri})")


def _check_qdrant() -> ComponentHealth:
    """Probe Qdrant by listing collections."""
    if not settings.qdrant_url:
        return ComponentHealth(status="not_configured", detail="QDRANT_URL is unset")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.qdrant_url, timeout=int(_PROBE_TIMEOUT_SECONDS)
        )
        try:
            collections = client.get_collections().collections
        finally:
            client.close()
    except Exception as exc:
        logger.warning("Qdrant health probe failed: %s", exc)
        return ComponentHealth(status="error", detail=str(exc))

    names = sorted(c.name for c in collections)
    return ComponentHealth(
        status="ok", detail=f"connected, {len(names)} collection(s): {', '.join(names)}"
    )


def _check_llm() -> ComponentHealth:
    """Report LLM readiness from configuration.

    Intentionally does not call the provider: a live completion on every
    health poll would cost money and add seconds of latency. A missing API
    key is the failure this can actually catch, and it is the common one.
    """
    from reporag.config import _is_unset

    provider = settings.llm_provider
    model = (
        settings.anthropic_model if provider == "anthropic" else settings.openai_model
    )

    if _is_unset(settings.active_llm_api_key):
        return ComponentHealth(
            status="not_configured",
            detail=f"{provider.upper()}_API_KEY is unset or still a placeholder",
        )
    return ComponentHealth(
        status="ok", detail=f"{provider} configured (model: {model})"
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Pipeline component health",
    response_description="Per-component status of every pipeline dependency.",
)
async def health(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> HealthResponse:
    """Report the status of each pipeline component.

    Returns HTTP 200 whether or not dependencies are healthy; inspect
    status and components to tell the difference. A component that
    is unreachable is reported as "error"; one that has no configuration
    at all is "not_configured".
    """
    neo4j_health, qdrant_health = await asyncio.gather(
        run_in_threadpool(_check_neo4j),
        run_in_threadpool(_check_qdrant),
    )
    components: dict[str, ComponentHealth] = {
        "database": await _check_database(session),
        "neo4j": neo4j_health,
        "qdrant": qdrant_health,
        "llm": _check_llm(),
    }

    overall: str = (
        "ok" if all(c.status == "ok" for c in components.values()) else "degraded"
    )
    return HealthResponse(
        status=overall,
        version=API_VERSION,
        environment=settings.app_env,
        components=components,
    )

"""Health check endpoints for RepoRAG.

GET /api/v1/health - Returns status of each pipeline component
(Neo4j, Qdrant, LLM, Database).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.models import ComponentStatus, HealthResponse
from reporag.config import _is_unset, settings
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


async def check_database_health(session: AsyncSession) -> ComponentStatus:
    """Check relational database connectivity."""
    try:
        result = await session.execute(text("SELECT 1"))
        result.scalar_one()
        return ComponentStatus(
            status="healthy",
            details="Database connection active and responding to queries.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database health check failed: %s", exc)
        return ComponentStatus(status="unhealthy", details=str(exc))


def check_neo4j_health() -> ComponentStatus:
    """Check Neo4j graph database connectivity."""
    try:
        from neo4j import GraphDatabase

        password = (
            settings.neo4j_password.get_secret_value()
            if hasattr(settings.neo4j_password, "get_secret_value")
            else str(settings.neo4j_password)
        )
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, password),
            connection_timeout=2.0,
        )
        driver.verify_connectivity()
        driver.close()
        return ComponentStatus(
            status="healthy",
            details={"uri": settings.neo4j_uri, "user": settings.neo4j_user},
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Neo4j health check failed (falling back to NetworkX): %s", exc)
        return ComponentStatus(
            status="degraded",
            details=f"Neo4j unavailable at {settings.neo4j_uri}; running with in-memory NetworkX store.",
        )


def check_qdrant_health() -> ComponentStatus:
    """Check Qdrant vector database connectivity."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=2.0)
        client.get_collections()
        client.close()
        return ComponentStatus(
            status="healthy",
            details={"url": settings.qdrant_url},
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Qdrant health check failed: %s", exc)
        return ComponentStatus(
            status="degraded",
            details=f"Qdrant vector store unreachable at {settings.qdrant_url}: {exc}",
        )


def check_llm_health() -> ComponentStatus:
    """Check configured LLM provider and API key readiness."""
    provider = settings.llm_provider
    key = settings.active_llm_api_key
    model = (
        settings.anthropic_model if provider == "anthropic" else settings.openai_model
    )

    if _is_unset(key):
        return ComponentStatus(
            status="not_configured",
            details=f"{provider.upper()}_API_KEY is unset or placeholder; live generation will be mocked or fail.",
        )

    return ComponentStatus(
        status="healthy",
        details={"provider": provider, "model": model},
    )


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Get pipeline and system health status",
    description="Inspects the connectivity and operational readiness of Neo4j, Qdrant, LLM provider, and Database.",
)
async def get_health(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> HealthResponse:
    """Return health status of each pipeline component."""
    db_status = await check_database_health(db)
    neo4j_status = check_neo4j_health()
    qdrant_status = check_qdrant_health()
    llm_status = check_llm_health()

    components: dict[str, Any] = {
        "database": db_status.model_dump(),
        "neo4j": neo4j_status.model_dump(),
        "qdrant": qdrant_status.model_dump(),
        "llm": llm_status.model_dump(),
    }

    if db_status.status == "unhealthy":
        overall_status = "unhealthy"
    elif any(
        c.status in ("degraded", "not_configured", "unhealthy")
        for c in (neo4j_status, qdrant_status, llm_status)
    ):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        components=components,
        version="0.1.0",
    )

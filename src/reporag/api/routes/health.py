"""Health check endpoint.

GET /api/v1/health - Returns status of each pipeline component
(Neo4j, Qdrant, LLM, database).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.config import settings
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_CHECK_TIMEOUT_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ComponentStatusValue = Literal["ok", "degraded", "down", "not_configured"]


class ComponentStatus(BaseModel):
    """Status of a single pipeline dependency (Neo4j, Qdrant, LLM, DB)."""

    name: str
    status: ComponentStatusValue
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Response for ``GET /api/v1/health``."""

    status: Literal["ok", "degraded", "down"]
    components: list[ComponentStatus]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------


async def _check_database(db: AsyncSession) -> ComponentStatus:
    start = time.monotonic()
    try:
        await db.execute(text("SELECT 1"))
        return ComponentStatus(
            name="database", status="ok", latency_ms=(time.monotonic() - start) * 1000
        )
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("Database health check failed: %s", exc)
        return ComponentStatus(name="database", status="down", detail=str(exc))


def _check_qdrant_sync() -> None:
    from qdrant_client import QdrantClient

    if settings.qdrant_url == ":memory:":
        QdrantClient(location=":memory:").get_collections()
        return
    url = settings.qdrant_url
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    QdrantClient(url=url, timeout=_CHECK_TIMEOUT_SECONDS).get_collections()


async def _check_qdrant() -> ComponentStatus:
    start = time.monotonic()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_check_qdrant_sync), timeout=_CHECK_TIMEOUT_SECONDS
        )
        return ComponentStatus(
            name="qdrant", status="ok", latency_ms=(time.monotonic() - start) * 1000
        )
    except Exception as exc:  # pragma: no cover - depends on live Qdrant
        logger.warning("Qdrant health check failed: %r", exc)
        return ComponentStatus(
            name="qdrant", status="down", detail=str(exc) or repr(exc)
        )


def _check_neo4j_sync() -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


async def _check_neo4j() -> ComponentStatus:
    start = time.monotonic()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_check_neo4j_sync), timeout=_CHECK_TIMEOUT_SECONDS
        )
        return ComponentStatus(
            name="neo4j", status="ok", latency_ms=(time.monotonic() - start) * 1000
        )
    except Exception as exc:  # pragma: no cover - depends on live Neo4j
        logger.warning("Neo4j health check failed: %r", exc)
        return ComponentStatus(
            name="neo4j", status="down", detail=str(exc) or repr(exc)
        )


def _check_llm() -> ComponentStatus:
    # Deliberately no network call here -- health checks run often and
    # shouldn't spend real LLM API quota. This only confirms a key is
    # configured for the active provider.
    key = settings.active_llm_api_key.get_secret_value().strip()
    placeholder = {"", "sk-your-key-here", "sk-ant-your-key-here"}
    if key in placeholder:
        return ComponentStatus(
            name="llm",
            status="not_configured",
            detail=f"No API key configured for provider {settings.llm_provider!r}.",
        )
    return ComponentStatus(
        name="llm", status="ok", detail=f"provider={settings.llm_provider}"
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(db: Annotated[AsyncSession, Depends(get_db)]) -> HealthResponse:
    """Report the status of every component the pipeline depends on."""
    db_status, qdrant_status, neo4j_status = await asyncio.gather(
        _check_database(db), _check_qdrant(), _check_neo4j()
    )
    llm_status = _check_llm()

    components = [db_status, qdrant_status, neo4j_status, llm_status]

    if all(c.status == "ok" for c in components):
        overall = "ok"
    elif db_status.status == "down":
        # The database is load-bearing for every endpoint; anything else
        # being down is a degraded-but-usable state (retrieval strategies
        # degrade individually -- see search_* helpers in query.py).
        overall = "down"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, components=components)

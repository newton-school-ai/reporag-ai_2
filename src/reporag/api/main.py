"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan
events. Run with: uvicorn reporag.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from reporag.api.routes import health, query, repos
from reporag.config import settings
from reporag.db.models import Base
from reporag.db.session import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    Neo4j and Qdrant connections are made lazily by the retrievers
    themselves on first use (see ``LiveRetrievalEngine`` in routes/query.py),
    so there's nothing to eagerly connect here. In development, the SQL
    schema is created automatically so the app runs with zero setup; in
    staging/production the schema is expected to be managed by Alembic
    migrations instead.
    """
    if not settings.is_production:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema ensured (development mode).")

    yield

    await engine.dispose()


app = FastAPI(
    title="RepoRAG API",
    version="0.1.0",
    description=(
        "Code-Aware Repository Intelligence -- Agentic RAG for Codebases. "
        "Ingest a Git repository, then ask natural-language questions about "
        "it and get cited answers."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Permissive by default for local development against the frontend dev
# server; tighten this to the deployed frontend origin(s) before shipping
# to staging/production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(repos.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/", tags=["health"])
def root() -> dict:
    """Basic liveness/info endpoint. See /api/v1/health for component status."""
    return {"name": "RepoRAG API", "version": app.version, "docs": "/docs"}

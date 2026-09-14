"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan events.
Run with: uvicorn src.reporag.api.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from reporag.api.routes.health import router as health_router
from reporag.api.routes.query import router as query_router
from reporag.api.routes.repos import router as repos_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown tasks."""
    logger.info("Starting RepoRAG API server...")
    yield
    logger.info("Shutting down RepoRAG API server...")


app = FastAPI(
    title="RepoRAG API",
    version="0.1.0",
    description="Intelligent repository RAG system for searching, querying, and understanding codebases with line-level citations.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers with /api/v1 prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(repos_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


@app.get("/health", tags=["Health"], summary="Top-level health check")
def root_health() -> dict[str, str]:
    """Simple ping health check for load balancers and orchestrators."""
    return {"status": "ok"}

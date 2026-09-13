"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan
events. Run with: uvicorn reporag.api.main:app --reload

Architecture notes:
- Use create_app() factory so tests can build isolated instances with
  custom state / engine without polluting the module-level singleton.
- Lifespan creates database schema outside staging/production so a fresh
  clone works immediately without running manual migrations.
- Liveness probe is static GET /health returning {"status": "ok"}.
- Component diagnostics live at GET /api/v1/health.
- Standardized error handlers render uniform JSON error responses.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from reporag.api.routes import health as health_routes
from reporag.api.routes import query as query_routes
from reporag.api.routes import repos as repos_routes
from reporag.config import settings
from reporag.db.models import Base
from reporag.db.session import engine as default_engine

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"

_ALEMBIC_MANAGED_ENVS = frozenset({"staging", "production"})

DESCRIPTION = """
Code-aware repository intelligence: ask questions about a codebase and get
answers cited to the exact file and line range.

* `POST /api/v1/repos/ingest` queues a repository for ingestion.
* `GET /api/v1/repos` reports ingestion status.
* `POST /api/v1/query` answers a question with citations.
* `GET /api/v1/health` reports the status of every pipeline component.
"""


def _db_engine(app: FastAPI) -> AsyncEngine:
    """Return the engine this app should manage.

    Tests build an isolated app against a temporary database and record its
    engine on app.state.db_engine; everything else uses the process-wide
    engine from reporag.db.session.
    """
    return getattr(app.state, "db_engine", None) or default_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    In development and test environments, ensures the database schema exists.
    In staging and production, schema is managed via Alembic migrations.
    On shutdown, disposes database engine and cleans up cached pipeline
    handles.
    """
    logger.info(
        "RepoRAG API starting (env=%s, llm=%s)",
        settings.app_env,
        settings.llm_provider,
    )

    engine = _db_engine(app)
    if settings.app_env not in _ALEMBIC_MANAGED_ENVS:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema ensured (env=%s)", settings.app_env)
        except Exception:
            logger.exception("Could not create database schema; continuing")

    yield

    # Clean up pipeline components if any were cached on app.state
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline and isinstance(pipeline, dict):
        for name, component in pipeline.items():
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Failed to close %s", name, exc_info=True)

    await engine.dispose()
    logger.info("RepoRAG API stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        The configured application instance.
    """
    app = FastAPI(
        title="RepoRAG API",
        version=health_routes.API_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Permissive by default for local development and frontend dev servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router, prefix=API_V1_PREFIX)
    app.include_router(repos_routes.router, prefix=API_V1_PREFIX)
    app.include_router(query_routes.router, prefix=API_V1_PREFIX)

    _register_exception_handlers(app)

    @app.get(
        "/",
        tags=["root"],
        summary="API root",
        response_description="Basic API information and link to documentation.",
    )
    async def root() -> dict[str, str]:
        """Return basic service metadata."""
        return {
            "name": "RepoRAG API",
            "version": health_routes.API_VERSION,
            "docs": "/docs",
        }

    @app.get(
        "/health",
        tags=["health"],
        summary="Liveness probe",
        response_description="Static acknowledgement that the process is up.",
    )
    async def liveness() -> dict[str, str]:
        """Report that the process is running.

        Deliberately static and dependency-free: orchestrators polling this
        never pay for downstream service checks.
        """
        return {"status": "ok"}

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Render every error as a uniform JSON structure."""

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return 422 with the offending fields named."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "Request validation failed.",
                "status_code": 422,
                "errors": [
                    {
                        "field": ".".join(str(p) for p in err.get("loc", ())),
                        "message": err.get("msg", ""),
                    }
                    for err in exc.errors()
                ],
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        """Render a raised HTTPException in the shared error shape."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "detail": exc.detail,
                "status_code": exc.status_code,
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Return 500 without leaking internal tracebacks."""
        logger.exception("Unhandled error for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
                "status_code": 500,
            },
        )


app = create_app()

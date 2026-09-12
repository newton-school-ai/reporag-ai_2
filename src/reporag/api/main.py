"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan
events. Run with: uvicorn reporag.api.main:app --reload

Why
---
The API is the seam between the RAG pipeline and everything outside it. It
stays deliberately thin: routers own their own request and response models,
and this module only assembles them, sets policy that applies to every
route, and makes failures uniform.

Design
------
* **Two health endpoints, on purpose.** ``GET /health`` is an unversioned
  liveness probe answering only "is this process up?" -- the shape
  container orchestrators and load balancers expect, and cheap enough to
  poll every second. ``GET /api/v1/health`` is the diagnostic view that
  probes each dependency. Collapsing them would make the liveness check pay
  for network round trips.
* **Errors are one shape.** Any handler can raise ``HTTPException``, and
  anything unhandled would otherwise leak a stack trace. Both are rendered
  as ``{"error", "detail", "status_code"}`` so clients parse one schema.
* **Lifespan touches one dependency, not four.** Connecting to Neo4j, Qdrant
  and the LLM at startup would make the API refuse to boot whenever any of
  them is slow, and the pieces that need them build lazily on first use
  anyway. The relational schema is the exception: every route depends on it,
  and outside production it is created at startup so a fresh clone runs
  without a migration step. Staging and production leave the schema to
  Alembic.
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

# Environments whose schema is owned by Alembic migrations. Creating tables
# from the ORM metadata there would silently diverge from migration history,
# so startup leaves the schema alone and lets a missing table surface as the
# deployment error it is.
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
    engine on ``app.state.db_engine``; everything else uses the process-wide
    engine from :mod:`reporag.db.session`. Without this seam the startup
    hook below would create tables in the real ``reporag.db`` file every
    time a test instantiated the app.
    """
    return getattr(app.state, "db_engine", None) or default_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup does one piece of I/O and no more. The retrieval backends and
    LLM client connect lazily on first use, so binding them here would only
    make the API fail to start when a dependency is briefly unavailable,
    without making any request faster. The SQL schema is different: every
    route holds a session, and a fresh clone has no tables, so outside
    production the schema is created here rather than making ``POST
    /repos/ingest`` the thing that discovers the database is empty. Under
    ``APP_ENV=staging`` or ``production`` the schema belongs to Alembic and
    is left alone.
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
            # A missing schema makes most routes fail, but not the liveness
            # probe or the health endpoint that would explain why -- and
            # those are exactly what an operator reaches for here. Refusing
            # to boot would take them away too.
            logger.exception("Could not create database schema; continuing")

    yield

    # Pipeline components cached by the query route hold Qdrant and Neo4j
    # handles; close what exposes a close().
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline:
        for name, component in pipeline.items():
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - shutdown must not fail
                    logger.warning("Failed to close %s", name, exc_info=True)

    await engine.dispose()
    logger.info("RepoRAG API stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    A factory rather than a module-level constant so tests can build an
    isolated app -- ``app.state`` caches the pipeline, and sharing that
    across tests would leak one test's fakes into the next.

    Returns:
        The configured application.
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

    # Permissive by default so the Vite dev server (Issue 30) can call the
    # API from another origin. Issue 29 narrows this for production.
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
        "/health",
        tags=["health"],
        summary="Liveness probe",
        response_description="Static acknowledgement that the process is up.",
    )
    async def liveness() -> dict[str, str]:
        """Report that the process is running.

        Deliberately static and dependency-free: an orchestrator polling
        this must not be told to restart a healthy process because Neo4j is
        slow. Use ``GET /api/v1/health`` for component diagnostics.
        """
        return {"status": "ok"}

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Render every error as one JSON shape.

    Clients should parse a single schema whatever went wrong, and an
    unhandled exception must never return a stack trace to the caller.
    """

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
        """Return 500 without leaking internals.

        The traceback goes to the log, where operators can see it; the
        caller gets a stable message.
        """
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

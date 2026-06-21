"""FastAPI application entrypoint.

Configures the FastAPI app and exposes a minimal liveness health check so
the API container starts (the Docker image runs
``uvicorn src.reporag.api.main:app``) and can be smoke-tested.

Full wiring -- repo/query routers, auth + CORS + rate-limit middleware,
lifespan events (connect to Neo4j + Qdrant on startup), and component-level
health checks -- is implemented in Issue 26.
"""

from fastapi import FastAPI

app = FastAPI(
    title="RepoRAG AI",
    description="Code-aware repository intelligence API.",
    version="0.1.0",
)


@app.get("/api/v1/health", tags=["health"])
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns a static status so orchestrators and smoke tests can confirm the
    API process is up. Component-level health (Neo4j, Qdrant, LLM, database)
    is added in Issue 26.
    """
    return {"status": "ok", "service": "reporag-api", "version": app.version}


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    """Root endpoint pointing at the interactive API docs."""
    return {
        "name": "RepoRAG AI",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan
events. Run with: uvicorn src.reporag.api.main:app --reload
"""

# TODO: Implement in Issue 26
# - Create FastAPI app with metadata (title, version, description)
# - Include routers: repos, query, health, auth
# - Register middleware: CORS, auth, rate limiter, logging, error handler
# - Lifespan events: connect to Neo4j + Qdrant on startup, close on shutdown
# - OpenAPI docs at /docs

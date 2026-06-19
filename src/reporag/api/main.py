from fastapi import FastAPI
from reporag.api.routes import health

app = FastAPI(
    title="RepoRAG AI",
    version="0.1.0",
    description="Code-Aware Repository Intelligence - Agentic RAG for Codebases",
)

# Register health router
app.include_router(health.router, prefix="/api/v1")


# Root level /health route for direct checks
@app.get("/health")
async def root_health():
    """Root-level health check endpoint."""
    return {"status": "healthy"}

# RepoRAG AI - API server image
FROM python:3.11-slim

LABEL maintainer="RepoRAG AI Pod"
LABEL description="FastAPI server for RepoRAG AI - code-aware repository intelligence"

WORKDIR /app

# System dependencies required to build tree-sitter grammars and other
# compiled wheels (torch, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached unless
# requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.reporag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

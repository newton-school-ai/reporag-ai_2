"""Unit tests for the FastAPI health check endpoint."""

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

from reporag.api.main import app

client = TestClient(app)


def test_health_check():
    """Test that GET /health returns 200 OK and status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_health_check():
    """Test that GET /api/v1/health returns 200 OK and status healthy."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

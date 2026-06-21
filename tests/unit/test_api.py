"""API smoke tests.

Confirms the FastAPI app imports (i.e. the ``uvicorn ...main:app`` target the
Docker container runs exists) and that the liveness health endpoint responds.
"""

from fastapi.testclient import TestClient

from reporag.api.main import app

client = TestClient(app)


def test_app_starts_and_exposes_app_object():
    assert app.title == "RepoRAG AI"
    assert app.version == "0.1.0"


def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "reporag-api"


def test_root_endpoint_points_to_docs():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "RepoRAG AI"

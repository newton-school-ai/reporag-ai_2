"""Tests for the FastAPI app wiring in main.py.

Covers:
* Root GET / endpoint
* OpenAPI docs at /docs and /redoc
* CORS headers
* All three routers mounted and reachable
* Lifespan: DB tables created on startup (smoke test)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    async def test_get_root_returns_200(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200

    async def test_root_has_name_field(self, async_client):
        resp = await async_client.get("/")
        assert "name" in resp.json()

    async def test_root_has_version_field(self, async_client):
        resp = await async_client.get("/")
        assert "version" in resp.json()

    async def test_root_has_docs_field(self, async_client):
        resp = await async_client.get("/")
        assert "docs" in resp.json()

    async def test_root_docs_field_points_to_docs(self, async_client):
        resp = await async_client.get("/")
        assert resp.json()["docs"] == "/docs"


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------


class TestOpenAPIDocs:
    async def test_docs_ui_returns_200(self, async_client):
        resp = await async_client.get("/docs")
        assert resp.status_code == 200

    async def test_redoc_ui_returns_200(self, async_client):
        resp = await async_client.get("/redoc")
        assert resp.status_code == 200

    async def test_openapi_json_returns_200(self, async_client):
        resp = await async_client.get("/openapi.json")
        assert resp.status_code == 200

    async def test_openapi_schema_contains_all_endpoints(self, async_client):
        resp = await async_client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/health" in paths
        assert "/api/v1/repos/ingest" in paths
        assert "/api/v1/repos" in paths
        assert "/api/v1/query" in paths

    async def test_openapi_schema_has_reporag_title(self, async_client):
        resp = await async_client.get("/openapi.json")
        info = resp.json()["info"]
        assert "RepoRAG" in info["title"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    async def test_cors_allow_origin_wildcard(self, async_client):
        resp = await async_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI/Starlette responds 200 for CORS preflight
        assert resp.status_code in (200, 204)
        assert (
            resp.headers.get("access-control-allow-origin") == "*"
            or resp.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )

    async def test_cors_header_present_on_get(self, async_client):
        with (
            patch(
                "reporag.api.routes.health._check_qdrant",
                new_callable=AsyncMock,
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="qdrant", status="ok"),
            ),
            patch(
                "reporag.api.routes.health._check_neo4j",
                new_callable=AsyncMock,
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="neo4j", status="ok"),
            ),
            patch(
                "reporag.api.routes.health._check_llm",
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="llm", status="ok"),
            ),
        ):
            resp = await async_client.get(
                "/api/v1/health", headers={"Origin": "http://example.com"}
            )
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Routers mounted and reachable
# ---------------------------------------------------------------------------


class TestRoutersMounted:
    async def test_health_router_reachable(self, async_client):
        with (
            patch(
                "reporag.api.routes.health._check_qdrant",
                new_callable=AsyncMock,
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="qdrant", status="ok"),
            ),
            patch(
                "reporag.api.routes.health._check_neo4j",
                new_callable=AsyncMock,
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="neo4j", status="ok"),
            ),
            patch(
                "reporag.api.routes.health._check_llm",
                return_value=__import__(
                    "reporag.api.routes.health", fromlist=["ComponentStatus"]
                ).ComponentStatus(name="llm", status="ok"),
            ),
        ):
            resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_repos_router_reachable(self, async_client):
        """GET /api/v1/repos should return 200, not 404."""
        resp = await async_client.get("/api/v1/repos")
        assert resp.status_code == 200

    async def test_query_router_reachable(self, async_client):
        """POST /api/v1/query with valid body should not return 404."""
        from unittest.mock import patch

        from tests.unit.test_api_query import _make_pipeline_result

        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post("/api/v1/query", json={"query": "hello"})
        assert resp.status_code != 404

    async def test_unknown_route_returns_404(self, async_client):
        resp = await async_client.get("/api/v1/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Lifespan smoke test
# ---------------------------------------------------------------------------


class TestLifespan:
    async def test_db_tables_exist_after_startup(self, async_client, db_session):
        """Verify the ORM tables are accessible after the lifespan hook runs."""
        from sqlalchemy import text

        result = await db_session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = {row[0] for row in result.fetchall()}
        assert "repositories" in tables
        assert "users" in tables
        assert "ingestion_jobs" in tables

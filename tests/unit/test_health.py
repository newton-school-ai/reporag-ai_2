"""Tests for GET /api/v1/health.

Covers every acceptance criterion for the health endpoint:

* Response shape: {status, components[{name, status, latency_ms, detail}], metadata}
* Overall status logic:
    - all ok             -> "ok"
    - DB down            -> "down"
    - any other down     -> "degraded"
    - LLM key not set    -> component "not_configured", overall "degraded"
* OpenAPI docs (/docs) reachable

External backends (Qdrant, Neo4j) are patched with AsyncMock so no real
network is contacted; the DB check uses the real in-memory test session.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from reporag.api.routes.health import ComponentStatus

# ---------------------------------------------------------------------------
# Reusable ComponentStatus fixtures
# ---------------------------------------------------------------------------

_OK_QDRANT = ComponentStatus(name="qdrant", status="ok", latency_ms=1.0)
_OK_NEO4J = ComponentStatus(name="neo4j", status="ok", latency_ms=1.0)
_OK_LLM = ComponentStatus(name="llm", status="ok", detail="provider='openai'")
_DOWN_QDRANT = ComponentStatus(
    name="qdrant", status="down", detail="Connection refused"
)
_DOWN_NEO4J = ComponentStatus(name="neo4j", status="down", detail="Service unavailable")
_LLM_NOT_CONFIGURED = ComponentStatus(
    name="llm",
    status="not_configured",
    detail="No API key configured for provider 'openai'.",
)


def _patch_externals(qdrant=_OK_QDRANT, neo4j=_OK_NEO4J, llm=_OK_LLM):
    """Stack the three non-DB component check patches and return as a list for ExitStack."""
    return [
        patch(
            "reporag.api.routes.health._check_qdrant",
            new_callable=AsyncMock,
            return_value=qdrant,
        ),
        patch(
            "reporag.api.routes.health._check_neo4j",
            new_callable=AsyncMock,
            return_value=neo4j,
        ),
        patch(
            "reporag.api.routes.health._check_llm",
            return_value=llm,
        ),
    ]


@contextmanager
def _apply_patches(patches):
    """Enter a list of patch context managers together."""
    entered = []
    try:
        for p in patches:
            entered.append(p.__enter__())
        yield entered
    finally:
        for p, _entered_ctx in zip(patches, entered, strict=False):
            p.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


class TestHealthResponseShape:
    async def test_returns_200(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_top_level_fields_present(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "components" in data
        assert "metadata" in data

    async def test_components_is_a_list_of_four(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        components = resp.json()["components"]
        assert isinstance(components, list)
        assert len(components) == 4  # database, qdrant, neo4j, llm

    async def test_each_component_has_required_fields(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        for comp in resp.json()["components"]:
            assert "name" in comp
            assert "status" in comp
            assert "latency_ms" in comp
            assert "detail" in comp

    async def test_component_names_are_expected_set(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        names = {c["name"] for c in resp.json()["components"]}
        assert names == {"database", "qdrant", "neo4j", "llm"}

    async def test_metadata_defaults_to_empty_dict(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["metadata"] == {}


# ---------------------------------------------------------------------------
# Overall status logic
# ---------------------------------------------------------------------------


class TestHealthOverallStatus:
    async def test_all_ok_gives_ok_status(self, async_client):
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"

    async def test_database_component_is_ok_with_test_db(self, async_client):
        """DB check uses a real in-memory session -- should always be ok."""
        with _apply_patches(_patch_externals()):
            resp = await async_client.get("/api/v1/health")
        db_comp = next(c for c in resp.json()["components"] if c["name"] == "database")
        assert db_comp["status"] == "ok"
        assert db_comp["latency_ms"] is not None
        assert db_comp["latency_ms"] >= 0

    async def test_qdrant_down_gives_degraded(self, async_client):
        with _apply_patches(_patch_externals(qdrant=_DOWN_QDRANT)):
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"

    async def test_neo4j_down_gives_degraded(self, async_client):
        with _apply_patches(_patch_externals(neo4j=_DOWN_NEO4J)):
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"

    async def test_both_qdrant_and_neo4j_down_gives_degraded(self, async_client):
        with _apply_patches(_patch_externals(qdrant=_DOWN_QDRANT, neo4j=_DOWN_NEO4J)):
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"

    async def test_db_down_gives_overall_down(self, async_client):
        down_db = ComponentStatus(
            name="database", status="down", detail="no connection"
        )
        extra = patch(
            "reporag.api.routes.health._check_database",
            new_callable=AsyncMock,
            return_value=down_db,
        )
        with _apply_patches(_patch_externals()), extra:
            resp = await async_client.get("/api/v1/health")
        assert resp.json()["status"] == "down"

    async def test_llm_not_configured_gives_degraded(self, async_client):
        with _apply_patches(_patch_externals(llm=_LLM_NOT_CONFIGURED)):
            resp = await async_client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "degraded"
        llm_comp = next(c for c in data["components"] if c["name"] == "llm")
        assert llm_comp["status"] == "not_configured"

    async def test_llm_not_configured_detail_mentions_api_key(self, async_client):
        with _apply_patches(_patch_externals(llm=_LLM_NOT_CONFIGURED)):
            resp = await async_client.get("/api/v1/health")
        llm_comp = next(c for c in resp.json()["components"] if c["name"] == "llm")
        detail = llm_comp["detail"] or ""
        assert "API key" in detail or "provider" in detail or "configured" in detail


# ---------------------------------------------------------------------------
# LLM key check -- unit-level, no HTTP
# ---------------------------------------------------------------------------


class TestCheckLLM:
    def test_empty_key_gives_not_configured(self):
        from reporag.api.routes.health import _check_llm

        with patch("reporag.api.routes.health.settings") as mock_settings:
            mock_settings.active_llm_api_key.get_secret_value.return_value = ""
            mock_settings.llm_provider = "openai"
            result = _check_llm()
        assert result.status == "not_configured"
        assert result.name == "llm"

    def test_real_key_gives_ok(self):
        from reporag.api.routes.health import _check_llm

        with patch("reporag.api.routes.health.settings") as mock_settings:
            mock_settings.active_llm_api_key.get_secret_value.return_value = (
                "sk-real-key"
            )
            mock_settings.llm_provider = "openai"
            result = _check_llm()
        assert result.status == "ok"

    def test_placeholder_sk_your_key_gives_not_configured(self):
        from reporag.api.routes.health import _check_llm

        with patch("reporag.api.routes.health.settings") as mock_settings:
            mock_settings.active_llm_api_key.get_secret_value.return_value = (
                "sk-your-key-here"
            )
            mock_settings.llm_provider = "openai"
            result = _check_llm()
        assert result.status == "not_configured"

    def test_anthropic_placeholder_gives_not_configured(self):
        from reporag.api.routes.health import _check_llm

        with patch("reporag.api.routes.health.settings") as mock_settings:
            mock_settings.active_llm_api_key.get_secret_value.return_value = (
                "sk-ant-your-key-here"
            )
            mock_settings.llm_provider = "anthropic"
            result = _check_llm()
        assert result.status == "not_configured"


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------


class TestOpenAPIDocs:
    async def test_docs_endpoint_returns_200(self, async_client):
        resp = await async_client.get("/docs")
        assert resp.status_code == 200

    async def test_openapi_json_endpoint_returns_schema(self, async_client):
        resp = await async_client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/api/v1/health" in schema["paths"]

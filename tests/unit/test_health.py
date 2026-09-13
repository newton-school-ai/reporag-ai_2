"""Unit tests for the health check endpoints (Issue 26).

Covers both endpoints:
* GET /health is the liveness probe: static, dependency-free and fast.
* GET /api/v1/health is the diagnostic view: probes each dependency.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from reporag.api.routes import health as health_routes
from reporag.api.routes.health import (
    API_VERSION,
    _check_llm,
    _check_neo4j,
    _check_qdrant,
)
from reporag.config import settings as live_settings


def _stub_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    neo4j: str = "ok",
    qdrant: str = "ok",
    llm: str = "ok",
) -> None:
    """Replace the three out-of-process probes with fixed verdicts."""
    for name, verdict in (("neo4j", neo4j), ("qdrant", qdrant), ("llm", llm)):
        monkeypatch.setattr(
            health_routes,
            f"_check_{name}",
            lambda verdict=verdict: health_routes.ComponentHealth(
                status=verdict, detail=""
            ),
        )


class TestLivenessProbe:
    """GET /health - the unversioned, dependency-free probe."""

    def test_reports_ok(self, client: Any) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_does_not_probe_dependencies(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("liveness must not touch a dependency")

        monkeypatch.setattr(health_routes, "_check_neo4j", fail)
        monkeypatch.setattr(health_routes, "_check_qdrant", fail)
        assert client.get("/health").status_code == 200


class TestComponentHealth:
    """GET /api/v1/health - the diagnostic view."""

    def test_reports_every_component(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        body = client.get("/api/v1/health").json()
        assert set(body["components"]) == {"database", "neo4j", "qdrant", "llm"}

    def test_reports_version_and_environment(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        body = client.get("/api/v1/health").json()
        assert body["version"] == API_VERSION
        assert body["environment"] == live_settings.app_env

    def test_all_healthy_reports_ok(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        assert client.get("/api/v1/health").json()["status"] == "ok"

    @pytest.mark.parametrize("failing", ["neo4j", "qdrant", "llm"])
    def test_one_bad_component_degrades_the_whole(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, failing: str
    ) -> None:
        _stub_probes(monkeypatch, **{failing: "error"})
        body = client.get("/api/v1/health").json()
        assert body["status"] == "degraded"
        assert body["components"][failing]["status"] == "error"

    def test_unconfigured_component_also_degrades(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch, llm="not_configured")
        assert client.get("/api/v1/health").json()["status"] == "degraded"

    def test_degraded_still_returns_200(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch, neo4j="error", qdrant="error")
        assert client.get("/api/v1/health").status_code == 200


class TestNeo4jProbe:
    def test_unset_uri_is_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_settings, "neo4j_uri", "")
        result = _check_neo4j()
        assert result.status == "not_configured"
        assert "NEO4J_URI" in result.detail

    def test_unreachable_server_is_an_error_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "neo4j_uri", "bolt://localhost:7687")

        class _Driver:
            def verify_connectivity(self) -> None:
                raise ConnectionError("refused")

            def close(self) -> None:
                pass

        import neo4j

        monkeypatch.setattr(
            neo4j.GraphDatabase, "driver", staticmethod(lambda *a, **k: _Driver())
        )
        result = _check_neo4j()
        assert result.status == "error"
        assert "refused" in result.detail


class TestQdrantProbe:
    def test_unset_url_is_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_settings, "qdrant_url", "")
        assert _check_qdrant().status == "not_configured"

    def test_unreachable_server_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "qdrant_url", "http://localhost:6333")

        import qdrant_client

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("qdrant unreachable")

        monkeypatch.setattr(qdrant_client, "QdrantClient", boom)
        result = _check_qdrant()
        assert result.status == "error"
        assert "unreachable" in result.detail


class TestLlmProbe:
    def test_missing_key_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "llm_provider", "anthropic")
        monkeypatch.setattr(live_settings, "anthropic_api_key", SecretStr(""))
        result = _check_llm()
        assert result.status == "not_configured"
        assert "ANTHROPIC_API_KEY" in result.detail

    def test_placeholder_key_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "llm_provider", "anthropic")
        monkeypatch.setattr(
            live_settings, "anthropic_api_key", SecretStr("sk-ant-your-key-here")
        )
        assert _check_llm().status == "not_configured"

    def test_configured_key_is_ok_and_names_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "llm_provider", "anthropic")
        monkeypatch.setattr(
            live_settings, "anthropic_api_key", SecretStr("sk-ant-real-key")
        )
        result = _check_llm()
        assert result.status == "ok"
        assert live_settings.anthropic_model in result.detail

    def test_no_network_call_is_made(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_settings, "llm_provider", "openai")
        monkeypatch.setattr(live_settings, "openai_api_key", SecretStr("sk-real"))
        import httpx

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the LLM probe must not call the provider")

        monkeypatch.setattr(httpx.Client, "send", forbidden)
        assert _check_llm().status == "ok"

"""Integration tests for the end-to-end query HTTP flow.

Sends a real POST /api/v1/query request through the full ASGI stack and
verifies the complete response envelope -- answer, citations, metadata.

The RAG pipeline (_run_query_pipeline) is mocked so no embedding models,
Qdrant, Neo4j, or LLM API are called. The purpose is to verify the
HTTP-level contract: routing, serialisation, and response shape.
"""

from __future__ import annotations

from unittest.mock import patch

from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult

# ---------------------------------------------------------------------------
# Helpers (mirrors test_api_query._make_pipeline_result)
# ---------------------------------------------------------------------------


def _make_result(
    answer: str = "It authenticates the user via JWT.",
    citations: list[Citation] | None = None,
    success: bool = True,
    error: str | None = None,
) -> dict:
    citations = citations or [
        Citation(
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            valid=True,
            snippet="def login(payload):\n    ...",
        )
    ]
    gen = GenerationResult(
        text=answer,
        success=success,
        error=error,
        model="gpt-4o",
        provider="openai",
        latency_seconds=0.5,
    )
    report = CitationReport(
        citations=citations,
        coverage=1.0,
        valid_count=sum(1 for c in citations if c.valid),
        invalid_count=sum(1 for c in citations if not c.valid),
    )
    return {
        "answered": AnsweredQuery(answer=answer, citations=report, generation=gen),
        "query_type": "simple-lookup",
        "needs_decomposition": False,
        "num_sub_queries": 1,
        "strategies_used": ["bm25"],
        "latency_seconds": 0.5,
    }


# ---------------------------------------------------------------------------
# E2E round-trip
# ---------------------------------------------------------------------------


class TestE2EQueryRoundTrip:
    async def test_full_request_returns_200(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        assert resp.status_code == 200

    async def test_response_envelope_shape(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        data = resp.json()
        assert set(data.keys()) >= {"answer", "citations", "metadata"}

    async def test_answer_text_passed_through(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(answer="JWT is used for authentication."),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        assert resp.json()["answer"] == "JWT is used for authentication."

    async def test_citation_serialised_correctly(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        cit = resp.json()["citations"][0]
        assert cit["file_path"] == "src/auth.py"
        assert cit["start_line"] == 10
        assert cit["end_line"] == 25
        assert cit["valid"] is True

    async def test_metadata_latency_positive(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        assert resp.json()["metadata"]["latency_seconds"] > 0

    async def test_content_type_is_json(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "How does authentication work?"},
            )
        assert "application/json" in resp.headers["content-type"]

    async def test_top_k_forwarded_to_pipeline(self, async_client):
        received = {}

        def capturing_pipeline(query, top_k):
            received["top_k"] = top_k
            return _make_result()

        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            side_effect=capturing_pipeline,
        ):
            await async_client.post(
                "/api/v1/query", json={"query": "hello", "top_k": 15}
            )
        assert received["top_k"] == 15

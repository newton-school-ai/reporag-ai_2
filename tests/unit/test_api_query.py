"""Tests for POST /api/v1/query.

Acceptance criteria exercised:
* Returns {answer, citations, metadata} on success
* Validates request with Pydantic (422 for bad input)
* Pipeline ValueError -> 422
* Pipeline unexpected exception -> 502
* Generation failure path -> 200 with success=False in metadata

The full RAG pipeline (_run_query_pipeline) is replaced with a controllable
mock so no embedding models, Qdrant, Neo4j, or LLM API are contacted.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_result(
    answer: str = "The function authenticates the user.",
    query_type: str = "simple-lookup",
    needs_decomposition: bool = False,
    num_sub_queries: int = 1,
    strategies_used: list[str] | None = None,
    citation_coverage: float = 1.0,
    model: str = "gpt-4o",
    provider: str = "openai",
    latency_seconds: float = 0.25,
    success: bool = True,
    error: str | None = None,
    citations: list[Citation] | None = None,
) -> dict:
    """Build a dict matching what _run_query_pipeline returns."""
    gen_result = GenerationResult(
        text=answer,
        success=success,
        error=error,
        model=model,
        provider=provider,
        latency_seconds=latency_seconds,
    )
    citation_list = citations or []
    report = CitationReport(
        citations=citation_list,
        coverage=citation_coverage,
        valid_count=sum(1 for c in citation_list if c.valid),
        invalid_count=sum(1 for c in citation_list if not c.valid),
    )
    answered = AnsweredQuery(answer=answer, citations=report, generation=gen_result)
    return {
        "answered": answered,
        "query_type": query_type,
        "needs_decomposition": needs_decomposition,
        "num_sub_queries": num_sub_queries,
        "strategies_used": strategies_used or [],
        "latency_seconds": latency_seconds,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/query -- happy path
# ---------------------------------------------------------------------------


class TestQueryHappyPath:
    async def test_returns_200(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert resp.status_code == 200

    async def test_response_has_answer_citations_metadata(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "metadata" in data

    async def test_answer_is_string(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(answer="Some answer text."),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert isinstance(resp.json()["answer"], str)
        assert resp.json()["answer"] == "Some answer text."

    async def test_citations_is_list(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert isinstance(resp.json()["citations"], list)

    async def test_citation_fields_present(self, async_client):
        citation = Citation(
            file_path="src/auth.py", start_line=10, end_line=20, valid=True, snippet="x"
        )
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(citations=[citation]),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        c = resp.json()["citations"][0]
        assert c["file_path"] == "src/auth.py"
        assert c["start_line"] == 10
        assert c["end_line"] == 20
        assert c["valid"] is True
        assert "snippet" in c

    async def test_metadata_fields_present(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        meta = resp.json()["metadata"]
        for field in (
            "query_type",
            "needs_decomposition",
            "num_sub_queries",
            "strategies_used",
            "citation_coverage",
            "model",
            "provider",
            "latency_seconds",
            "success",
            "error",
        ):
            assert field in meta, f"metadata missing field: {field}"

    async def test_metadata_values_match_pipeline(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(
                query_type="multi-hop",
                needs_decomposition=True,
                num_sub_queries=3,
                strategies_used=["bm25", "vector"],
                citation_coverage=0.9,
                model="gpt-4o",
                provider="openai",
                success=True,
            ),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "How are imports resolved?"}
            )
        meta = resp.json()["metadata"]
        assert meta["query_type"] == "multi-hop"
        assert meta["needs_decomposition"] is True
        assert meta["num_sub_queries"] == 3
        assert set(meta["strategies_used"]) == {"bm25", "vector"}
        assert meta["citation_coverage"] == pytest.approx(0.9)
        assert meta["model"] == "gpt-4o"
        assert meta["provider"] == "openai"
        assert meta["success"] is True

    async def test_latency_seconds_is_non_negative_float(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(latency_seconds=0.123),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        latency = resp.json()["metadata"]["latency_seconds"]
        assert isinstance(latency, float)
        assert latency >= 0

    async def test_repository_id_optional_and_accepted(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query",
                json={"query": "What does login_route do?", "repository_id": 42},
            )
        assert resp.status_code == 200

    async def test_repository_id_absent_is_fine(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/query -- top_k validation
# ---------------------------------------------------------------------------


class TestQueryTopK:
    async def test_top_k_default_accepted(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post("/api/v1/query", json={"query": "hello"})
        assert resp.status_code == 200

    async def test_top_k_1_accepted(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "hello", "top_k": 1}
            )
        assert resp.status_code == 200

    async def test_top_k_50_accepted(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "hello", "top_k": 50}
            )
        assert resp.status_code == 200

    async def test_top_k_0_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/v1/query", json={"query": "hello", "top_k": 0}
        )
        assert resp.status_code == 422

    async def test_top_k_51_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/v1/query", json={"query": "hello", "top_k": 51}
        )
        assert resp.status_code == 422

    async def test_top_k_negative_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/v1/query", json={"query": "hello", "top_k": -1}
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/query -- query field validation
# ---------------------------------------------------------------------------


class TestQueryValidation:
    async def test_missing_query_field_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    async def test_empty_query_string_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    async def test_whitespace_only_query_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/query", json={"query": "   "})
        assert resp.status_code == 422

    async def test_single_char_query_accepted(self, async_client):
        """min_length=1, so a single non-whitespace char is valid."""
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(),
        ):
            resp = await async_client.post("/api/v1/query", json={"query": "x"})
        assert resp.status_code == 200

    async def test_non_json_body_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/v1/query",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    async def test_query_is_stripped_before_processing(self, async_client):
        """Whitespace-padded non-blank query is stripped and accepted."""
        calls = []

        def capturing_pipeline(query, top_k):
            calls.append(query)
            return _make_pipeline_result()

        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            side_effect=capturing_pipeline,
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "  what is this?  "}
            )
        assert resp.status_code == 200
        assert calls[0] == "what is this?"


# ---------------------------------------------------------------------------
# POST /api/v1/query -- error paths
# ---------------------------------------------------------------------------


class TestQueryErrorPaths:
    async def test_pipeline_value_error_returns_422(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            side_effect=ValueError("bad input"),
        ):
            resp = await async_client.post("/api/v1/query", json={"query": "what?"})
        assert resp.status_code == 422

    async def test_pipeline_unexpected_error_returns_502(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            side_effect=RuntimeError("pipeline exploded"),
        ):
            resp = await async_client.post("/api/v1/query", json={"query": "what?"})
        assert resp.status_code == 502

    async def test_generation_failure_returns_200_with_success_false(
        self, async_client
    ):
        """Even when LLM fails, the endpoint returns 200 (not a 5xx).
        The failure is surfaced in metadata.success = False."""
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(
                answer="",
                success=False,
                error="LLM returned empty response.",
            ),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["success"] is False
        assert meta["error"] == "LLM returned empty response."

    async def test_generation_failure_answer_is_empty_string(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(answer="", success=False, error="oops"),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert resp.json()["answer"] == ""

    async def test_generation_failure_citations_empty(self, async_client):
        with patch(
            "reporag.api.routes.query._run_query_pipeline",
            return_value=_make_pipeline_result(
                answer="", success=False, error="oops", citations=[]
            ),
        ):
            resp = await async_client.post(
                "/api/v1/query", json={"query": "What does login_route do?"}
            )
        assert resp.json()["citations"] == []

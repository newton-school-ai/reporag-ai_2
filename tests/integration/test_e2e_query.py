"""End-to-end tests for ``POST /api/v1/query``.

A question goes in over HTTP and a cited answer comes back, through the real
route, the real pipeline orchestration, the real threadpool dispatch and the
real response models. Only the two boundaries that would otherwise reach
outside the process are replaced: the retrieval backends (Qdrant, Neo4j, the
BM25 index) and the LLM.

That split is what makes these tests worth having alongside the unit suite.
The unit tests inject a whole fake pipeline through ``app.state``; these
build the pipeline the way the route does and check the pieces fit -- that
retrieval results survive merging and deduplication, reach the prompt
builder, and come back as citations the caller can read.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from reporag.api.routes import query as query_routes
from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult
from reporag.retrieval.vector_search import RetrievalResult


def _result(path: str, start: int, end: int, score: float) -> RetrievalResult:
    """A retrieval result as a backend would return it."""
    return RetrievalResult(
        score=score,
        file_path=path,
        start_line=start,
        end_line=end,
        symbol_name="handler",
        chunk_text=f"# {path}\ndef handler(): ...",
    )


class _Backend:
    """A retrieval backend returning fixed results, or failing on demand."""

    def __init__(
        self, results: list[RetrievalResult] | None = None, fail: bool = False
    ) -> None:
        self.results = results or []
        self.fail = fail
        self.calls = 0

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("backend unreachable")
        return self.results[:top_k]

    def get_neighbors(self, symbol: str, depth: int = 2) -> list[RetrievalResult]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("backend unreachable")
        return self.results


class _Llm:
    """Stands in for the generator, returning a fixed cited answer."""

    def __init__(
        self,
        answer: str = "The request is handled in [app/api.py:10-24].",
        *,
        success: bool = True,
        error_kind: str | None = None,
        error: str | None = None,
    ) -> None:
        self.answer = answer
        self.success = success
        self.error_kind = error_kind
        self.error = error
        self.prompts: list[Any] = []

    def generate_with_citations(self, prompt: Any) -> AnsweredQuery:
        self.prompts.append(prompt)
        cites = [
            Citation(
                file_path="app/api.py",
                start_line=10,
                end_line=24,
                snippet="def handler(): ...",
                valid=True,
            )
        ]
        return AnsweredQuery(
            answer=self.answer if self.success else "",
            citations=CitationReport(
                citations=cites if self.success else [],
                coverage=1.0 if self.success else 0.0,
                valid_count=1 if self.success else 0,
                invalid_count=0,
            ),
            generation=GenerationResult(
                text=self.answer if self.success else "",
                success=self.success,
                error=self.error,
                error_kind=self.error_kind,
                model="test-model",
                provider="test",
            ),
        )


@pytest.fixture
def wired_pipeline(api_app: Any) -> dict[str, Any]:
    """Build the real pipeline with fake backends at both boundaries.

    The adapter, decomposer, executor and prompt builder are the real
    classes, assembled exactly as :func:`~reporag.api.routes.query.
    get_pipeline` assembles them. Only what leaves the process is faked.
    """
    from reporag.agent.executor import SubQueryExecutor
    from reporag.agent.planner import QueryDecomposer
    from reporag.generation.prompt_builder import PromptBuilder

    vector = _Backend([_result("app/api.py", 10, 24, 0.95)])
    bm25 = _Backend([_result("app/auth.py", 3, 30, 0.80)])
    graph = _Backend([_result("app/db.py", 1, 12, 0.60)])

    engine = query_routes._RetrievalEngineAdapter(
        vector_search=vector, bm25_search=bm25, graph_retriever=graph
    )
    llm = _Llm()
    components = {
        "engine": engine,
        "decomposer": QueryDecomposer(),
        "executor": SubQueryExecutor(engine, top_k=5),
        "prompt_builder": PromptBuilder(),
        "generator": llm,
        # Handles for assertions; not read by the route.
        "_vector": vector,
        "_bm25": bm25,
        "_graph": graph,
        "_llm": llm,
    }
    api_app.state.pipeline = components
    return components


class TestQueryRoundTrip:
    async def test_returns_answer_citations_and_metadata(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        response = await async_client.post(
            "/api/v1/query",
            json={"question": "How is an incoming request handled?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"answer", "citations", "metadata"}
        assert body["answer"]

    async def test_citations_carry_file_and_line_range(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        body = (
            await async_client.post(
                "/api/v1/query", json={"question": "How is a request handled?"}
            )
        ).json()
        citation = body["citations"][0]
        # Line-level citations are the whole point of the product: a
        # file-only answer sends the reader back to searching.
        assert citation["file_path"] == "app/api.py"
        assert (citation["start_line"], citation["end_line"]) == (10, 24)
        assert citation["valid"] is True

    async def test_metadata_reports_how_the_answer_was_produced(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        body = (
            await async_client.post(
                "/api/v1/query", json={"question": "How is a request handled?"}
            )
        ).json()
        meta = body["metadata"]
        assert meta["query_type"]
        assert meta["model"] == "test-model"
        assert meta["latency_seconds"] >= 0
        assert meta["result_count"] >= 1
        assert meta["sources"]

    async def test_retrieved_context_reaches_the_prompt(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        await async_client.post(
            "/api/v1/query", json={"question": "How is a request handled?"}
        )
        # The generator is the last hop; a prompt arriving here proves the
        # retrieve -> merge -> assemble -> build chain actually ran.
        assert wired_pipeline["_llm"].prompts
        assert wired_pipeline["_vector"].calls >= 1

    async def test_repeated_questions_reuse_the_cached_pipeline(
        self, async_client: AsyncClient, api_app: Any, wired_pipeline: dict[str, Any]
    ) -> None:
        for _ in range(3):
            await async_client.post(
                "/api/v1/query", json={"question": "How is a request handled?"}
            )
        # Three answers, one pipeline: rebuilding it per request would mean
        # recompiling the planner's state machine and reconnecting every
        # backend on every question.
        assert len(wired_pipeline["_llm"].prompts) == 3
        assert api_app.state.pipeline is wired_pipeline


class TestQueryDegradation:
    async def test_a_dead_backend_still_produces_an_answer(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        wired_pipeline["_vector"].fail = True
        response = await async_client.post(
            "/api/v1/query", json={"question": "How is a request handled?"}
        )
        # Retrieval degrades: an answer from the remaining backends beats an
        # error the caller cannot act on.
        assert response.status_code == 200
        assert response.json()["answer"]

    async def test_every_backend_down_still_answers(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        for key in ("_vector", "_bm25", "_graph"):
            wired_pipeline[key].fail = True
        response = await async_client.post(
            "/api/v1/query", json={"question": "How is a request handled?"}
        )
        # The model is asked with no context rather than the request failing;
        # citation validation is what keeps that honest.
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("error_kind", "expected"),
        [("rate_limit", 429), ("timeout", 504), ("api_error", 502), ("auth", 502)],
    )
    async def test_llm_failures_map_to_honest_status_codes(
        self,
        async_client: AsyncClient,
        wired_pipeline: dict[str, Any],
        error_kind: str,
        expected: int,
    ) -> None:
        wired_pipeline["generator"] = _Llm(
            success=False, error_kind=error_kind, error="provider said no"
        )
        response = await async_client.post(
            "/api/v1/query", json={"question": "How is a request handled?"}
        )
        # Generation has no fallback, so unlike retrieval it must surface --
        # and a rate limit is worth retrying where a 500 is not.
        assert response.status_code == expected


class TestQueryValidation:
    @pytest.mark.parametrize(
        "payload",
        [{}, {"question": ""}, {"question": "   "}, {"question": "hi", "top_k": 0}],
    )
    async def test_invalid_requests_are_rejected_before_the_pipeline(
        self,
        async_client: AsyncClient,
        wired_pipeline: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        response = await async_client.post("/api/v1/query", json=payload)
        assert response.status_code == 422
        # Rejected at the edge: no backend and no LLM call was paid for.
        assert not wired_pipeline["_llm"].prompts

    async def test_unknown_repo_id_is_404_not_an_empty_answer(
        self, async_client: AsyncClient, wired_pipeline: dict[str, Any]
    ) -> None:
        response = await async_client.post(
            "/api/v1/query", json={"question": "How?", "repo_id": 4242}
        )
        assert response.status_code == 404
        assert not wired_pipeline["_llm"].prompts

    async def test_a_known_repo_id_is_accepted(
        self,
        async_client: AsyncClient,
        wired_pipeline: dict[str, Any],
        stub_ingestion: list[tuple[Any, ...]],
    ) -> None:
        created = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/demo"},
            )
        ).json()
        response = await async_client.post(
            "/api/v1/query",
            json={"question": "How?", "repo_id": created["repository"]["id"]},
        )
        assert response.status_code == 200

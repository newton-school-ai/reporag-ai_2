"""End-to-end integration tests for POST /api/v1/query.

Verifies the HTTP query endpoint through the full orchestration pipeline
(decomposer, executor, prompt builder, generator) with fakes only at
the external network boundaries (vector store, graph database, LLM provider).
"""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from reporag.api.routes import query as query_routes
from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult
from reporag.retrieval.vector_search import RetrievalResult


def _sample_result(
    file_path: str, start: int, end: int, score: float = 0.9
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start,
        end_line=end,
        symbol_name="login",
        chunk_text=f"# {file_path}\ndef login(): pass",
    )


class _MockBackend:
    def __init__(
        self, results: list[RetrievalResult] | None = None, fail: bool = False
    ):
        self.results = results or []
        self.fail = fail

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.fail:
            raise ConnectionError("Backend unavailable")
        return self.results[:top_k]

    def get_neighbors(self, symbol: str, depth: int = 2) -> list[RetrievalResult]:
        if self.fail:
            raise ConnectionError("Backend unavailable")
        return self.results


class _MockLlm:
    def __init__(
        self, answer: str = "The auth flow is defined in [src/auth.py:10-25]."
    ):
        self.answer = answer
        self.prompts: list[Any] = []

    def generate_with_citations(self, prompt: Any) -> AnsweredQuery:
        self.prompts.append(prompt)
        cites = [
            Citation(
                file_path="src/auth.py",
                start_line=10,
                end_line=25,
                snippet="def login(): pass",
                valid=True,
            )
        ]
        return AnsweredQuery(
            answer=self.answer,
            citations=CitationReport(
                citations=cites,
                coverage=1.0,
                valid_count=1,
                invalid_count=0,
            ),
            generation=GenerationResult(
                success=True,
                text=self.answer,
                model="gpt-4o",
                provider="openai",
            ),
        )


class TestE2EQueryFlow:
    async def test_e2e_query_execution(
        self, async_client: AsyncClient, api_app: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(query_routes.settings, "enable_reranker", False)
        backend_results = [_sample_result("src/auth.py", 10, 25, 0.95)]
        bm25_mock = _MockBackend(backend_results)
        vector_mock = _MockBackend(backend_results)
        graph_mock = _MockBackend(backend_results)
        llm_mock = _MockLlm()

        engine = query_routes._RetrievalEngineAdapter(
            vector_search=vector_mock,
            bm25_search=bm25_mock,
            graph_retriever=graph_mock,
        )

        from reporag.agent.executor import SubQueryExecutor
        from reporag.agent.planner import QueryDecomposer
        from reporag.generation.prompt_builder import PromptBuilder

        api_app.state.pipeline = {
            "engine": engine,
            "decomposer": QueryDecomposer(),
            "executor": SubQueryExecutor(engine, top_k=5),
            "prompt_builder": PromptBuilder(),
            "generator": llm_mock,
        }

        response = await async_client.post(
            "/api/v1/query",
            json={"question": "Where is the authentication login logic?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "auth flow is defined in [src/auth.py:10-25]" in body["answer"]
        assert len(body["citations"]) == 1
        assert body["citations"][0]["file_path"] == "src/auth.py"
        assert body["citations"][0]["valid"] is True
        assert body["metadata"]["query_type"] != ""
        assert body["metadata"]["latency_seconds"] >= 0

    async def test_e2e_query_with_partial_retrieval_failure(
        self, async_client: AsyncClient, api_app: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(query_routes.settings, "enable_reranker", False)
        # Vector search fails, but BM25 succeeds
        bm25_mock = _MockBackend([_sample_result("src/fallback.py", 1, 10)])
        vector_mock = _MockBackend(fail=True)
        graph_mock = _MockBackend(fail=True)
        llm_mock = _MockLlm("Handled in [src/fallback.py:1-10].")

        engine = query_routes._RetrievalEngineAdapter(
            vector_search=vector_mock,
            bm25_search=bm25_mock,
            graph_retriever=graph_mock,
        )

        from reporag.agent.executor import SubQueryExecutor
        from reporag.agent.planner import QueryDecomposer
        from reporag.generation.prompt_builder import PromptBuilder

        api_app.state.pipeline = {
            "engine": engine,
            "decomposer": QueryDecomposer(),
            "executor": SubQueryExecutor(engine, top_k=5),
            "prompt_builder": PromptBuilder(),
            "generator": llm_mock,
        }

        response = await async_client.post(
            "/api/v1/query",
            json={"question": "Find fallback implementation"},
        )
        # Should gracefully succeed with available BM25 results rather than 500 error
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] != ""

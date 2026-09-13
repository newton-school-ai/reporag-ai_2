"""Unit tests for the FastAPI application routes and middleware (Issue 26).

Covers all acceptance criteria of Issue 26:
- POST /api/v1/repos/ingest triggers async background ingestion
- GET /api/v1/repos returns list with status
- POST /api/v1/query returns {answer, citations, metadata}
- GET /api/v1/health returns pipeline component status
- OpenAPI docs available at /docs and /openapi.json
- Request validation with Pydantic models, clear error responses
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from reporag.api.routes import query as query_routes
from reporag.api.routes import repos as repos_routes
from reporag.api.routes.query import _extract_symbols, _RetrievalEngineAdapter
from reporag.api.routes.repos import _IngestionStats, run_ingestion
from reporag.db.models import (
    IngestionJob,
    JobStatus,
    Repository,
    RepositoryStatus,
    User,
)
from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult
from reporag.retrieval.vector_search import RetrievalResult


class _MockClassification:
    def __init__(self, query_type: str = "multi-hop"):
        self.query_type = query_type
        self.confidence = 0.95


class _MockStep:
    def __init__(self, step_id: str, query: str, depends_on: tuple[str, ...] = ()):
        self.id = step_id
        self.query = query
        self.depends_on = depends_on
        self.strategy = "vector"


class _MockPlan:
    def __init__(
        self, steps: list[_MockStep] | None = None, query_type: str = "multi-hop"
    ):
        self.steps = steps or [_MockStep("step-1", "where is auth?")]
        self.classification = _MockClassification(query_type)
        self.needs_decomposition = len(self.steps) > 1


class _MockDecomposer:
    def __init__(self, plan: _MockPlan | None = None):
        self.plan = plan or _MockPlan()

    def decompose(self, question: str) -> _MockPlan:
        return self.plan


class _MockStepResult:
    def __init__(self, results: list[Any] | None = None, strategy: str = "vector"):
        self.results = results or []
        self.strategy = strategy
        self.context_summary = "Found authentication implementation."


class _MockExecutor:
    def __init__(self, results: list[Any] | None = None):
        self.results = results or [
            RetrievalResult(
                score=0.92,
                file_path="src/auth.py",
                start_line=10,
                end_line=25,
                symbol_name="login",
                chunk_text="def login(): pass",
            )
        ]

    def execute(self, steps: list[Any]) -> dict[str, Any]:
        return {step.id: _MockStepResult(self.results) for step in steps}


class _MockBuiltPrompt:
    def __init__(self):
        self.sections = {"context": "def login(): pass"}
        self.token_count = 150
        self.truncated = False


class _MockPromptBuilder:
    def build_from_results(
        self, question: str, classification: Any, results: Any, sub_answers: Any
    ) -> _MockBuiltPrompt:
        return _MockBuiltPrompt()


class _MockGenerator:
    def __init__(
        self,
        answer: str = "Auth in [src/auth.py:10-25].",
        success: bool = True,
        error_kind: str | None = None,
    ):
        self.answer = answer
        self.success = success
        self.error_kind = error_kind

    def generate_with_citations(self, prompt: Any) -> AnsweredQuery:
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
            answer=self.answer if self.success else "",
            citations=CitationReport(
                citations=cites if self.success else [],
                coverage=1.0 if self.success else 0.0,
                valid_count=1 if self.success else 0,
                invalid_count=0,
            ),
            generation=GenerationResult(
                success=self.success,
                text=self.answer if self.success else "",
                model="gpt-4o",
                provider="openai",
                error_kind=self.error_kind or ("" if self.success else "api_error"),
                error="" if self.success else "Mock generation error",
            ),
        )


def _inject_mock_pipeline(
    app: Any,
    generator: _MockGenerator | None = None,
    decomposer: _MockDecomposer | None = None,
    executor: _MockExecutor | None = None,
) -> None:
    app.state.pipeline = {
        "engine": _RetrievalEngineAdapter(),
        "decomposer": decomposer or _MockDecomposer(),
        "executor": executor or _MockExecutor(),
        "prompt_builder": _MockPromptBuilder(),
        "generator": generator or _MockGenerator(),
    }


# ===========================================================================
# 1. Main Application & Framework Tests
# ===========================================================================


class TestAppFramework:
    def test_liveness_probe(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "RepoRAG API"
        assert body["docs"] == "/docs"

    def test_openapi_docs_available(self, client: TestClient) -> None:
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200

        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        schema = openapi_response.json()
        assert schema["info"]["title"] == "RepoRAG API"
        assert "/api/v1/health" in schema["paths"]
        assert "/api/v1/repos/ingest" in schema["paths"]
        assert "/api/v1/repos" in schema["paths"]
        assert "/api/v1/query" in schema["paths"]

    def test_cors_headers_returned(self, client: TestClient) -> None:
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

    def test_validation_error_shape(self, client: TestClient) -> None:
        response = client.post("/api/v1/repos/ingest", json={})
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert body["status_code"] == 422
        assert len(body["errors"]) > 0

    def test_http_exception_shape(self, client: TestClient) -> None:
        response = client.get("/api/v1/repos/999999")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "http_error"
        assert body["status_code"] == 404
        assert "not found" in body["detail"].lower()


# ===========================================================================
# 2. Repository Ingestion Endpoints Tests
# ===========================================================================


class TestRepositoryRoutes:
    def test_ingest_repository_queues_background_task(
        self, client: TestClient, stub_ingestion: list[Any]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={
                "repo_url": "https://github.com/octocat/Hello-World",
                "branch": "main",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["repository"]["status"] == "queued"
        assert body["job_id"] > 0
        assert body["branch"] == "main"
        assert "queued" in body["message"].lower()

        assert len(stub_ingestion) == 1
        repo_id, repo_url, branch, job_id = stub_ingestion[0]
        assert repo_url == "https://github.com/octocat/Hello-World"
        assert branch == "main"
        assert repo_id == body["repository"]["id"]
        assert job_id == body["job_id"]

    def test_ingest_repository_with_url_alias(
        self, client: TestClient, stub_ingestion: list[Any]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"url": "https://github.com/psf/black"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["repository"]["name"] == "black"
        assert len(stub_ingestion) == 1

    def test_ingest_repository_rejects_non_http_scheme(
        self, client: TestClient, stub_ingestion: list[Any]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "git@github.com:psf/black.git"},
        )
        assert response.status_code == 422
        assert len(stub_ingestion) == 0

    def test_ingest_repository_rejects_empty_url(
        self, client: TestClient, stub_ingestion: list[Any]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "   "},
        )
        assert response.status_code == 422
        assert len(stub_ingestion) == 0

    def test_list_repositories_empty(self, client: TestClient) -> None:
        response = client.get("/api/v1/repos")
        assert response.status_code == 200
        body = response.json()
        assert body["repositories"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_repositories_pagination(
        self, client: TestClient, db_session: Any
    ) -> None:
        user = User(username="testuser", email="test@test.local", hashed_password="!")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        for i in range(5):
            repo = Repository(
                owner_id=user.id,
                name=f"repo-{i}",
                url=f"https://github.com/org/repo-{i}",
                status=RepositoryStatus.READY,
            )
            db_session.add(repo)
        await db_session.commit()

        # Test limit
        response = client.get("/api/v1/repos?limit=2&offset=0")
        assert response.status_code == 200
        body = response.json()
        assert len(body["repositories"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 0

        # Test offset
        response_page2 = client.get("/api/v1/repos?limit=2&offset=2")
        assert response_page2.status_code == 200
        body_page2 = response_page2.json()
        assert len(body_page2["repositories"]) == 2
        assert body_page2["offset"] == 2

    @pytest.mark.asyncio
    async def test_get_repository_by_id(
        self, client: TestClient, db_session: Any
    ) -> None:
        user = User(username="testuser2", email="test2@test.local", hashed_password="!")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        repo = Repository(
            owner_id=user.id,
            name="demo-repo",
            url="https://github.com/org/demo-repo",
            status=RepositoryStatus.PROCESSING,
        )
        db_session.add(repo)
        await db_session.commit()
        await db_session.refresh(repo)

        response = client.get(f"/api/v1/repos/{repo.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == repo.id
        assert body["name"] == "demo-repo"
        assert body["status"] == "processing"


class TestBackgroundIngestionWorker:
    @pytest.mark.asyncio
    async def test_successful_ingestion_lifecycle(
        self, api_app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            repos_routes, "async_session_maker", api_app.state.session_maker
        )

        # Seed repository and job
        async with api_app.state.session_maker() as session:
            user = User(username="sys", email="sys@test.local", hashed_password="!")
            session.add(user)
            await session.commit()
            await session.refresh(user)

            repo = Repository(
                owner_id=user.id, name="repo1", url="https://github.com/org/repo1"
            )
            session.add(repo)
            await session.flush()
            job = IngestionJob(repository_id=repo.id)
            session.add(job)
            await session.commit()
            repo_id, job_id = repo.id, job.id

        # Mock parse to succeed with 3 files
        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: _IngestionStats(
                files=3, symbols=15, chunks=20, languages=["python"]
            ),
        )

        await run_ingestion(repo_id, "https://github.com/org/repo1", "main", job_id)

        async with api_app.state.session_maker() as session:
            updated_repo = await session.get(Repository, repo_id)
            updated_job = await session.get(IngestionJob, job_id)
            assert updated_repo.status == RepositoryStatus.READY
            assert updated_job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingestion_fails_when_zero_files(
        self, api_app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            repos_routes, "async_session_maker", api_app.state.session_maker
        )

        async with api_app.state.session_maker() as session:
            user = User(username="sys2", email="sys2@test.local", hashed_password="!")
            session.add(user)
            await session.commit()
            repo = Repository(
                owner_id=user.id,
                name="repo_empty",
                url="https://github.com/org/repo_empty",
            )
            session.add(repo)
            await session.flush()
            job = IngestionJob(repository_id=repo.id)
            session.add(job)
            await session.commit()
            repo_id, job_id = repo.id, job.id

        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: _IngestionStats(files=0),
        )

        await run_ingestion(repo_id, "https://github.com/org/repo_empty", None, job_id)

        async with api_app.state.session_maker() as session:
            updated_repo = await session.get(Repository, repo_id)
            updated_job = await session.get(IngestionJob, job_id)
            assert updated_repo.status == RepositoryStatus.FAILED
            assert updated_job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_ingestion_fails_on_exception(
        self, api_app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            repos_routes, "async_session_maker", api_app.state.session_maker
        )

        async with api_app.state.session_maker() as session:
            user = User(username="sys3", email="sys3@test.local", hashed_password="!")
            session.add(user)
            await session.commit()
            repo = Repository(
                owner_id=user.id, name="repo_err", url="https://github.com/org/repo_err"
            )
            session.add(repo)
            await session.flush()
            job = IngestionJob(repository_id=repo.id)
            session.add(job)
            await session.commit()
            repo_id, job_id = repo.id, job.id

        def blow_up(url: str, branch: Any) -> Any:
            raise RuntimeError("Clone failed")

        monkeypatch.setattr(repos_routes, "_parse_repository", blow_up)

        await run_ingestion(repo_id, "https://github.com/org/repo_err", None, job_id)

        async with api_app.state.session_maker() as session:
            updated_repo = await session.get(Repository, repo_id)
            updated_job = await session.get(IngestionJob, job_id)
            assert updated_repo.status == RepositoryStatus.FAILED
            assert updated_job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_or_create_system_user_handles_race_condition(
        self, api_app: Any
    ) -> None:
        async with api_app.state.session_maker() as session:
            user1 = await repos_routes._get_or_create_system_user(session)
            assert user1.username == repos_routes._SYSTEM_USERNAME

            user2 = await repos_routes._get_or_create_system_user(session)
            assert user2.id == user1.id

    def test_parse_repository_resolves_cloned_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from reporag.ingestion.cloner import FileEntry

        clone_dir = tmp_path / "clone_test"
        clone_dir.mkdir()
        test_file = clone_dir / "sample.py"
        test_file.write_text("def foo(): pass", encoding="utf-8")

        class MockCloner:
            last_clone_path = clone_dir

            def clone_and_discover(
                self, repo_url: str, branch: str | None = None
            ) -> list[FileEntry]:
                return [FileEntry(path="sample.py", language="python", size_bytes=16)]

            def cleanup(self) -> None:
                pass

        extracted_paths: list[str] = []

        class MockExtractor:
            def extract_from_file(
                self, file_path: str, language: str | None = None
            ) -> list[Any]:
                extracted_paths.append(file_path)
                return [1]

        class MockChunker:
            def chunk_file(
                self, file_path: str, language: str | None = None
            ) -> list[Any]:
                return [1]

        monkeypatch.setattr("reporag.ingestion.cloner.RepoCloner", MockCloner)
        monkeypatch.setattr(
            "reporag.ingestion.symbol_extractor.SymbolExtractor", MockExtractor
        )
        monkeypatch.setattr("reporag.ingestion.chunker.SemanticChunker", MockChunker)

        stats = repos_routes._parse_repository("https://github.com/org/repo", "main")
        assert stats.files == 1
        assert stats.symbols == 1
        assert stats.chunks == 1
        assert len(extracted_paths) == 1
        assert extracted_paths[0] == (clone_dir / "sample.py").as_posix()


# ===========================================================================
# 3. Query Endpoint Tests
# ===========================================================================


class TestQueryRoute:
    def test_query_success(self, api_app: Any, client: TestClient) -> None:
        _inject_mock_pipeline(api_app)

        response = client.post(
            "/api/v1/query",
            json={"question": "How does authentication work?", "top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert "Auth in [src/auth.py:10-25]" in body["answer"]
        assert len(body["citations"]) == 1
        citation = body["citations"][0]
        assert citation["file_path"] == "src/auth.py"
        assert citation["start_line"] == 10
        assert citation["end_line"] == 25
        assert citation["valid"] is True
        assert body["metadata"]["query_type"] == "multi-hop"
        assert body["metadata"]["result_count"] > 0
        assert body["metadata"]["citation_coverage"] == 1.0

    def test_query_with_query_alias(self, api_app: Any, client: TestClient) -> None:
        _inject_mock_pipeline(api_app)

        response = client.post(
            "/api/v1/query",
            json={"query": "Where is the router?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] != ""

    def test_query_rejects_blank_question(self, client: TestClient) -> None:
        response = client.post("/api/v1/query", json={"question": "    "})
        assert response.status_code == 422

    def test_query_rejects_nonexistent_repo_id(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/query",
            json={"question": "Where is config?", "repo_id": 999999},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_query_accepts_valid_repo_id(
        self, api_app: Any, client: TestClient, db_session: Any
    ) -> None:
        _inject_mock_pipeline(api_app)

        user = User(username="q_user", email="q@test.local", hashed_password="!")
        db_session.add(user)
        await db_session.commit()

        repo = Repository(
            owner_id=user.id, name="q_repo", url="https://github.com/org/q_repo"
        )
        db_session.add(repo)
        await db_session.commit()
        await db_session.refresh(repo)

        response = client.post(
            "/api/v1/query",
            json={"question": "How to query?", "repo_id": repo.id},
        )
        assert response.status_code == 200

    def test_query_rate_limit_mapped_to_429(
        self, api_app: Any, client: TestClient
    ) -> None:
        _inject_mock_pipeline(
            api_app, generator=_MockGenerator(success=False, error_kind="rate_limit")
        )
        response = client.post("/api/v1/query", json={"question": "Any question"})
        assert response.status_code == 429

    def test_query_timeout_mapped_to_504(
        self, api_app: Any, client: TestClient
    ) -> None:
        _inject_mock_pipeline(
            api_app, generator=_MockGenerator(success=False, error_kind="timeout")
        )
        response = client.post("/api/v1/query", json={"question": "Any question"})
        assert response.status_code == 504

    def test_query_auth_or_api_error_mapped_to_502(
        self, api_app: Any, client: TestClient
    ) -> None:
        _inject_mock_pipeline(
            api_app, generator=_MockGenerator(success=False, error_kind="auth")
        )
        response = client.post("/api/v1/query", json={"question": "Any question"})
        assert response.status_code == 502

    def test_symbol_extraction_helper(self) -> None:
        symbols = _extract_symbols("Where is auth_service.loginUser defined?")
        assert "auth_service.loginUser" in symbols

        # Plain prose words without underscore or camelCase should not be treated as code symbols
        prose_symbols = _extract_symbols("what does this function do")
        assert len(prose_symbols) == 0

    def test_retrieval_adapter_graceful_degradation(self) -> None:
        class _FailingSearch:
            def search(self, *a: Any, **k: Any) -> Any:
                raise ConnectionError("Qdrant down")

        adapter = _RetrievalEngineAdapter(vector_search=_FailingSearch())
        results = adapter.search_vector("test query", 10)
        assert results == []

    def test_query_pipeline_construction_failure_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_constructor_err(*a: Any, **k: Any) -> Any:
            raise RuntimeError("Missing LLM API keys")

        monkeypatch.setattr(
            query_routes, "_RetrievalEngineAdapter", raise_constructor_err
        )
        response = client.post("/api/v1/query", json={"question": "Any question"})
        assert response.status_code == 503
        assert "not available" in response.json()["detail"].lower()

    def test_query_with_invalid_citations(
        self, api_app: Any, client: TestClient
    ) -> None:
        class _MockGeneratorWithInvalidCitation:
            def generate_with_citations(self, prompt: Any) -> AnsweredQuery:
                cites = [
                    Citation(
                        file_path="src/nonexistent.py",
                        start_line=1,
                        end_line=5,
                        snippet="",
                        valid=False,
                    )
                ]
                return AnsweredQuery(
                    answer="Hallucinated claim in [src/nonexistent.py:1-5].",
                    citations=CitationReport(
                        citations=cites,
                        coverage=0.5,
                        valid_count=0,
                        invalid_count=1,
                    ),
                    generation=GenerationResult(
                        success=True,
                        text="Hallucinated claim in [src/nonexistent.py:1-5].",
                        model="gpt-4o",
                        provider="openai",
                    ),
                )

        _inject_mock_pipeline(api_app, generator=_MockGeneratorWithInvalidCitation())
        response = client.post(
            "/api/v1/query", json={"question": "Where is non-existent code?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["citations"]) == 1
        assert body["citations"][0]["valid"] is False
        assert body["metadata"]["invalid_citations"] == 1
        assert body["metadata"]["valid_citations"] == 0

    def test_ingest_with_explicit_name(
        self, client: TestClient, stub_ingestion: list[Any]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={
                "repo_url": "https://github.com/pallets/flask",
                "name": "Custom-Flask-Name",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["repository"]["name"] == "Custom-Flask-Name"

    def test_unhandled_exception_returns_500_without_leak(
        self, api_app: Any, client: TestClient
    ) -> None:
        def raise_unhandled(*a: Any, **k: Any) -> Any:
            raise RuntimeError(
                "Secret DB connection string: postgres://user:secret@db/prod"
            )

        api_app.state.pipeline = {
            "engine": None,
            "decomposer": None,
            "executor": None,
            "prompt_builder": None,
            "generator": None,
        }
        # Simulate an unhandled exception inside a route handler
        from fastapi import APIRouter

        faulty_router = APIRouter()

        @faulty_router.get("/api/v1/faulty")
        def faulty():
            raise_unhandled()

        api_app.include_router(faulty_router)

        with TestClient(api_app, raise_server_exceptions=False) as err_client:
            response = err_client.get("/api/v1/faulty")
            assert response.status_code == 500
            body = response.json()
            assert body["error"] == "internal_error"
            assert body["status_code"] == 500
            assert "secret" not in body["detail"].lower()
            assert body["detail"] == "An unexpected error occurred."

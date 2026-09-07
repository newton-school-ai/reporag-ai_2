"""Unit tests for the FastAPI application (Issue 26).

Covers every acceptance criterion of Issue 26:

* ``POST /api/v1/repos/ingest`` triggers async background ingestion.
* ``GET /api/v1/repos`` returns the list with status
  (queued / cloning / processing / ready / failed).
* ``POST /api/v1/query`` returns ``{answer, citations, metadata}``.
* ``GET /api/v1/health`` returns the status of each component
  (neo4j, qdrant, llm, database).
* OpenAPI docs are available at ``/docs``.
* Request validation uses Pydantic models and produces clear error responses.

Beyond the acceptance criteria, the suite pins the design contract:

* the unversioned ``GET /health`` liveness probe keeps its original shape,
* pipeline components are built once and cached on ``app.state``,
* LLM failure categories map to honest HTTP status codes,
* retrieval backend failures degrade instead of failing the request,
* unhandled exceptions never leak a traceback to the caller.

No test makes a real network, database, LLM or Hugging Face call: the
database is a temporary SQLite file and every pipeline component is a
hand-written fake injected through ``app.state`` or ``dependency_overrides``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from reporag.api.main import create_app
from reporag.api.routes import query as query_routes
from reporag.api.routes import repos as repos_routes
from reporag.api.routes.repos import run_ingestion as real_run_ingestion
from reporag.db.models import Base, Repository, RepositoryStatus, User
from reporag.db.session import get_db
from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStep:
    """Stands in for a planner ``DecompositionStep``."""

    def __init__(self, step_id: str, query: str) -> None:
        self.id = step_id
        self.query = query


class _FakeClassification:
    """Stands in for a planner ``ClassificationResult``."""

    def __init__(self, query_type: str = "multi-hop") -> None:
        self.query_type = query_type
        self.confidence = 0.9


class _FakePlan:
    """Stands in for a planner ``DecompositionPlan``."""

    def __init__(self, steps: list[_FakeStep], query_type: str = "multi-hop") -> None:
        self.steps = steps
        self.classification = _FakeClassification(query_type)


class _FakeDecomposer:
    """Returns a canned plan and records the questions it was asked."""

    def __init__(self, plan: _FakePlan | None = None) -> None:
        self.plan = plan or _FakePlan([_FakeStep("s1", "locate the entry point")])
        self.calls: list[str] = []

    def decompose(self, query: str, repo_context: Any = None) -> _FakePlan:
        self.calls.append(query)
        return self.plan


class _FakeStepResult:
    """Stands in for an executor ``StepResult``."""

    def __init__(
        self,
        strategy: str = "hybrid",
        results: list[RetrievalResult] | None = None,
        context_summary: str = "",
    ) -> None:
        self.strategy = strategy
        self.results = results or []
        self.context_summary = context_summary


class _FakeExecutor:
    """Returns canned step results without touching a retrieval backend."""

    def __init__(self, step_results: dict[str, _FakeStepResult] | None = None) -> None:
        self.step_results = step_results if step_results is not None else {}
        self.calls = 0

    def execute(self, steps: Any) -> dict[str, _FakeStepResult]:
        self.calls += 1
        return self.step_results


class _FakePrompt:
    """Stands in for a ``BuiltPrompt``."""

    def __init__(self, token_count: int = 512, truncated: bool = False) -> None:
        self.token_count = token_count
        self.truncated = truncated
        self.text = "prompt"


class _FakePromptBuilder:
    """Records what it was asked to assemble and returns a fixed prompt."""

    def __init__(self, prompt: _FakePrompt | None = None) -> None:
        self.prompt = prompt or _FakePrompt()
        self.last_results: list[RetrievalResult] = []
        self.last_sub_answers: list[Any] = []

    def build_from_results(
        self, query: str, query_type: Any, results: Any, sub_query_answers: Any
    ) -> _FakePrompt:
        self.last_results = list(results)
        self.last_sub_answers = list(sub_query_answers or [])
        return self.prompt


def _answered(
    text: str = "The flow starts in [app.py:1-5].",
    *,
    success: bool = True,
    error: str | None = None,
    error_kind: str | None = None,
    citations: list[Citation] | None = None,
) -> AnsweredQuery:
    """Build an ``AnsweredQuery`` the way the real generator would."""
    cites = (
        citations
        if citations is not None
        else [
            Citation(
                file_path="app.py", start_line=1, end_line=5, snippet="x", valid=True
            )
        ]
    )
    report = CitationReport(
        citations=cites,
        coverage=1.0,
        valid_count=sum(1 for c in cites if c.valid),
        invalid_count=sum(1 for c in cites if c.valid is False),
    )
    return AnsweredQuery(
        answer=text if success else "",
        citations=report,
        generation=GenerationResult(
            text=text if success else "",
            success=success,
            error=error,
            error_kind=error_kind,
            model="fake-model",
            provider="fake",
        ),
    )


class _FakeGenerator:
    """Returns a canned ``AnsweredQuery``."""

    def __init__(self, answered: AnsweredQuery | None = None) -> None:
        self.answered = answered or _answered()
        self.calls = 0

    def generate_with_citations(
        self, prompt: Any, context: Any = None
    ) -> AnsweredQuery:
        self.calls += 1
        return self.answered


def _result(
    file_path: str = "app.py",
    start: int = 1,
    end: int = 5,
    score: float = 0.9,
) -> RetrievalResult:
    """Build a ``RetrievalResult`` for use in fakes."""
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start,
        end_line=end,
        symbol_name="handler",
        chunk_text="def handler(): ...",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_url(tmp_path: Any) -> str:
    """Create a real SQLite file with the schema applied.

    A file rather than ``:memory:`` so the schema created here is visible to
    the async engine the app uses; an in-memory database would be private to
    the connection that created it.
    """
    path = tmp_path / "test.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    return f"sqlite+aiosqlite:///{path}"


@pytest.fixture
def app(db_url: str) -> Any:
    """Build an isolated app wired to the temporary database.

    ``create_app`` is used rather than the module-level ``app`` because the
    query route caches pipeline components on ``app.state``; a shared
    instance would leak one test's fakes into the next.
    """
    application = create_app()
    engine = create_async_engine(db_url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> Any:
        async with session_maker() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db
    application.state.session_maker = session_maker
    return application


@pytest.fixture
def client(app: Any) -> Any:
    """A ``TestClient`` bound to the isolated app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pipeline(app: Any) -> dict[str, Any]:
    """Install fake pipeline components and return them for assertions."""
    components = {
        "engine": object(),
        "decomposer": _FakeDecomposer(),
        "executor": _FakeExecutor(),
        "prompt_builder": _FakePromptBuilder(),
        "generator": _FakeGenerator(),
    }
    app.state.pipeline = components
    return components


@pytest.fixture(autouse=True)
def _no_real_ingestion(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace the background ingestion task so no test clones a repository."""
    calls: list[tuple[Any, ...]] = []

    async def fake_run_ingestion(
        repo_id: int, repo_url: str, branch: str | None
    ) -> None:
        calls.append((repo_id, repo_url, branch))

    monkeypatch.setattr(repos_routes, "run_ingestion", fake_run_ingestion)
    return calls


# ---------------------------------------------------------------------------
# Liveness probe (pre-existing contract)
# ---------------------------------------------------------------------------


class TestLivenessProbe:
    def test_unversioned_health_keeps_its_original_shape(self, client: Any) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_liveness_does_not_touch_dependencies(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A probe that consulted Neo4j would fail here; it must not.
        def explode() -> Any:
            raise AssertionError("liveness probed a dependency")

        monkeypatch.setattr("reporag.api.routes.health._check_neo4j", explode)
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Component health
# ---------------------------------------------------------------------------


def _stub_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    neo4j: str = "ok",
    qdrant: str = "ok",
    llm: str = "ok",
) -> None:
    """Force each external probe to a chosen status without any I/O."""
    from reporag.api.routes.health import ComponentHealth

    monkeypatch.setattr(
        "reporag.api.routes.health._check_neo4j",
        lambda: ComponentHealth(status=neo4j, detail="stub"),
    )
    monkeypatch.setattr(
        "reporag.api.routes.health._check_qdrant",
        lambda: ComponentHealth(status=qdrant, detail="stub"),
    )
    monkeypatch.setattr(
        "reporag.api.routes.health._check_llm",
        lambda: ComponentHealth(status=llm, detail="stub"),
    )


class TestComponentHealth:
    def test_reports_every_component(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        body = client.get("/api/v1/health").json()
        assert set(body["components"]) == {"database", "neo4j", "qdrant", "llm"}

    def test_all_healthy_reports_ok(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["components"]["database"]["status"] == "ok"

    @pytest.mark.parametrize(
        ("component", "state"),
        [("neo4j", "error"), ("qdrant", "error"), ("llm", "not_configured")],
    )
    def test_any_unhealthy_component_degrades_overall(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        component: str,
        state: str,
    ) -> None:
        _stub_probes(monkeypatch, **{component: state})
        body = client.get("/api/v1/health").json()
        assert body["status"] == "degraded"
        assert body["components"][component]["status"] == state

    def test_degraded_still_returns_200(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dead dependency must not take the whole node out of rotation.
        _stub_probes(monkeypatch, neo4j="error", qdrant="error")
        assert client.get("/api/v1/health").status_code == 200

    def test_reports_version_and_environment(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        body = client.get("/api/v1/health").json()
        assert body["version"]
        assert body["environment"]

    def test_probe_failure_is_reported_not_raised(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qdrant_client import QdrantClient

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("qdrant unreachable")

        monkeypatch.setattr(QdrantClient, "__init__", boom)
        monkeypatch.setattr(
            "reporag.api.routes.health._check_neo4j",
            lambda: __import__(
                "reporag.api.routes.health", fromlist=["ComponentHealth"]
            ).ComponentHealth(status="ok", detail="stub"),
        )
        body = client.get("/api/v1/health").json()
        assert body["components"]["qdrant"]["status"] == "error"
        assert "unreachable" in body["components"]["qdrant"]["detail"]

    @pytest.mark.parametrize(
        "key", ["", "   ", "sk-your-key-here", "sk-ant-your-key-here", "change-me"]
    )
    def test_placeholder_llm_key_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch, key: str
    ) -> None:
        # An unedited .env copied from .env.example must not report healthy;
        # otherwise the first real query is what discovers the missing key.
        from pydantic import SecretStr

        from reporag.api.routes.health import _check_llm
        from reporag.config import settings as live_settings

        monkeypatch.setattr(live_settings, "llm_provider", "openai")
        monkeypatch.setattr(live_settings, "openai_api_key", SecretStr(key))
        assert _check_llm().status == "not_configured"

    def test_real_llm_key_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr

        from reporag.api.routes.health import _check_llm
        from reporag.config import settings as live_settings

        monkeypatch.setattr(live_settings, "llm_provider", "openai")
        monkeypatch.setattr(live_settings, "openai_api_key", SecretStr("sk-real-abc"))
        assert _check_llm().status == "ok"

    def test_database_probe_failure_is_reported(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def broken_db() -> Any:
            raise ConnectionError("db down")
            yield  # pragma: no cover

        _stub_probes(monkeypatch)
        app.dependency_overrides[get_db] = broken_db
        # raise_server_exceptions=False so the app's own 500 handler renders
        # the response instead of TestClient re-raising into the test.
        with TestClient(app, raise_server_exceptions=False) as broken_client:
            # The dependency itself fails, so the shared error shape applies.
            response = broken_client.get("/api/v1/health")
        assert response.status_code == 500
        assert response.json()["error"] == "internal_error"


# ---------------------------------------------------------------------------
# Repository ingestion
# ---------------------------------------------------------------------------


class TestIngestRepository:
    def test_returns_202_and_queued_status(self, client: Any) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click", "branch": "main"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["repository"]["status"] == RepositoryStatus.QUEUED.value
        assert body["branch"] == "main"

    def test_schedules_the_background_task(
        self, client: Any, _no_real_ingestion: list[tuple[Any, ...]]
    ) -> None:
        client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click", "branch": "main"},
        )
        assert len(_no_real_ingestion) == 1
        _, url, branch = _no_real_ingestion[0]
        assert url == "https://github.com/pallets/click"
        assert branch == "main"

    def test_derives_repository_name_from_url(self, client: Any) -> None:
        body = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click.git"},
        ).json()
        assert body["repository"]["name"] == "click"

    def test_branch_is_optional(self, client: Any) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        )
        assert response.status_code == 202
        assert response.json()["branch"] is None

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"repo_url": ""},
            {"repo_url": "   "},
            {"repo_url": "git@github.com:pallets/click.git"},
            {"repo_url": "ftp://example.com/repo"},
            {"repo_url": "https://github.com/x/y", "branch": "has space"},
        ],
    )
    def test_invalid_payloads_are_rejected(
        self, client: Any, payload: dict[str, Any]
    ) -> None:
        response = client.post("/api/v1/repos/ingest", json=payload)
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_validation_error_names_the_field(self, client: Any) -> None:
        body = client.post("/api/v1/repos/ingest", json={}).json()
        assert any("repo_url" in err["field"] for err in body["errors"])


class TestListRepositories:
    def test_empty_by_default(self, client: Any) -> None:
        assert client.get("/api/v1/repos").json() == []

    def test_lists_ingested_repositories_with_status(self, client: Any) -> None:
        client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        )
        client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/psf/requests"},
        )
        body = client.get("/api/v1/repos").json()
        assert [r["name"] for r in body] == ["click", "requests"]
        assert all(r["status"] == RepositoryStatus.QUEUED.value for r in body)

    @pytest.mark.parametrize(
        "state",
        [
            RepositoryStatus.QUEUED,
            RepositoryStatus.CLONING,
            RepositoryStatus.PROCESSING,
            RepositoryStatus.READY,
            RepositoryStatus.FAILED,
        ],
    )
    def test_every_status_round_trips(
        self, app: Any, client: Any, state: RepositoryStatus
    ) -> None:
        import anyio

        async def seed() -> None:
            async with app.state.session_maker() as session:
                user = User(username="u", email="u@example.com", hashed_password="!")
                session.add(user)
                await session.flush()
                session.add(
                    Repository(
                        owner_id=user.id,
                        name="demo",
                        url="https://example.com/demo",
                        status=state,
                    )
                )
                await session.commit()

        anyio.run(seed)
        body = client.get("/api/v1/repos").json()
        assert body[0]["status"] == state.value


class TestGetRepository:
    def test_returns_the_repository(self, client: Any) -> None:
        created = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        ).json()["repository"]
        body = client.get(f"/api/v1/repos/{created['id']}").json()
        assert body["id"] == created["id"]
        assert body["name"] == "click"

    def test_unknown_id_returns_404(self, client: Any) -> None:
        response = client.get("/api/v1/repos/9999")
        assert response.status_code == 404
        assert response.json() == {
            "error": "http_error",
            "detail": "Repository 9999 not found",
            "status_code": 404,
        }

    def test_non_positive_id_is_rejected(self, client: Any) -> None:
        assert client.get("/api/v1/repos/0").status_code == 422


class TestBackgroundIngestion:
    """The background task's status machine and failure handling.

    ``real_run_ingestion`` is used rather than the module attribute, which
    the autouse fixture replaces to keep every other test from cloning.
    """

    @staticmethod
    async def _seed_repo(app: Any) -> int:
        """Insert a queued repository and return its id."""
        async with app.state.session_maker() as session:
            user = User(username="u", email="u@example.com", hashed_password="!")
            session.add(user)
            await session.flush()
            repo = Repository(
                owner_id=user.id,
                name="demo",
                url="https://example.com/demo",
                status=RepositoryStatus.QUEUED,
            )
            session.add(repo)
            await session.commit()
            return repo.id

    @staticmethod
    async def _status(app: Any, repo_id: int) -> RepositoryStatus:
        """Read a repository's current status."""
        async with app.state.session_maker() as session:
            repo = await session.get(Repository, repo_id)
            return repo.status

    async def test_successful_run_ends_ready(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_id = await self._seed_repo(app)
        monkeypatch.setattr(
            repos_routes, "async_session_maker", app.state.session_maker
        )
        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: repos_routes._IngestionStats(
                files=3, symbols=10, chunks=12, languages=["python"]
            ),
        )
        await real_run_ingestion(repo_id, "https://example.com/demo", None)
        assert await self._status(app, repo_id) == RepositoryStatus.READY

    async def test_repository_with_no_source_files_fails(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_id = await self._seed_repo(app)
        monkeypatch.setattr(
            repos_routes, "async_session_maker", app.state.session_maker
        )
        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: repos_routes._IngestionStats(files=0),
        )
        await real_run_ingestion(repo_id, "https://example.com/demo", None)
        assert await self._status(app, repo_id) == RepositoryStatus.FAILED

    async def test_clone_failure_is_recorded_not_raised(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_id = await self._seed_repo(app)
        monkeypatch.setattr(
            repos_routes, "async_session_maker", app.state.session_maker
        )

        def boom(url: str, branch: str | None) -> Any:
            raise RuntimeError("git clone failed")

        monkeypatch.setattr(repos_routes, "_parse_repository", boom)
        # Nothing is listening to a background task, so it must never raise.
        await real_run_ingestion(repo_id, "https://example.com/demo", None)
        assert await self._status(app, repo_id) == RepositoryStatus.FAILED

    async def test_progresses_through_cloning_and_processing(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_id = await self._seed_repo(app)
        monkeypatch.setattr(
            repos_routes, "async_session_maker", app.state.session_maker
        )
        seen: list[RepositoryStatus] = []
        original = repos_routes._set_status

        async def recording(
            rid: int, new_status: RepositoryStatus, *, detail: str = ""
        ) -> None:
            seen.append(new_status)
            await original(rid, new_status, detail=detail)

        monkeypatch.setattr(repos_routes, "_set_status", recording)
        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: repos_routes._IngestionStats(files=2, symbols=4),
        )
        await real_run_ingestion(repo_id, "https://example.com/demo", None)
        assert seen == [
            RepositoryStatus.CLONING,
            RepositoryStatus.PROCESSING,
            RepositoryStatus.READY,
        ]

    async def test_missing_repository_row_is_survivable(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            repos_routes, "async_session_maker", app.state.session_maker
        )
        monkeypatch.setattr(
            repos_routes,
            "_parse_repository",
            lambda url, branch: repos_routes._IngestionStats(files=1),
        )
        # A repository deleted mid-ingest must not crash the task.
        await real_run_ingestion(4242, "https://example.com/demo", None)


class TestIngestionStats:
    def test_summary_reports_every_count(self) -> None:
        stats = repos_routes._IngestionStats(
            files=3, symbols=10, chunks=12, languages=["python"], skipped=1
        )
        summary = stats.summary()
        assert "3 file(s)" in summary
        assert "10 symbol(s)" in summary
        assert "12 chunk(s)" in summary
        assert "1 skipped" in summary
        assert "python" in summary

    def test_summary_handles_no_languages(self) -> None:
        assert "none" in repos_routes._IngestionStats().summary()


class TestRepoNameDerivation:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/pallets/click", "click"),
            ("https://github.com/pallets/click.git", "click"),
            ("https://github.com/pallets/click/", "click"),
            ("https://example.com", "example.com"),
        ],
    )
    def test_name_from_url(self, url: str, expected: str) -> None:
        assert repos_routes._repo_name_from_url(url) == expected


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_returns_answer_citations_and_metadata(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        response = client.post(
            "/api/v1/query", json={"question": "How does auth work?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"answer", "citations", "metadata"}
        assert body["answer"] == "The flow starts in [app.py:1-5]."

    def test_citations_carry_file_and_line_range(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        body = client.post("/api/v1/query", json={"question": "How?"}).json()
        assert body["citations"] == [
            {
                "file_path": "app.py",
                "start_line": 1,
                "end_line": 5,
                "snippet": "x",
                "valid": True,
            }
        ]

    def test_metadata_reports_pipeline_facts(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        pipeline["executor"].step_results = {
            "s1": _FakeStepResult(
                strategy="graph",
                results=[_result("auth.py", 10, 20)],
                context_summary="found the verifier",
            )
        }
        meta = client.post("/api/v1/query", json={"question": "How?"}).json()[
            "metadata"
        ]
        assert meta["query_type"] == "multi-hop"
        assert meta["strategies"] == {"s1": "graph"}
        assert meta["sources"] == ["auth.py"]
        assert meta["result_count"] == 1
        assert meta["model"] == "fake-model"
        assert meta["sub_queries"] == ["locate the entry point"]

    def test_invalid_citations_are_reported_not_hidden(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        pipeline["generator"].answered = _answered(
            citations=[
                Citation(file_path="ghost.py", start_line=1, end_line=2, valid=False)
            ]
        )
        body = client.post("/api/v1/query", json={"question": "How?"}).json()
        assert body["citations"][0]["valid"] is False
        assert body["metadata"]["invalid_citations"] == 1

    def test_deduplicates_results_across_steps_keeping_best_score(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        shared_low = _result("auth.py", 1, 9, score=0.2)
        shared_high = _result("auth.py", 1, 9, score=0.8)
        pipeline["executor"].step_results = {
            "s1": _FakeStepResult(results=[shared_low]),
            "s2": _FakeStepResult(results=[shared_high, _result("db.py", 1, 4)]),
        }
        body = client.post("/api/v1/query", json={"question": "How?"}).json()
        assert body["metadata"]["result_count"] == 2
        kept = pipeline["prompt_builder"].last_results
        auth = [r for r in kept if r.file_path == "auth.py"]
        assert len(auth) == 1
        assert auth[0].score == 0.8

    def test_step_summaries_are_forwarded_to_the_prompt(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        pipeline["executor"].step_results = {
            "s1": _FakeStepResult(context_summary="the router calls verify_token"),
            "s2": _FakeStepResult(context_summary="   "),
        }
        client.post("/api/v1/query", json={"question": "How?"})
        forwarded = pipeline["prompt_builder"].last_sub_answers
        # The blank summary is dropped: it would spend context saying nothing.
        assert len(forwarded) == 1
        assert forwarded[0].answer == "the router calls verify_token"

    def test_empty_retrieval_still_produces_an_answer(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        pipeline["executor"].step_results = {}
        response = client.post("/api/v1/query", json={"question": "How?"})
        assert response.status_code == 200
        assert response.json()["metadata"]["result_count"] == 0

    def test_executor_failure_degrades_instead_of_500(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        class _BrokenExecutor:
            def execute(self, steps: Any) -> dict[str, Any]:
                raise RuntimeError("qdrant is down")

        pipeline["executor"] = _BrokenExecutor()
        response = client.post("/api/v1/query", json={"question": "How?"})
        assert response.status_code == 200
        assert response.json()["metadata"]["result_count"] == 0

    def test_unknown_repo_id_returns_404(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        response = client.post(
            "/api/v1/query", json={"question": "How?", "repo_id": 4242}
        )
        assert response.status_code == 404

    def test_known_repo_id_is_accepted(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        created = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        ).json()["repository"]
        response = client.post(
            "/api/v1/query", json={"question": "How?", "repo_id": created["id"]}
        )
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "   "},
            {"question": "ok", "repo_id": 0},
            {"question": "ok", "top_k": 0},
            {"question": "ok", "top_k": 999},
        ],
    )
    def test_invalid_payloads_are_rejected(
        self, client: Any, pipeline: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"


class TestQueryErrorMapping:
    @pytest.mark.parametrize(
        ("error_kind", "expected_status"),
        [
            ("rate_limit", 429),
            ("timeout", 504),
            ("auth", 502),
            ("invalid_response", 502),
            ("api_error", 502),
            (None, 502),
        ],
    )
    def test_generation_failure_maps_to_status(
        self,
        client: Any,
        pipeline: dict[str, Any],
        error_kind: str | None,
        expected_status: int,
    ) -> None:
        pipeline["generator"].answered = _answered(
            success=False, error="upstream failed", error_kind=error_kind
        )
        response = client.post("/api/v1/query", json={"question": "How?"})
        assert response.status_code == expected_status
        assert response.json()["detail"] == "upstream failed"


class TestPipelineCaching:
    def test_components_are_built_once_and_reused(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        client.post("/api/v1/query", json={"question": "first"})
        client.post("/api/v1/query", json={"question": "second"})
        # A rebuild per request would reset the fake's call counter.
        assert pipeline["generator"].calls == 2
        assert pipeline["decomposer"].calls == ["first", "second"]


# ---------------------------------------------------------------------------
# Retrieval adapter
# ---------------------------------------------------------------------------


class _FakeBackend:
    """A retrieval backend returning canned results, or raising."""

    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.error:
            raise self.error
        return self.results

    def get_neighbors(self, symbol: str, depth: int = 1) -> list[RetrievalResult]:
        if self.error:
            raise self.error
        return self.results


class TestRetrievalAdapter:
    def test_satisfies_the_executor_protocol(self) -> None:
        from reporag.agent.executor import RetrievalEngine

        adapter = query_routes._RetrievalEngineAdapter(
            vector_search=_FakeBackend(),
            bm25_search=_FakeBackend(),
            graph_retriever=_FakeBackend(),
        )
        assert isinstance(adapter, RetrievalEngine)

    def test_bm25_and_vector_delegate_to_their_backends(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            bm25_search=_FakeBackend([_result("a.py")]),
            vector_search=_FakeBackend([_result("b.py")]),
            graph_retriever=_FakeBackend(),
        )
        assert adapter.search_bm25("q", 5)[0].file_path == "a.py"
        assert adapter.search_vector("q", 5)[0].file_path == "b.py"

    def test_backend_failure_returns_empty_not_raise(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            bm25_search=_FakeBackend(error=ConnectionError("down")),
            vector_search=_FakeBackend(),
            graph_retriever=_FakeBackend(),
        )
        assert adapter.search_bm25("q", 5) == []

    def test_graph_search_without_a_symbol_retrieves_nothing(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            vector_search=_FakeBackend(),
            bm25_search=_FakeBackend(),
            graph_retriever=_FakeBackend([_result("a.py")]),
        )
        # No code-shaped identifier means no node to traverse from.
        assert adapter.search_graph("explain the architecture", 5) == []

    def test_graph_search_with_a_symbol_traverses(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            vector_search=_FakeBackend(),
            bm25_search=_FakeBackend(),
            graph_retriever=_FakeBackend([_result("auth.py")]),
        )
        found = adapter.search_graph("what calls authenticate_user?", 5)
        assert [r.file_path for r in found] == ["auth.py"]

    def test_hybrid_fuses_every_backend(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            bm25_search=_FakeBackend([_result("a.py", 1, 5)]),
            vector_search=_FakeBackend([_result("b.py", 1, 5)]),
            graph_retriever=_FakeBackend([_result("c.py", 1, 5)]),
        )
        found = adapter.search_hybrid("trace handle_request", 10)
        assert {r.file_path for r in found} == {"a.py", "b.py", "c.py"}

    def test_hybrid_survives_a_dead_backend(self) -> None:
        adapter = query_routes._RetrievalEngineAdapter(
            bm25_search=_FakeBackend([_result("a.py")]),
            vector_search=_FakeBackend(error=ConnectionError("qdrant down")),
            graph_retriever=_FakeBackend(),
        )
        assert [r.file_path for r in adapter.search_hybrid("q", 10)] == ["a.py"]


class TestSymbolExtraction:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("what calls authenticate_user?", ["authenticate_user"]),
            ("how does db.execute work", ["db.execute"]),
            ("where is getUserById defined", ["getUserById"]),
            ("explain the overall architecture", []),
            ("how does the app handle requests", []),
        ],
    )
    def test_extracts_code_shaped_identifiers_only(
        self, query: str, expected: list[str]
    ) -> None:
        assert query_routes._extract_symbols(query) == expected

    def test_deduplicates_and_limits(self) -> None:
        query = " ".join(f"sym_{i}" for i in range(10)) + " sym_0"
        found = query_routes._extract_symbols(query, limit=3)
        assert found == ["sym_0", "sym_1", "sym_2"]


# ---------------------------------------------------------------------------
# OpenAPI and error shape
# ---------------------------------------------------------------------------


class TestOpenAPI:
    def test_docs_are_served(self, client: Any) -> None:
        assert client.get("/docs").status_code == 200

    def test_schema_documents_every_endpoint(self, client: Any) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/health" in paths
        assert "/api/v1/repos/ingest" in paths
        assert "/api/v1/repos" in paths
        assert "/api/v1/repos/{repo_id}" in paths
        assert "/api/v1/query" in paths
        assert "/health" in paths


class TestErrorShape:
    def test_unhandled_exception_returns_clean_json(
        self, app: Any, pipeline: dict[str, Any]
    ) -> None:
        class _Exploding:
            def decompose(self, query: str, repo_context: Any = None) -> Any:
                raise RuntimeError("boom: /secret/path/leaked")

        pipeline["decomposer"] = _Exploding()
        # raise_server_exceptions=False so the app's own 500 handler renders
        # the response instead of TestClient re-raising into the test.
        with TestClient(app, raise_server_exceptions=False) as raw_client:
            response = raw_client.post("/api/v1/query", json={"question": "How?"})
        assert response.status_code == 500
        body = response.json()
        assert body == {
            "error": "internal_error",
            "detail": "An unexpected error occurred.",
            "status_code": 500,
        }
        # The traceback belongs in the log, never in the response.
        assert "secret" not in response.text


class TestAcceptanceCriteria:
    """One test per Issue 26 acceptance criterion."""

    def test_ingest_triggers_async_background_ingestion(
        self, client: Any, _no_real_ingestion: list[tuple[Any, ...]]
    ) -> None:
        response = client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        )
        assert response.status_code == 202
        assert len(_no_real_ingestion) == 1

    def test_repos_returns_list_with_status(self, client: Any) -> None:
        client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/pallets/click"},
        )
        body = client.get("/api/v1/repos").json()
        assert body and "status" in body[0]

    def test_query_returns_answer_citations_metadata(
        self, client: Any, pipeline: dict[str, Any]
    ) -> None:
        body = client.post("/api/v1/query", json={"question": "How?"}).json()
        assert set(body) == {"answer", "citations", "metadata"}

    def test_health_returns_component_status(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probes(monkeypatch)
        components = client.get("/api/v1/health").json()["components"]
        assert {"neo4j", "qdrant", "llm"} <= set(components)

    def test_openapi_docs_available(self, client: Any) -> None:
        assert client.get("/docs").status_code == 200

    def test_pydantic_validation_gives_clear_errors(self, client: Any) -> None:
        body = client.post("/api/v1/repos/ingest", json={"repo_url": "nope"}).json()
        assert body["error"] == "validation_error"
        assert body["errors"][0]["message"]

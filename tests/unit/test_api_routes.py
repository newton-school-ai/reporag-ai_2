"""Unit tests for RepoRAG API routes (repo, query, and OpenAPI documentation)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from reporag.api.main import app
from reporag.api.routes.repos import run_ingestion_pipeline
from reporag.db.models import Base, Repository, RepositoryStatus, User
from reporag.db.session import get_db
from reporag.generation.citation import Citation, CitationReport
from reporag.generation.generator import AnsweredQuery, GenerationResult
from reporag.ingestion.cloner import FileEntry
from reporag.retrieval.vector_search import RetrievalResult

client = TestClient(app)


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory sqlite database session for API tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_db())

    async def _override():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override
    yield session_factory, engine
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# OpenAPI & Documentation Tests
# ---------------------------------------------------------------------------


def test_openapi_docs_available():
    """Verify OpenAPI UI and JSON schema endpoints are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower() or "openapi" in response.text.lower()

    response_json = client.get("/openapi.json")
    assert response_json.status_code == 200
    schema = response_json.json()
    assert schema["info"]["title"] == "RepoRAG API"
    assert "/api/v1/repos/ingest" in schema["paths"]
    assert "/api/v1/repos" in schema["paths"]
    assert "/api/v1/query" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]


# ---------------------------------------------------------------------------
# Repository Endpoints Tests
# ---------------------------------------------------------------------------


def test_repo_ingest_endpoint(test_db_session):
    """Test POST /api/v1/repos/ingest triggers background task and returns 202."""
    with patch("reporag.api.routes.repos.run_ingestion_pipeline"):
        payload = {
            "repo_url": "https://github.com/test-org/test-repo.git",
            "branch": "main",
            "shallow": True,
        }
        response = client.post("/api/v1/repos/ingest", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert data["repo_id"] > 0
        assert data["name"] == "test-repo"
        assert data["url"] == "https://github.com/test-org/test-repo.git"
        assert data["status"] == "queued"
        assert "message" in data


def test_repo_ingest_validation_error():
    """Test POST /api/v1/repos/ingest validation failure with empty URL."""
    payload = {"repo_url": ""}
    response = client.post("/api/v1/repos/ingest", json=payload)
    assert response.status_code == 422


def test_list_and_get_repos(test_db_session):
    """Test GET /api/v1/repos and GET /api/v1/repos/{id}."""
    session_factory, _ = test_db_session

    # Seed data
    async def seed():
        async with session_factory() as session:
            user = User(
                username="seeduser", email="seed@test.com", hashed_password="pw"
            )
            session.add(user)
            await session.commit()
            repo = Repository(
                owner_id=user.id,
                name="my-fastapi-repo",
                url="https://github.com/test/my-fastapi-repo",
                status=RepositoryStatus.READY,
            )
            session.add(repo)
            await session.commit()
            return repo.id

    repo_id = asyncio.run(seed())

    # List repositories
    response = client.get("/api/v1/repos")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any(r["id"] == repo_id for r in data["repositories"])

    # Get single repository
    get_res = client.get(f"/api/v1/repos/{repo_id}")
    assert get_res.status_code == 200
    repo_data = get_res.json()
    assert repo_data["id"] == repo_id
    assert repo_data["name"] == "my-fastapi-repo"
    assert repo_data["status"] == "ready"


def test_get_nonexistent_repo(test_db_session):
    """Test GET /api/v1/repos/{id} returns 404 for unknown repository."""
    response = client.get("/api/v1/repos/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Background Ingestion Pipeline Runner Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_success(monkeypatch, tmp_path):
    """Test run_ingestion_pipeline clones, chunks, and marks repository ready."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr("reporag.api.routes.repos.async_session_maker", session_factory)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(username="admin", email="admin@test.com", hashed_password="pw")
        session.add(user)
        await session.commit()
        repo = Repository(
            owner_id=user.id,
            name="pipeline-test",
            url="https://github.com/org/pipeline-test",
            status=RepositoryStatus.QUEUED,
        )
        session.add(repo)
        await session.commit()
        repo_id = repo.id

    # Create dummy clone files in temp dir
    clone_dir = tmp_path / "cloned_repo"
    clone_dir.mkdir()
    sample_file = clone_dir / "sample.py"
    sample_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    mock_cloner = MagicMock()
    mock_cloner.last_clone_path = clone_dir
    mock_cloner.clone_and_discover.return_value = [
        FileEntry(path="sample.py", language="python", size_bytes=35)
    ]

    with patch("reporag.api.routes.repos.RepoCloner", return_value=mock_cloner):
        await run_ingestion_pipeline(
            repo_id=repo_id,
            repo_url="https://github.com/org/pipeline-test",
            branch="main",
            shallow=True,
        )

    # Verify repository status is READY
    async with session_factory() as session:
        from sqlalchemy import select

        res = await session.execute(select(Repository).where(Repository.id == repo_id))
        updated_repo = res.scalar_one()
        assert updated_repo.status == RepositoryStatus.READY


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_failure_handling(monkeypatch):
    """Test run_ingestion_pipeline marks repository FAILED on clone error."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr("reporag.api.routes.repos.async_session_maker", session_factory)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(username="admin2", email="admin2@test.com", hashed_password="pw")
        session.add(user)
        await session.commit()
        repo = Repository(
            owner_id=user.id,
            name="failing-repo",
            url="https://github.com/org/failing-repo",
            status=RepositoryStatus.QUEUED,
        )
        session.add(repo)
        await session.commit()
        repo_id = repo.id

    mock_cloner = MagicMock()
    mock_cloner.clone_and_discover.side_effect = RuntimeError("Git clone network error")
    mock_cloner.last_clone_path = None

    with patch("reporag.api.routes.repos.RepoCloner", return_value=mock_cloner):
        await run_ingestion_pipeline(
            repo_id=repo_id,
            repo_url="https://github.com/org/failing-repo",
        )

    async with session_factory() as session:
        from sqlalchemy import select

        res = await session.execute(select(Repository).where(Repository.id == repo_id))
        updated_repo = res.scalar_one()
        assert updated_repo.status == RepositoryStatus.FAILED


# ---------------------------------------------------------------------------
# Query Endpoint Tests
# ---------------------------------------------------------------------------


def test_query_endpoint_success(test_db_session):
    """Test POST /api/v1/query returns answer with citations and metadata."""
    session_factory, _ = test_db_session

    async def seed_repo():
        async with session_factory() as session:
            user = User(username="quser", email="q@test.com", hashed_password="pw")
            session.add(user)
            await session.commit()
            repo = Repository(
                owner_id=user.id,
                name="query-repo",
                url="https://github.com/test/query-repo",
                status=RepositoryStatus.READY,
            )
            session.add(repo)
            await session.commit()
            return repo.id

    repo_id = asyncio.run(seed_repo())

    mock_answered = AnsweredQuery(
        answer="Authentication is implemented using JWT tokens [src/auth.py:10-25].",
        citations=CitationReport(
            citations=[
                Citation(
                    file_path="src/auth.py",
                    start_line=10,
                    end_line=25,
                    snippet="def create_access_token(): ...",
                    valid=True,
                    raw="[src/auth.py:10-25]",
                )
            ],
            coverage=1.0,
            valid_count=1,
            invalid_count=0,
        ),
        generation=GenerationResult(
            text="Authentication is implemented using JWT tokens [src/auth.py:10-25].",
            success=True,
            model="gpt-4o",
            provider="openai",
            latency_seconds=0.45,
        ),
    )

    mock_retrieval = [
        RetrievalResult(
            score=0.92,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            symbol_name="create_access_token",
            chunk_text="def create_access_token():\n    return 'token'",
        )
    ]

    with (
        patch(
            "reporag.api.routes.query._execute_retrieval",
            return_value=mock_retrieval,
        ),
        patch(
            "reporag.generation.generator.AnswerGenerator.generate_with_citations",
            return_value=mock_answered,
        ),
    ):
        payload = {
            "question": "How is authentication handled?",
            "repo_id": repo_id,
            "top_k": 5,
            "include_citations": True,
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "JWT tokens" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["file_path"] == "src/auth.py"
        assert data["citations"][0]["valid"] is True
        assert data["metadata"]["valid_citations"] == 1
        assert data["metadata"]["coverage"] == 1.0
        assert data["metadata"]["chunks_retrieved"] == 1


def test_query_endpoint_nonexistent_repo(test_db_session):
    """Test POST /api/v1/query returns 404 for unknown repository id."""
    payload = {
        "question": "What does this do?",
        "repo_id": 88888,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_query_endpoint_validation_error():
    """Test POST /api/v1/query validation error on empty question."""
    payload = {
        "question": "",
        "repo_id": 1,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 422

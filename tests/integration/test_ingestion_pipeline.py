"""Integration tests for the background ingestion pipeline.

Tests _run_ingestion_job directly (not via HTTP) against an in-memory DB,
with RepoCloner.clone_and_discover mocked so no real git clone runs.

Status transitions verified:
  QUEUED -> CLONING -> READY   (happy path)
  QUEUED -> CLONING -> FAILED  (CloneError)
  QUEUED -> CLONING -> FAILED  (unexpected exception)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reporag.api.routes.repos import _run_ingestion_job
from reporag.db.models import (
    Base,
    IngestionJob,
    JobStatus,
    Repository,
    RepositoryStatus,
    User,
)
from reporag.ingestion.cloner import CloneError, FileEntry

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def integration_db():
    """Isolated in-memory DB for direct pipeline calls."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_maker() as session:
        # Seed: demo user + repository + ingestion job
        user = User(username="test-user", email="test@test.local", hashed_password="x")
        session.add(user)
        await session.flush()

        repo = Repository(
            owner_id=user.id,
            name="test-repo",
            url="https://github.com/a/b",
            status=RepositoryStatus.QUEUED,
        )
        session.add(repo)
        await session.flush()

        job = IngestionJob(repository_id=repo.id, status=JobStatus.PENDING)
        session.add(job)
        await session.commit()
        await session.refresh(repo)
        await session.refresh(job)

    yield engine, session_maker, repo.id, job.id
    await engine.dispose()


# ---------------------------------------------------------------------------
# Happy path: QUEUED -> CLONING -> READY
# ---------------------------------------------------------------------------


class TestIngestionPipelineHappyPath:
    async def test_status_transitions_to_ready_on_success(self, integration_db):
        engine, session_maker, repo_id, job_id = integration_db
        fake_manifest = [
            FileEntry(path="src/app.py", language="python", size_bytes=100)
        ]

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.return_value = fake_manifest
            mock_cloner_cls.return_value = mock_cloner

            await _run_ingestion_job(repo_id, job_id, "https://github.com/a/b", None)

        async with session_maker() as db:
            repo = await db.get(Repository, repo_id)
            job = await db.get(IngestionJob, job_id)

        assert repo.status == RepositoryStatus.READY
        assert job.status == JobStatus.COMPLETED

    async def test_cloner_called_with_correct_url_and_branch(self, integration_db):
        engine, session_maker, repo_id, job_id = integration_db
        fake_manifest = [FileEntry(path="a.py", language="python", size_bytes=10)]

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.return_value = fake_manifest
            mock_cloner_cls.return_value = mock_cloner

            await _run_ingestion_job(
                repo_id, job_id, "https://github.com/a/b", "feature/dev"
            )
            mock_cloner.clone_and_discover.assert_called_once_with(
                "https://github.com/a/b", "feature/dev"
            )


# ---------------------------------------------------------------------------
# Error path: CloneError -> FAILED
# ---------------------------------------------------------------------------


class TestIngestionPipelineCloneError:
    async def test_status_transitions_to_failed_on_clone_error(self, integration_db):
        engine, session_maker, repo_id, job_id = integration_db

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.side_effect = CloneError(
                "Connection refused"
            )
            mock_cloner_cls.return_value = mock_cloner

            await _run_ingestion_job(repo_id, job_id, "https://github.com/a/b", None)

        async with session_maker() as db:
            repo = await db.get(Repository, repo_id)
            job = await db.get(IngestionJob, job_id)

        assert repo.status == RepositoryStatus.FAILED
        assert job.status == JobStatus.FAILED

    async def test_no_exception_propagates_on_clone_error(self, integration_db):
        """The background task must not crash the event loop."""
        engine, session_maker, repo_id, job_id = integration_db

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.side_effect = CloneError("bad url")
            mock_cloner_cls.return_value = mock_cloner

            # Must not raise
            await _run_ingestion_job(repo_id, job_id, "https://bad-url", None)


# ---------------------------------------------------------------------------
# Error path: unexpected exception -> FAILED
# ---------------------------------------------------------------------------


class TestIngestionPipelineUnexpectedError:
    async def test_status_transitions_to_failed_on_unexpected_error(
        self, integration_db
    ):
        engine, session_maker, repo_id, job_id = integration_db

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.side_effect = OSError("disk full")
            mock_cloner_cls.return_value = mock_cloner

            await _run_ingestion_job(repo_id, job_id, "https://github.com/a/b", None)

        async with session_maker() as db:
            repo = await db.get(Repository, repo_id)
            job = await db.get(IngestionJob, job_id)

        assert repo.status == RepositoryStatus.FAILED
        assert job.status == JobStatus.FAILED

    async def test_no_exception_propagates_on_unexpected_error(self, integration_db):
        engine, session_maker, repo_id, job_id = integration_db

        with (
            patch("reporag.api.routes.repos.async_session_maker", session_maker),
            patch("reporag.api.routes.repos.RepoCloner") as mock_cloner_cls,
        ):
            mock_cloner = MagicMock()
            mock_cloner.clone_and_discover.side_effect = MemoryError("OOM")
            mock_cloner_cls.return_value = mock_cloner

            # Must not raise
            await _run_ingestion_job(repo_id, job_id, "https://github.com/a/b", None)


# ---------------------------------------------------------------------------
# Edge: job/repo missing from DB (race condition guard)
# ---------------------------------------------------------------------------


class TestIngestionPipelineMissingRecords:
    async def test_missing_repo_does_not_raise(self, integration_db):
        engine, session_maker, repo_id, job_id = integration_db

        with patch("reporag.api.routes.repos.async_session_maker", session_maker):
            # Pass a non-existent repo/job id -- must return silently
            await _run_ingestion_job(99999, 99999, "https://github.com/a/b", None)

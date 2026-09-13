"""Integration tests for the ingestion pipeline behind POST /api/v1/repos/ingest.

Tests the full background task execution against a real database session:
- Request schedules background task
- Task advances Repository status (queued -> cloning -> processing -> ready/failed)
- Task advances IngestionJob status (pending -> in_progress -> completed/failed)
- Client polling GET /api/v1/repos/{id} observes status changes
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from reporag.api.routes import repos as repos_routes
from reporag.db.models import IngestionJob, JobStatus, Repository, RepositoryStatus


@pytest.fixture
def live_ingestion(api_app: Any, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Wire the background task's session maker to the test database."""
    monkeypatch.setattr(
        repos_routes, "async_session_maker", api_app.state.session_maker
    )
    return monkeypatch


def _parse_returning(**counts: Any) -> Any:
    return lambda url, branch: repos_routes._IngestionStats(**counts)


async def _repo(api_app: Any, repo_id: int) -> Repository:
    async with api_app.state.session_maker() as session:
        return await session.get(Repository, repo_id)


async def _job(api_app: Any, job_id: int) -> IngestionJob:
    async with api_app.state.session_maker() as session:
        return await session.get(IngestionJob, job_id)


class TestIngestionPipelineIntegration:
    async def test_request_to_ready_end_to_end(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes,
            "_parse_repository",
            _parse_returning(files=4, symbols=20, chunks=25, languages=["python"]),
        )
        response = await async_client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/acme/demo", "branch": "main"},
        )
        assert response.status_code == 202
        body = response.json()
        repo_id = body["repository"]["id"]
        job_id = body["job_id"]

        repo = await _repo(api_app, repo_id)
        job = await _job(api_app, job_id)
        assert repo.status == RepositoryStatus.READY
        assert job.status == JobStatus.COMPLETED

    async def test_client_polling_observes_status(
        self,
        async_client: AsyncClient,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes,
            "_parse_repository",
            _parse_returning(files=2, symbols=5, chunks=8, languages=["python"]),
        )
        created = await async_client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/acme/demo2"},
        )
        repo_id = created.json()["repository"]["id"]

        polled = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert polled.status_code == 200
        assert polled.json()["status"] == RepositoryStatus.READY.value

    async def test_ingestion_failure_reflected_in_repository_and_job(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        def raise_clone_error(url: str, branch: Any) -> Any:
            raise RuntimeError("Failed to clone remote repository")

        live_ingestion.setattr(repos_routes, "_parse_repository", raise_clone_error)

        created = await async_client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/acme/failing-repo"},
        )
        assert created.status_code == 202
        body = created.json()
        repo_id = body["repository"]["id"]
        job_id = body["job_id"]

        repo = await _repo(api_app, repo_id)
        job = await _job(api_app, job_id)
        assert repo.status == RepositoryStatus.FAILED
        assert job.status == JobStatus.FAILED

        polled = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert polled.json()["status"] == RepositoryStatus.FAILED.value

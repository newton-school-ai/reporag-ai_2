"""Integration tests for the ingestion pipeline behind ``POST /repos/ingest``.

Unit tests call :func:`~reporag.api.routes.repos.run_ingestion` directly.
These drive it the way the API does -- an HTTP request schedules it, it runs
against a real database, and the result is observed by polling the same
endpoints a client would. That covers the seams unit tests cannot: that the
row committed by the request is the one the task later finds, that the
background session is separate from the request session, and that the status
a client polls actually changes.

What stays fake: the clone and parse step. Cloning a real repository over
the network would make this suite slow and dependent on GitHub being up.
:func:`~reporag.api.routes.repos._parse_repository` is the seam, so
everything above it -- scheduling, sessions, transactions, status
transitions, the endpoints -- is the real implementation.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from reporag.api.routes import repos as repos_routes
from reporag.db.models import IngestionJob, JobStatus, Repository, RepositoryStatus


@pytest.fixture
def live_ingestion(api_app: Any, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Let the real background task run against the test database.

    The task opens its own session from the module-level
    ``async_session_maker``, which points at the production database; it is
    redirected at the temporary one here. The ``stub_ingestion`` fixture is
    deliberately *not* used -- these tests want the real task.
    """
    monkeypatch.setattr(
        repos_routes, "async_session_maker", api_app.state.session_maker
    )
    return monkeypatch


def _parse_returning(**counts: Any) -> Any:
    """Build a stand-in for the clone-and-parse step with fixed counts."""
    return lambda url, branch: repos_routes._IngestionStats(**counts)


async def _repo(api_app: Any, repo_id: int) -> Repository:
    """Read a repository row straight from the database."""
    async with api_app.state.session_maker() as session:
        return await session.get(Repository, repo_id)


async def _job(api_app: Any, job_id: int) -> IngestionJob:
    """Read an ingestion job row straight from the database."""
    async with api_app.state.session_maker() as session:
        return await session.get(IngestionJob, job_id)


class TestIngestionLifecycle:
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
            json={"repo_url": "https://github.com/acme/demo"},
        )
        assert response.status_code == 202
        body = response.json()

        # httpx runs BackgroundTasks before returning, so by this point the
        # task the request scheduled has already completed.
        repo = await _repo(api_app, body["repository"]["id"])
        assert repo.status == RepositoryStatus.READY

    async def test_the_status_a_client_polls_actually_changes(
        self,
        async_client: AsyncClient,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=2)
        )
        created = await async_client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/acme/demo"},
        )
        repo_id = created.json()["repository"]["id"]
        assert created.json()["repository"]["status"] == RepositoryStatus.QUEUED.value

        polled = await async_client.get(f"/api/v1/repos/{repo_id}")
        # The client sees the end state through the same endpoint it would
        # poll -- not through a test-only back channel.
        assert polled.json()["status"] == RepositoryStatus.READY.value

    async def test_job_row_completes_alongside_the_repository(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=1)
        )
        body = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/demo"},
            )
        ).json()
        job = await _job(api_app, body["job_id"])
        assert job.status == JobStatus.COMPLETED
        assert job.repository_id == body["repository"]["id"]

    async def test_the_branch_reaches_the_clone_step(
        self,
        async_client: AsyncClient,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        seen: list[tuple[str, str | None]] = []

        def record(url: str, branch: str | None) -> Any:
            seen.append((url, branch))
            return repos_routes._IngestionStats(files=1)

        live_ingestion.setattr(repos_routes, "_parse_repository", record)
        await async_client.post(
            "/api/v1/repos/ingest",
            json={"repo_url": "https://github.com/acme/demo", "branch": "release/2.x"},
        )
        # The branch survives validation, the response, the scheduling hop
        # and the threadpool dispatch.
        assert seen == [("https://github.com/acme/demo", "release/2.x")]


class TestIngestionFailures:
    async def test_clone_failure_lands_on_failed(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        def boom(url: str, branch: str | None) -> Any:
            raise RuntimeError("fatal: repository not found")

        live_ingestion.setattr(repos_routes, "_parse_repository", boom)
        body = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/missing"},
            )
        ).json()

        # The request still succeeded -- the failure is in the background,
        # and the only way a client learns about it is the status.
        repo = await _repo(api_app, body["repository"]["id"])
        job = await _job(api_app, body["job_id"])
        assert repo.status == RepositoryStatus.FAILED
        assert job.status == JobStatus.FAILED

    async def test_empty_repository_is_a_failure_not_a_ready_repo(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=0)
        )
        body = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/empty"},
            )
        ).json()
        # Marking it ready would mean queries against it return nothing with
        # no explanation.
        repo = await _repo(api_app, body["repository"]["id"])
        assert repo.status == RepositoryStatus.FAILED

    async def test_a_failed_ingest_does_not_break_the_next_one(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        def boom(url: str, branch: str | None) -> Any:
            raise RuntimeError("transient network error")

        live_ingestion.setattr(repos_routes, "_parse_repository", boom)
        first = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/one"},
            )
        ).json()

        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=3)
        )
        second = (
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": "https://github.com/acme/two"},
            )
        ).json()

        assert (
            await _repo(api_app, first["repository"]["id"])
        ).status == RepositoryStatus.FAILED
        assert (
            await _repo(api_app, second["repository"]["id"])
        ).status == RepositoryStatus.READY


class TestIngestionPersistence:
    async def test_repositories_accumulate_and_are_listed(
        self,
        async_client: AsyncClient,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=1)
        )
        for name in ("alpha", "beta", "gamma"):
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"repo_url": f"https://github.com/acme/{name}"},
            )
        body = (await async_client.get("/api/v1/repos")).json()
        assert body["total"] == 3
        assert {r["name"] for r in body["repositories"]} == {
            "alpha",
            "beta",
            "gamma",
        }

    async def test_every_ingest_shares_one_owner(
        self,
        async_client: AsyncClient,
        api_app: Any,
        live_ingestion: pytest.MonkeyPatch,
    ) -> None:
        live_ingestion.setattr(
            repos_routes, "_parse_repository", _parse_returning(files=1)
        )
        ids = []
        for name in ("alpha", "beta"):
            body = (
                await async_client.post(
                    "/api/v1/repos/ingest",
                    json={"repo_url": f"https://github.com/acme/{name}"},
                )
            ).json()
            ids.append(body["repository"]["id"])

        owners = {(await _repo(api_app, rid)).owner_id for rid in ids}
        # The placeholder account is created once and reused; a second
        # insert would violate the unique constraint on email.
        assert len(owners) == 1

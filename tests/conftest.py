"""Shared fixtures for tests that exercise the HTTP API.

Every API test needs the same three things: a database with the schema
applied, an application wired to it, and a client that speaks to that
application without a socket. Defining them once here keeps the unit and
integration suites agreeing on what "the app under test" means.

Why a factory, not the module-level app
---------------------------------------
``reporag.api.main.app`` is a process-wide singleton, and the query route
caches its pipeline components on ``app.state``. Tests that shared it would
leak one test's fakes into the next, in an order-dependent way that only
shows up when the suite is run in a different order. Every fixture here
builds a fresh application with :func:`~reporag.api.main.create_app`
instead, so isolation is structural rather than something each test has to
remember to clean up.

Why a SQLite file, not ``:memory:``
-----------------------------------
An in-memory SQLite database is private to the connection that created it.
The schema created by one connection would be invisible to the pooled
connections the app's engine hands out, and every request would fail on a
missing table. A file under ``tmp_path`` is shared across connections and
is thrown away with the test.

No test here touches the network: Qdrant, Neo4j and the LLM are never
constructed, and the ingestion background task is stubbed by
:func:`stub_ingestion` in the modules that need it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from reporag.api.main import create_app
from reporag.api.routes import repos as repos_routes
from reporag.db.models import Base
from reporag.db.session import get_db


@pytest.fixture
def db_url(tmp_path: Any) -> str:
    """A temporary SQLite database with every ORM table created."""
    path = tmp_path / "test.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    return f"sqlite+aiosqlite:///{path}"


@pytest.fixture
def db_engine(db_url: str) -> AsyncEngine:
    """An async engine bound to the temporary database."""
    return create_async_engine(db_url)


@pytest.fixture
def session_maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory for the temporary database.

    ``expire_on_commit=False`` so a test can still read a row's attributes
    after committing it, which is how most assertions here are written.
    """
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
def api_app(
    db_engine: AsyncEngine, session_maker: async_sessionmaker[AsyncSession]
) -> FastAPI:
    """An isolated application wired to the temporary database.

    ``app.state.db_engine`` is what the startup hook in
    :mod:`reporag.api.main` reads, so the schema-creation step runs against
    this engine and never touches the real ``reporag.db`` file. The
    ``get_db`` override covers the request path.
    """
    application = create_app()
    application.state.db_engine = db_engine

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_maker() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db
    application.state.session_maker = session_maker
    return application


@pytest.fixture
def client(api_app: FastAPI) -> Iterator[TestClient]:
    """A synchronous client bound to the isolated app.

    Entered as a context manager so the lifespan hooks run; a bare
    ``TestClient(app)`` would skip startup and shutdown entirely.
    """
    with TestClient(api_app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An httpx client speaking ASGI directly to the isolated app.

    For tests that must await application code themselves -- driving a
    background task, or asserting on the database between requests --
    where the threaded :class:`TestClient` would put the app and the
    assertions in different event loops.
    """
    async with AsyncClient(
        transport=ASGITransport(app=api_app), base_url="http://test"
    ) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def db_session(
    session_maker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A session on the temporary database, for seeding and assertions."""
    async with session_maker() as session:
        yield session


@pytest.fixture
def stub_ingestion(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace the background ingestion task and record its calls.

    ``POST /repos/ingest`` schedules :func:`~reporag.api.routes.repos.
    run_ingestion`, which clones a real repository over the network. Tests
    assert that it was scheduled with the right arguments instead.

    Returns:
        The list that each call appends ``(repo_id, repo_url, branch,
        job_id)`` to.
    """
    calls: list[tuple[Any, ...]] = []

    async def fake_run_ingestion(
        repo_id: int,
        repo_url: str,
        branch: str | None,
        job_id: int | None = None,
    ) -> None:
        calls.append((repo_id, repo_url, branch, job_id))

    monkeypatch.setattr(repos_routes, "run_ingestion", fake_run_ingestion)
    return calls

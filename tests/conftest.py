"""Shared pytest fixtures for the full test suite.

Every test that needs an HTTP client or a database session uses these fixtures.
No real network, Qdrant, Neo4j, or LLM is contacted -- all external deps are
patched at the call-site in each test module.

Fixture summary
---------------
db_engine       -- fresh in-memory SQLite engine; tables are created once per test.
async_client    -- httpx AsyncClient wired to the app; overrides ``get_db`` with
                   the in-memory engine and patches ``reporag.api.main.engine``
                   so the lifespan also uses the test engine.
db_session      -- raw async session for seeding data or asserting DB state.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reporag.api.main import app
from reporag.db.models import Base
from reporag.db.session import get_db

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_engine():
    """In-memory SQLite engine with all ORM tables pre-created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_client(db_engine):
    """httpx AsyncClient connected to the full ASGI app via in-memory DB.

    * Overrides the ``get_db`` FastAPI dependency so every request-scoped
      session talks to the test engine.
    * Patches ``reporag.api.main.engine`` so the startup lifespan hook
      (``Base.metadata.create_all``) also uses the test engine -- no
      ``reporag.db`` file is touched.
    """
    test_session_maker = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _override_get_db():
        async with test_session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    with patch("reporag.api.main.engine", db_engine):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client

    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Raw async DB session for direct DB inspection or seeding in tests."""
    test_session_maker = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with test_session_maker() as session:
        yield session

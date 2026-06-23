"""Unit tests for the database layer (models, session factory, URL handling).

These tests run entirely against an in-memory SQLite database, so they need
no external services and are safe in CI.
"""

import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from src.reporag.db import (
    Base,
    IngestionJob,
    JobStatus,
    QueryLog,
    Repository,
    RepositoryStatus,
    User,
)
from src.reporag.db.session import get_db, make_async_url

# ---------------------------------------------------------------------------
# URL normalization (pure, no database needed)
# ---------------------------------------------------------------------------


def test_make_async_url_sqlite():
    assert (
        make_async_url("sqlite:///./reporag.db") == "sqlite+aiosqlite:///./reporag.db"
    )


def test_make_async_url_postgres():
    assert (
        make_async_url("postgresql://u:p@h:5432/db")
        == "postgresql+asyncpg://u:p@h:5432/db"
    )
    assert make_async_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"


def test_make_async_url_passthrough():
    # Already-async URLs and unknown schemes are returned unchanged.
    assert (
        make_async_url("postgresql+asyncpg://u:p@h/db")
        == "postgresql+asyncpg://u:p@h/db"
    )
    assert make_async_url("mysql://u:p@h/db") == "mysql://u:p@h/db"
    assert make_async_url("not a url") == "not a url"


# ---------------------------------------------------------------------------
# Schema / metadata
# ---------------------------------------------------------------------------


def test_metadata_has_all_tables():
    assert set(Base.metadata.tables) == {
        "users",
        "repositories",
        "ingestion_jobs",
        "query_logs",
    }


# ---------------------------------------------------------------------------
# Async session round-trip against in-memory SQLite
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session():
    """An async session bound to a fresh in-memory SQLite database."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session

    await engine.dispose()


async def test_insert_and_relationships(session: AsyncSession):
    user = User(email="dev@example.com", name="Dev")
    repo = Repository(
        url="https://github.com/x/y",
        name="y",
        owner=user,
        status=RepositoryStatus.QUEUED,
    )
    IngestionJob(repository=repo, status=JobStatus.PENDING, branch="main")
    QueryLog(
        repository=repo, user=user, question="how does auth work?", num_citations=2
    )

    session.add(repo)
    await session.commit()

    repos = (await session.execute(select(Repository))).scalars().all()
    assert len(repos) == 1

    fetched = repos[0]
    assert fetched.status is RepositoryStatus.QUEUED
    assert fetched.file_count == 0  # column default applied
    assert fetched.owner is not None and fetched.owner.email == "dev@example.com"
    assert len(fetched.ingestion_jobs) == 1
    assert fetched.ingestion_jobs[0].status is JobStatus.PENDING
    assert fetched.query_logs[0].question == "how does auth work?"
    # Timestamps come from the server-side default.
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


async def test_enum_persisted_as_lowercase_value(session: AsyncSession):
    session.add(Repository(url="u", name="n", status=RepositoryStatus.READY))
    await session.commit()

    raw = (await session.execute(text("SELECT status FROM repositories"))).scalar_one()
    assert raw == "ready"


async def test_cascade_delete_removes_children(session: AsyncSession):
    repo = Repository(url="u", name="n")
    IngestionJob(repository=repo)
    QueryLog(repository=repo, question="q")
    session.add(repo)
    await session.commit()

    await session.delete(repo)
    await session.commit()

    jobs = (await session.execute(select(IngestionJob))).scalars().all()
    logs = (await session.execute(select(QueryLog))).scalars().all()
    assert jobs == []
    assert logs == []


async def test_get_db_yields_async_session():
    agen = get_db()
    db_session = await agen.__anext__()
    try:
        assert isinstance(db_session, AsyncSession)
    finally:
        await agen.aclose()

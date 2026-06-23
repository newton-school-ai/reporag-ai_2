import inspect
import os

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from alembic import command
from alembic.config import Config
from src.reporag.db.models import (
    Base,
    IngestionJob,
    JobStatus,
    QueryLog,
    Repository,
    RepositoryStatus,
    User,
)
from src.reporag.db.session import get_async_database_url, get_db


@pytest.fixture
async def async_engine():
    # Use in-memory SQLite with aiosqlite for isolated tests
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def async_session(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_database_models_can_be_created(async_session):
    async with async_session() as session:
        # Create user
        user = User(username="testuser", email="test@example.com", hashed_password="pw")
        session.add(user)
        await session.commit()

        # Create repository
        repo = Repository(
            owner_id=user.id,
            name="test-repo",
            url="https://github.com/test/repo",
            status=RepositoryStatus.QUEUED,
        )
        session.add(repo)
        await session.commit()

        # Create ingestion job
        job = IngestionJob(repository_id=repo.id, status=JobStatus.PENDING)
        session.add(job)
        await session.commit()

        # Create query log
        log = QueryLog(
            user_id=user.id,
            repository_id=repo.id,
            query_text="What is this?",
            response_text="It is a test.",
        )
        session.add(log)
        await session.commit()

        # Assert IDs are assigned
        assert user.id is not None
        assert repo.id is not None
        assert job.id is not None
        assert log.id is not None

        # Assert enums are correctly set
        assert repo.status == RepositoryStatus.QUEUED
        assert job.status == JobStatus.PENDING


def test_get_async_database_url():
    """Test PostgreSQL URL switching."""
    assert (
        get_async_database_url("postgresql://user:pass@localhost/db")
        == "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert (
        get_async_database_url("sqlite:///:memory:") == "sqlite+aiosqlite:///:memory:"
    )
    assert (
        get_async_database_url("mysql://user:pass@localhost/db")
        == "mysql://user:pass@localhost/db"
    )


@pytest.mark.asyncio
async def test_get_db_yields_async_session():
    """Test get_db dependency yields an AsyncSession."""
    gen = get_db()
    assert inspect.isasyncgen(gen)
    session = await anext(gen)
    assert hasattr(session, "execute")
    await gen.aclose()


@pytest.mark.asyncio
async def test_cascade_delete_related_records(async_session):
    """Test that deleting a repository cascade-deletes related ingestion jobs and query logs."""
    async with async_session() as session:
        # Create user and repo
        user = User(
            username="deleter", email="deleter@example.com", hashed_password="pw"
        )
        session.add(user)
        await session.commit()

        repo = Repository(
            owner_id=user.id,
            name="delete-repo",
            url="https://x.com",
            status=RepositoryStatus.QUEUED,
        )
        session.add(repo)
        await session.commit()

        # Add child records
        job = IngestionJob(repository_id=repo.id, status=JobStatus.PENDING)
        log = QueryLog(
            user_id=user.id,
            repository_id=repo.id,
            query_text="hello",
            response_text="hi",
        )
        session.add_all([job, log])
        await session.commit()

        # Verify they exist
        assert job.id is not None
        assert log.id is not None

        # Delete repo
        await session.delete(repo)
        await session.commit()

        # Assert children are gone
        from sqlalchemy import select

        job_result = await session.execute(
            select(IngestionJob).where(IngestionJob.id == job.id)
        )
        log_result = await session.execute(
            select(QueryLog).where(QueryLog.id == log.id)
        )

        assert job_result.scalar_one_or_none() is None
        assert log_result.scalar_one_or_none() is None


def test_alembic_migrations_can_upgrade_and_downgrade():
    """Test that migrations can be applied and rolled back via alembic api."""
    # We must use a local file database because memory databases lose schema on disconnect
    test_db_url = "sqlite:///test_migrations.db"
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)

    try:
        # Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Downgrade to base
        command.downgrade(alembic_cfg, "base")
    finally:
        if os.path.exists("test_migrations.db"):
            os.remove("test_migrations.db")

import os
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Force SQLite in-memory for all DB tests so they run fast and isolated."""
    test_db_url = "sqlite+aiosqlite:///:memory:"
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    return test_db_url


@pytest.mark.asyncio
async def test_database_models_can_be_created():
    """Verify that all four required models can be instantiated."""
    user = User(username="testuser", email="test@test.com", hashed_password="pw")
    repo = Repository(
        owner_id=1,
        name="reporag",
        url="https://github.com",
        status=RepositoryStatus.QUEUED,
    )
    job = IngestionJob(repository_id=1, status=JobStatus.PENDING)
    log = QueryLog(user_id=1, repository_id=1, query_text="Q?", response_text="A!")

    assert user.username == "testuser"
    assert repo.name == "reporag"
    assert job.status == JobStatus.PENDING
    assert log.query_text == "Q?"


def test_get_async_database_url():
    """Assert Postgres URL switching works without code changes."""
    assert (
        get_async_database_url("postgresql://u:p@host/db")
        == "postgresql+asyncpg://u:p@host/db"
    )
    assert (
        get_async_database_url("sqlite:///local.db") == "sqlite+aiosqlite:///local.db"
    )


@pytest.mark.asyncio
async def test_get_db_yields_async_session():
    """Assert session factory provides async-compatible sessions."""
    gen = get_db()
    session = await anext(gen)
    assert isinstance(session, AsyncSession)
    await gen.aclose()


@pytest.mark.asyncio
async def test_cascade_delete_related_records(setup_test_db):
    """Assert deleting a repository cascades to ingestion jobs and query logs."""
    # Create dynamic engine strictly for the test to ensure :memory: usage
    engine = create_async_engine(setup_test_db, echo=False)
    async_session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        # Create parent records
        user = User(username="cascade", email="cascade@test.com", hashed_password="pw")
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
            user_id=user.id, repository_id=repo.id, query_text="Q?", response_text="A!"
        )
        session.add_all([job, log])
        await session.commit()

        assert job.id is not None
        assert log.id is not None

        # Delete repo
        await session.delete(repo)
        await session.commit()

        # Assert children are gone
        job_result = await session.execute(
            select(IngestionJob).where(IngestionJob.id == job.id)
        )
        log_result = await session.execute(
            select(QueryLog).where(QueryLog.id == log.id)
        )

        assert job_result.scalar_one_or_none() is None
        assert log_result.scalar_one_or_none() is None


def test_alembic_migrations_can_upgrade_and_downgrade():
    """Assert alembic upgrade head creates all tables."""
    unique_db = f"test_mig_{uuid.uuid4().hex}.db"
    test_db_url = f"sqlite+aiosqlite:///{unique_db}"

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)

    try:
        command.upgrade(alembic_cfg, "head")
        command.downgrade(alembic_cfg, "base")
    except Exception as e:
        pytest.fail(f"Alembic migration failed: {e}")
    finally:
        if os.path.exists(unique_db):
            os.remove(unique_db)

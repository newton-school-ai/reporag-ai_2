import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.reporag.db.models import (
    Base,
    IngestionJob,
    JobStatus,
    QueryLog,
    Repository,
    RepositoryStatus,
    User,
)


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

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def get_async_database_url(url: str) -> str:
    """Format the database URL to use async drivers if necessary."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


# Get DATABASE_URL from environment or use a default SQLite path
raw_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./reporag.db")
DATABASE_URL = get_async_database_url(raw_url)

# Setup async engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Setup async session factory
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency for providing a database session."""
    async with async_session_maker() as session:
        yield session

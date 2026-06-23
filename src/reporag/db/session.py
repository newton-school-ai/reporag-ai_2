"""Async database session factory.

Builds a single async SQLAlchemy engine from ``settings.database_url`` and
exposes an async session factory plus a FastAPI dependency (``get_db``).

The same code targets SQLite (default) and Postgres -- only DATABASE_URL
changes. A plain sync-style DSN such as ``sqlite:///./reporag.db`` or
``postgresql://user:pass@host/db`` is upgraded to its async driver
automatically, so config files do not need to spell out ``+aiosqlite`` /
``+asyncpg``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.reporag.config import settings

# Maps a bare (sync) scheme to the async driver RepoRAG ships with.
_ASYNC_DRIVERS = {
    "sqlite": "sqlite+aiosqlite",
    "postgresql": "postgresql+asyncpg",
    "postgres": "postgresql+asyncpg",
}


def make_async_url(database_url: str) -> str:
    """Normalize a database URL to use an async driver.

    URLs that already name a driver (``postgresql+asyncpg://...``) are
    returned unchanged. Bare schemes are upgraded to their async equivalent;
    anything we do not recognize is returned untouched.
    """
    scheme, sep, rest = database_url.partition("://")
    if not sep or "+" in scheme:
        return database_url
    async_scheme = _ASYNC_DRIVERS.get(scheme.lower())
    if async_scheme is None:
        return database_url
    return f"{async_scheme}://{rest}"


def create_engine(database_url: str | None = None, **kwargs: Any) -> AsyncEngine:
    """Create an async engine, defaulting to the configured DATABASE_URL."""
    url = make_async_url(database_url or settings.database_url)
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        # aiosqlite forwards this to sqlite3.connect; safe for async usage.
        connect_args["check_same_thread"] = False
    return create_async_engine(
        url,
        echo=settings.app_debug and settings.app_env != "production",
        pool_pre_ping=True,
        connect_args=connect_args,
        **kwargs,
    )


# Module-level engine + session factory used by the application.
engine: AsyncEngine = create_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async session and closes it.

    Usage::

        @router.get("/repos")
        async def list_repos(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session

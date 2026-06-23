"""Database package: ORM models and async session management.

Public surface::

    from src.reporag.db import Base, get_db, User, Repository
"""

from src.reporag.db.models import (
    Base,
    IngestionJob,
    JobStatus,
    QueryLog,
    Repository,
    RepositoryStatus,
    User,
)
from src.reporag.db.session import (
    AsyncSessionLocal,
    create_engine,
    engine,
    get_db,
    make_async_url,
)

__all__ = [
    "Base",
    "User",
    "Repository",
    "IngestionJob",
    "QueryLog",
    "RepositoryStatus",
    "JobStatus",
    "AsyncSessionLocal",
    "create_engine",
    "engine",
    "get_db",
    "make_async_url",
]

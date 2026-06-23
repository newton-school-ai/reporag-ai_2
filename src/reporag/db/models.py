"""SQLAlchemy ORM models for RepoRAG.

Defines the persistent schema: User, Repository, IngestionJob, QueryLog.
The same models target SQLite (local dev) and Postgres (production). Status
fields use SQLAlchemy Enum, which maps to a CHECK constraint on SQLite and a
native ENUM type on Postgres -- switching engines needs no model changes.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, MetaData, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names keep Alembic autogenerate diffs stable and
# make migrations reproducible across SQLite and Postgres.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Adds created_at / updated_at columns managed by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class RepositoryStatus(str, enum.Enum):
    """Lifecycle state of an ingested repository."""

    QUEUED = "queued"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class JobStatus(str, enum.Enum):
    """Lifecycle state of a single ingestion job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _enum_column(py_enum: type[enum.Enum], name: str) -> SAEnum:
    """Build an Enum column that stores the lowercase ``value`` (not the name)."""
    return SAEnum(
        py_enum,
        name=name,
        values_callable=lambda members: [m.value for m in members],
    )


class User(TimestampMixin, Base):
    """An authenticated user (created via Google OAuth in Issue 27)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255))
    google_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    avatar_url: Mapped[str | None] = mapped_column(String(1024))
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    repositories: Mapped[list[Repository]] = relationship(back_populates="owner")
    query_logs: Mapped[list[QueryLog]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


class Repository(TimestampMixin, Base):
    """A source repository that has been (or is being) ingested."""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    owner_login: Mapped[str | None] = mapped_column(String(255))
    default_branch: Mapped[str] = mapped_column(
        String(255), default="main", nullable=False
    )
    commit_sha: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[RepositoryStatus] = mapped_column(
        _enum_column(RepositoryStatus, "repository_status"),
        default=RepositoryStatus.QUEUED,
        nullable=False,
        index=True,
    )
    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    owner: Mapped[User | None] = relationship(back_populates="repositories")
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list[QueryLog]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Repository id={self.id} name={self.name!r} status={self.status}>"


class IngestionJob(TimestampMixin, Base):
    """A single run of the ingestion pipeline for a repository."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        _enum_column(JobStatus, "job_status"),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )
    branch: Mapped[str | None] = mapped_column(String(255))
    files_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    repository: Mapped[Repository] = relationship(back_populates="ingestion_jobs")

    def __repr__(self) -> str:
        return (
            f"<IngestionJob id={self.id} "
            f"repository_id={self.repository_id} status={self.status}>"
        )


class QueryLog(TimestampMixin, Base):
    """A record of one question asked against a repository and its answer."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    query_type: Mapped[str | None] = mapped_column(String(50))
    num_citations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    repository: Mapped[Repository] = relationship(back_populates="query_logs")
    user: Mapped[User | None] = relationship(back_populates="query_logs")

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} repository_id={self.repository_id}>"

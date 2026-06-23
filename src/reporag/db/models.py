"""
SQLAlchemy ORM models for RepoRAG.
Defines User, Repository, IngestionJob, QueryLog.
Supports both SQLite and Postgres gracefully.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, MetaData, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Standardized constraints to prevent alembic autogenerate issues
# when switching between sqlite and postgres dialects.
DB_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=DB_NAMING_CONVENTION)


class BaseTimestampModel:
    """Mixin to automatically populate creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class RepositoryStatus(enum.StrEnum):
    QUEUED = "queued"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def build_db_enum(enum_class: type[enum.Enum], enum_name: str) -> SAEnum:
    """Helper to register Python Enums natively in Postgres, using the string values."""
    return SAEnum(
        enum_class,
        name=enum_name,
        values_callable=lambda items: [i.value for i in items],
    )


class User(BaseTimestampModel, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    repositories: Mapped[list[Repository]] = relationship(back_populates="owner")
    query_logs: Mapped[list[QueryLog]] = relationship(back_populates="user")


class Repository(BaseTimestampModel, Base):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    url: Mapped[str] = mapped_column(String(255), nullable=False)
    default_branch: Mapped[str] = mapped_column(
        String(50), default="main", nullable=False
    )
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[RepositoryStatus] = mapped_column(
        build_db_enum(RepositoryStatus, "repository_status"),
        default=RepositoryStatus.QUEUED,
        nullable=False,
        index=True,
    )

    owner: Mapped[User] = relationship(back_populates="repositories")
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list[QueryLog]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )


class IngestionJob(BaseTimestampModel, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        build_db_enum(JobStatus, "job_status"),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    repository: Mapped[Repository] = relationship(back_populates="ingestion_jobs")


class QueryLog(BaseTimestampModel, Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="query_logs")
    user: Mapped[User | None] = relationship(back_populates="query_logs")

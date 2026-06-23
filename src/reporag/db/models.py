import enum
from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Custom naming convention for Alembic constraints (ensures SQLite/Postgres compatibility)
DB_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def build_db_enum(enum_class):
    """Helper to consistently create SQLAlchemy Enums with native naming conventions."""
    return Enum(enum_class, name=enum_class.__name__.lower(), create_constraint=True)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""

    metadata = MetaData(naming_convention=DB_NAMING_CONVENTION)


class BaseTimestampModel(Base):
    """Abstract base model that automatically maintains created_at and updated_at fields."""

    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RepositoryStatus(enum.StrEnum):
    QUEUED = "queued"
    CLONING = "cloning"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class User(BaseTimestampModel):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))

    # Relationships
    repositories: Mapped[list["Repository"]] = relationship(
        "Repository", back_populates="owner", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list["QueryLog"]] = relationship(
        "QueryLog", back_populates="user", cascade="all, delete-orphan"
    )


class Repository(BaseTimestampModel):
    __tablename__ = "repositories"

    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100), index=True)
    url: Mapped[str] = mapped_column(String(2048))
    status: Mapped[RepositoryStatus] = mapped_column(
        build_db_enum(RepositoryStatus), default=RepositoryStatus.QUEUED
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="repositories")
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        "IngestionJob", back_populates="repository", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list["QueryLog"]] = relationship(
        "QueryLog", back_populates="repository", cascade="all, delete-orphan"
    )


class IngestionJob(BaseTimestampModel):
    __tablename__ = "ingestion_jobs"

    repository_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        build_db_enum(JobStatus), default=JobStatus.PENDING
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="ingestion_jobs"
    )


class QueryLog(BaseTimestampModel):
    __tablename__ = "query_logs"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    repository_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    query_text: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="query_logs")
    repository: Mapped[Optional["Repository"]] = relationship(
        "Repository", back_populates="query_logs"
    )

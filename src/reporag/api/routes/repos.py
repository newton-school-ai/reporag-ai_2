"""Repository ingestion endpoints.

POST /api/v1/repos/ingest - Trigger async ingestion of a Git repository.
GET  /api/v1/repos         - List ingested repositories with status.
GET  /api/v1/repos/{id}    - Get repository details.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.db.models import (
    IngestionJob,
    JobStatus,
    Repository,
    RepositoryStatus,
    User,
)
from reporag.db.session import async_session_maker, get_db
from reporag.ingestion.cloner import CloneError, RepoCloner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["repos"])

# Placeholder identity used for every request until auth (Google OAuth + JWT,
# tracked separately) is wired up. Keeps a single well-known demo user so the
# ingestion endpoint's owner_id FK is satisfied without inventing per-request
# accounts.
_DEMO_USERNAME = "demo-user"
_DEMO_EMAIL = "demo-user@reporag.local"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Body for ``POST /api/v1/repos/ingest``."""

    url: str = Field(
        ...,
        description="Git remote URL (https://...) or local path to clone/discover.",
        examples=["https://github.com/octocat/Hello-World"],
    )
    name: str | None = Field(
        default=None,
        description="Display name for the repository. Derived from the URL when omitted.",
        max_length=100,
    )
    branch: str | None = Field(
        default=None,
        description="Branch to clone. Defaults to the repo's default branch.",
    )

    @field_validator("url")
    @classmethod
    def _url_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("url must be a non-empty string.")
        return value.strip()


class IngestResponse(BaseModel):
    """Response for ``POST /api/v1/repos/ingest``.

    Ingestion runs in a background task; the endpoint returns immediately
    with identifiers the client can poll via ``GET /api/v1/repos``.
    """

    repository_id: int
    job_id: int
    status: RepositoryStatus
    message: str = "Ingestion started."


class RepositoryOut(BaseModel):
    """A single repository row, as returned by the listing endpoints."""

    id: int
    name: str
    url: str
    status: RepositoryStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RepositoryListResponse(BaseModel):
    """Response for ``GET /api/v1/repos``."""

    repositories: list[RepositoryOut]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_repo_name(url: str) -> str:
    """Fall back to the last path segment of *url* when no name is given."""
    path = urlparse(url).path or url
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return name.removesuffix(".git") or url


async def _get_or_create_demo_user(db: AsyncSession) -> User:
    """Return the placeholder demo user, creating it on first use.

    TODO: remove once real auth resolves the current user from a validated
    JWT instead.
    """
    result = await db.execute(select(User).where(User.username == _DEMO_USERNAME))
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(username=_DEMO_USERNAME, email=_DEMO_EMAIL, hashed_password="unset")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _run_ingestion_job(
    repository_id: int, job_id: int, url: str, branch: str | None
) -> None:
    """Background task: clone *url* and discover its source files.

    Scope note: this wires up cloning + file discovery end to end with
    status tracking. Parsing, chunking, embedding, and graph/vector
    indexing aren't invoked here -- a repository reaches ``READY`` once it
    has been cloned and its source files discovered. Needs its own DB
    session since the request-scoped one is closed once the response is
    sent.
    """
    async with async_session_maker() as db:
        repository = await db.get(Repository, repository_id)
        job = await db.get(IngestionJob, job_id)
        if repository is None or job is None:
            logger.error(
                "Ingestion job %s / repository %s vanished before it could run.",
                job_id,
                repository_id,
            )
            return

        repository.status = RepositoryStatus.CLONING
        job.status = JobStatus.IN_PROGRESS
        await db.commit()

        try:
            manifest = await asyncio.to_thread(
                RepoCloner().clone_and_discover, url, branch
            )
            logger.info(
                "Ingestion job %s discovered %d files for repository %s.",
                job_id,
                len(manifest),
                repository_id,
            )
            repository.status = RepositoryStatus.READY
            job.status = JobStatus.COMPLETED
        except CloneError as exc:
            logger.error("Ingestion job %s failed: %s", job_id, exc)
            repository.status = RepositoryStatus.FAILED
            job.status = JobStatus.FAILED
        except Exception:
            logger.exception(
                "Ingestion job %s failed with an unexpected error.", job_id
            )
            repository.status = RepositoryStatus.FAILED
            job.status = JobStatus.FAILED

        await db.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/repos/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_repo(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> IngestResponse:
    """Register a repository and trigger its ingestion in the background.

    Returns immediately with ``repository_id`` / ``job_id`` so the client
    can poll ``GET /api/v1/repos`` for status rather than blocking on the
    clone + discovery work.
    """
    owner = await _get_or_create_demo_user(db)

    repository = Repository(
        owner_id=owner.id,
        name=request.name or _derive_repo_name(request.url),
        url=request.url,
        status=RepositoryStatus.QUEUED,
    )
    db.add(repository)
    await db.flush()

    job = IngestionJob(repository_id=repository.id, status=JobStatus.PENDING)
    db.add(job)
    await db.commit()
    await db.refresh(repository)
    await db.refresh(job)

    background_tasks.add_task(
        _run_ingestion_job, repository.id, job.id, request.url, request.branch
    )

    return IngestResponse(
        repository_id=repository.id,
        job_id=job.id,
        status=repository.status,
    )


@router.get("/repos", response_model=RepositoryListResponse)
async def list_repos(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
) -> RepositoryListResponse:
    """List ingested repositories, most recently created first."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    total = (
        await db.execute(select(func.count()).select_from(Repository))
    ).scalar_one()
    result = await db.execute(
        select(Repository)
        .order_by(Repository.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    repositories = result.scalars().all()

    return RepositoryListResponse(
        repositories=[RepositoryOut.model_validate(r) for r in repositories],
        total=total,
    )


@router.get("/repos/{repo_id}", response_model=RepositoryOut)
async def get_repo(
    repo_id: int, db: Annotated[AsyncSession, Depends(get_db)]
) -> RepositoryOut:
    """Get a single repository's details by id."""
    repository = await db.get(Repository, repo_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found."
        )
    return RepositoryOut.model_validate(repository)

"""Repository ingestion endpoints.

POST /api/v1/repos/ingest - Trigger async ingestion of a Git repository.
GET  /api/v1/repos         - List ingested repositories with status.
GET  /api/v1/repos/{id}    - Get repository details.

Why
---
Ingesting a repository means cloning it, parsing every source file, and
building the symbol table and code graph. That takes far longer than an
HTTP request should, so POST /ingest records the repository, returns
202 Accepted immediately, and does the work in a background task. The
caller polls GET /repos to watch status advance.

Design
------
* Status is the contract: A repository moves
  queued -> cloning -> processing -> ready, or lands on failed.
  That progression is the only thing a client needs to understand to use
  this API correctly.
* Every run gets a job row: ingestion_jobs records one attempt,
  moving pending -> in_progress -> completed or failed.
* The background task owns its own session: The request-scoped
  AsyncSession from get_db is closed once the response is sent, so the
  task opens a fresh one from async_session_maker.
* Failures are recorded, not raised: Every failure path marks the
  repository and its job failed and logs the reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repos", tags=["repositories"])

# Until Google OAuth lands (Issue 27) there is no authenticated principal,
# but Repository.owner_id is a non-nullable foreign key. Ingested repos are
# attributed to this placeholder account.
_SYSTEM_USER_EMAIL = "system@reporag.local"
_SYSTEM_USERNAME = "system"

_ALLOWED_SCHEMES = ("https://", "http://")

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Body of POST /api/v1/repos/ingest.

    Attributes:
        repo_url: HTTPS or HTTP URL of a Git repository to clone.
        branch: Branch to ingest. None lets the remote default apply.
        name: Optional display name for the repository.
    """

    repo_url: str = Field(
        ...,
        validation_alias=AliasChoices("repo_url", "url"),
        min_length=1,
        max_length=2048,
        description="HTTP(S) URL of the Git repository to ingest.",
        examples=["https://github.com/pallets/click"],
    )
    branch: str | None = Field(
        default=None,
        max_length=255,
        description="Branch to ingest. Defaults to the remote's default branch.",
        examples=["main"],
    )
    name: str | None = Field(
        default=None,
        max_length=100,
        description="Display name for the repository. Derived from the URL when omitted.",
    )

    @field_validator("repo_url")
    @classmethod
    def _validate_repo_url(cls, value: str) -> str:
        """Reject anything that is not an HTTP(S) clone URL."""
        cleaned = value.strip()
        if not cleaned.lower().startswith(_ALLOWED_SCHEMES):
            raise ValueError("repo_url must start with https:// or http://")
        return cleaned


class RepositoryResponse(BaseModel):
    """One repository row, as returned by the API."""

    id: int
    name: str
    url: str
    status: RepositoryStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    """Response of POST /api/v1/repos/ingest."""

    repository: RepositoryResponse
    job_id: int
    branch: str | None = None
    message: str = "Ingestion queued. Poll GET /api/v1/repos for status updates."
    repository_id: int | None = None
    status: RepositoryStatus | None = None

    @model_validator(mode="after")
    def _populate_convenience_fields(self) -> IngestResponse:
        if self.repository_id is None:
            self.repository_id = self.repository.id
        if self.status is None:
            self.status = self.repository.status
        return self

    model_config = {"from_attributes": True}


class RepositoryListResponse(BaseModel):
    """Page of repositories from GET /api/v1/repos."""

    repositories: list[RepositoryResponse]
    total: int
    limit: int = _DEFAULT_PAGE_SIZE
    offset: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_name_from_url(url: str) -> str:
    """Extract a repository name from its URL."""
    path = urlparse(url).path.rstrip("/")
    if not path:
        return "repository"
    tail = path.rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return tail or "repository"


async def _get_or_create_system_user(session: AsyncSession) -> User:
    """Return the placeholder system user, creating it on first use."""
    result = await session.execute(
        select(User).where(User.username == _SYSTEM_USERNAME)
    )
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(
        username=_SYSTEM_USERNAME,
        email=_SYSTEM_USER_EMAIL,
        hashed_password="!",
    )
    session.add(user)
    try:
        await session.commit()
        await session.refresh(user)
        return user
    except Exception:
        await session.rollback()
        result = await session.execute(
            select(User).where(User.username == _SYSTEM_USERNAME)
        )
        user = result.scalar_one_or_none()
        if user is not None:
            return user
        raise


async def _set_status(
    repo_id: int,
    repo_status: RepositoryStatus,
    *,
    job_id: int | None = None,
    detail: str = "",
) -> None:
    """Update repository and job status in a dedicated transaction."""
    async with async_session_maker() as session:
        repo = await session.get(Repository, repo_id)
        if repo is not None:
            repo.status = repo_status

        if job_id is not None:
            job = await session.get(IngestionJob, job_id)
            if job is not None:
                if repo_status in (
                    RepositoryStatus.CLONING,
                    RepositoryStatus.PROCESSING,
                ):
                    job.status = JobStatus.IN_PROGRESS
                elif repo_status == RepositoryStatus.READY:
                    job.status = JobStatus.COMPLETED
                elif repo_status == RepositoryStatus.FAILED:
                    job.status = JobStatus.FAILED

        await session.commit()
        logger.info(
            "Repository %s status -> %s (job=%s%s)",
            repo_id,
            repo_status,
            job_id,
            f", {detail}" if detail else "",
        )


@dataclass
class _IngestionStats:
    """Counts gathered while parsing a repository."""

    files: int = 0
    symbols: int = 0
    chunks: int = 0
    languages: list[str] = field(default_factory=list)
    skipped: int = 0

    def summary(self) -> str:
        """One-line description for the status log."""
        return (
            f"{self.files} file(s), {self.symbols} symbol(s), "
            f"{self.chunks} chunk(s), {self.skipped} skipped "
            f"[{', '.join(self.languages) or 'none'}]"
        )


def _parse_repository(repo_url: str, branch: str | None) -> _IngestionStats:
    """Clone repo_url and parse every discovered source file."""
    from reporag.ingestion.chunker import SemanticChunker
    from reporag.ingestion.cloner import RepoCloner
    from reporag.ingestion.symbol_extractor import SymbolExtractor

    cloner = RepoCloner()
    try:
        manifest = cloner.clone_and_discover(repo_url, branch=branch)
        stats = _IngestionStats(
            files=len(manifest),
            languages=sorted({entry.language for entry in manifest}),
        )

        extractor = SymbolExtractor()
        chunker = SemanticChunker()

        base_path = cloner.last_clone_path
        for entry in manifest:
            try:
                target_path = (
                    (base_path / entry.path).as_posix() if base_path else entry.path
                )
                stats.symbols += len(
                    extractor.extract_from_file(target_path, language=entry.language)
                )
                stats.chunks += len(
                    chunker.chunk_file(target_path, language=entry.language)
                )
            except Exception as exc:
                stats.skipped += 1
                logger.debug("Skipped %s: %s", entry.path, exc)

        return stats
    finally:
        cloner.cleanup()


async def run_ingestion(
    repo_id: int, repo_url: str, branch: str | None, job_id: int | None = None
) -> None:
    """Ingest a repository in the background, recording progress."""
    try:
        await _set_status(repo_id, RepositoryStatus.CLONING, job_id=job_id)
        stats = await run_in_threadpool(_parse_repository, repo_url, branch)
        await _set_status(
            repo_id,
            RepositoryStatus.PROCESSING,
            job_id=job_id,
            detail=stats.summary(),
        )

        if stats.files == 0:
            await _set_status(
                repo_id,
                RepositoryStatus.FAILED,
                job_id=job_id,
                detail="no parseable source files found",
            )
            return

        await _set_status(
            repo_id, RepositoryStatus.READY, job_id=job_id, detail=stats.summary()
        )
    except Exception as exc:
        logger.exception("Ingestion failed for repository %s", repo_id)
        await _set_status(
            repo_id, RepositoryStatus.FAILED, job_id=job_id, detail=str(exc)
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a repository for ingestion",
    response_description="The queued repository and what happens next.",
)
async def ingest_repository(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> IngestResponse:
    """Queue repo_url for ingestion and return immediately.

    Responds 202 Accepted once the repository is recorded; the clone and
    parse happen in the background.
    """
    owner = await _get_or_create_system_user(session)

    name = payload.name or _repo_name_from_url(payload.repo_url)
    repository = Repository(
        owner_id=owner.id,
        name=name,
        url=payload.repo_url,
        status=RepositoryStatus.QUEUED,
    )
    session.add(repository)
    await session.flush()

    job = IngestionJob(repository_id=repository.id, status=JobStatus.PENDING)
    session.add(job)
    await session.commit()
    await session.refresh(repository)
    await session.refresh(job)

    background_tasks.add_task(
        run_ingestion, repository.id, payload.repo_url, payload.branch, job.id
    )

    return IngestResponse(
        repository=RepositoryResponse.model_validate(repository),
        job_id=job.id,
        branch=payload.branch,
        message=(
            "Ingestion queued. Poll GET /api/v1/repos/"
            f"{repository.id} for status updates."
        ),
    )


@router.get(
    "",
    response_model=RepositoryListResponse,
    summary="List ingested repositories",
    response_description="A page of repositories with their ingestion status.",
)
async def list_repositories(
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[
        int, Query(ge=1, le=_MAX_PAGE_SIZE, description="Page size.")
    ] = _DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
) -> RepositoryListResponse:
    """Return a page of repositories, newest first."""
    total = await session.scalar(select(func.count()).select_from(Repository)) or 0
    result = await session.execute(
        select(Repository)
        .order_by(Repository.created_at.desc(), Repository.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return RepositoryListResponse(
        repositories=[
            RepositoryResponse.model_validate(repo) for repo in result.scalars().all()
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{repo_id}",
    response_model=RepositoryResponse,
    summary="Get one repository",
    response_description="The requested repository.",
    responses={404: {"description": "No repository with that id."}},
)
async def get_repository(
    repo_id: Annotated[int, Path(ge=1, description="Repository id.")],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> RepositoryResponse:
    """Return a single repository by id."""
    repository = await session.get(Repository, repo_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {repo_id} not found",
        )
    return RepositoryResponse.model_validate(repository)

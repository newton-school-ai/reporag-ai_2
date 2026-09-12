"""Repository ingestion endpoints.

POST /api/v1/repos/ingest - Trigger async ingestion of a Git repository.
GET  /api/v1/repos         - List ingested repositories with status.
GET  /api/v1/repos/{id}    - Get repository details.

Why
---
Ingesting a repository means cloning it, parsing every source file, and
building the symbol table and code graph. That takes far longer than an
HTTP request should, so ``POST /ingest`` records the repository, returns
``202 Accepted`` immediately, and does the work in a background task. The
caller polls ``GET /repos`` to watch ``status`` advance.

Design
------
* **Status is the contract.** A repository moves
  ``queued -> cloning -> processing -> ready``, or lands on ``failed``.
  That progression is the only thing a client needs to understand to use
  this API correctly.
* **Every run gets a job row.** ``ingestion_jobs`` records one attempt,
  moving ``pending -> in_progress -> completed`` or ``failed``. The
  repository's status says what the repository is now; the job says what
  this particular attempt did, which is what lets a re-ingest be told
  apart from the run before it.
* **The background task owns its own session.** The request-scoped
  ``AsyncSession`` from :func:`~reporag.db.session.get_db` is closed once
  the response is sent, so the task opens a fresh one from
  ``async_session_maker``. Reusing the request session would fail exactly
  when the work started succeeding.
* **Failures are recorded, not raised.** Nothing is listening when a
  background task raises. Every failure path marks the repository and its
  job ``failed`` and logs the reason, so a stuck ingest is visible through
  ``GET /repos`` rather than silently doing nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repos", tags=["repositories"])

# Until Google OAuth lands (Issue 27) there is no authenticated principal,
# but Repository.owner_id is a non-nullable foreign key. Ingested repos are
# attributed to this placeholder account; Issue 27 replaces it with the
# caller identity from the JWT.
_SYSTEM_USER_EMAIL = "system@reporag.local"
_SYSTEM_USERNAME = "system"

# Schemes accepted for cloning. Anything else (git://, ssh://, file://) is
# rejected at validation time rather than failing deep inside the clone.
_ALLOWED_SCHEMES = ("https://", "http://")

# Page size for GET /repos. The ceiling is a bound on response size, not a
# policy: a client that wants everything pages through it.
_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Body of ``POST /api/v1/repos/ingest``.

    Attributes:
        repo_url: HTTPS URL of a Git repository to clone.
        branch: Branch to ingest. ``None`` lets the remote default apply.
    """

    repo_url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="HTTPS URL of the Git repository to ingest.",
        examples=["https://github.com/pallets/click"],
    )
    branch: str | None = Field(
        default=None,
        max_length=255,
        description="Branch to ingest. Defaults to the remote's default branch.",
        examples=["main"],
    )

    @field_validator("repo_url")
    @classmethod
    def _validate_repo_url(cls, value: str) -> str:
        """Reject anything that is not an HTTP(S) clone URL."""
        cleaned = value.strip()
        if not cleaned.lower().startswith(_ALLOWED_SCHEMES):
            raise ValueError("repo_url must start with https:// or http://")
        return cleaned

    @field_validator("branch")
    @classmethod
    def _validate_branch(cls, value: str | None) -> str | None:
        """Normalise an empty branch to None and reject shell-unsafe names."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if any(ch.isspace() for ch in cleaned):
            raise ValueError("branch must not contain whitespace")
        return cleaned


class RepositoryResponse(BaseModel):
    """A repository row as returned by the API.

    Attributes:
        id: Database identifier, used as ``repo_id`` when querying.
        name: Repository name derived from its URL.
        url: The clone URL.
        status: Current ingestion status.
        created_at: When the repository was first queued.
        updated_at: When its status last changed -- how a client tells a
            run that is still working from one that stalled.
    """

    id: int
    name: str
    url: str
    status: RepositoryStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RepositoryListResponse(BaseModel):
    """Body of ``GET /api/v1/repos``.

    An object rather than a bare array so the page can carry ``total``
    alongside it: a client showing "12 of 340" cannot get that from the
    slice it was handed.

    Attributes:
        repositories: The requested page, newest first.
        total: Repositories that exist, not just the ones on this page.
        limit: Page size applied, echoed back so a client can page without
            tracking what it asked for.
        offset: Offset this page started at.
    """

    repositories: list[RepositoryResponse]
    total: int
    limit: int
    offset: int


class IngestResponse(BaseModel):
    """Acknowledgement returned by ``POST /api/v1/repos/ingest``.

    Attributes:
        repository: The queued repository row.
        job_id: The ingestion job recording this attempt.
        branch: The branch that will be ingested, echoed back.
        message: What happens next.
    """

    repository: RepositoryResponse
    job_id: int
    branch: str | None = None
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_name_from_url(repo_url: str) -> str:
    """Derive a display name from a clone URL.

    ``https://github.com/pallets/click.git`` becomes ``click``.
    """
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    # A URL that is nothing but a host leaves an empty tail; never store "".
    return tail[:100] or "repository"


async def _get_or_create_system_user(session: AsyncSession) -> User:
    """Return the placeholder owner, creating it on first ingest.

    Repository.owner_id is non-nullable and authentication does not exist
    until Issue 27, so every repository is attributed to one shared system
    account in the meantime.
    """
    result = await session.execute(select(User).where(User.email == _SYSTEM_USER_EMAIL))
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(
        username=_SYSTEM_USERNAME,
        email=_SYSTEM_USER_EMAIL,
        # Not a credential: this account cannot be logged into. Issue 27
        # replaces it with real OAuth identities.
        hashed_password="!",
    )
    session.add(user)
    await session.flush()
    return user


# Repository status -> the job status that goes with it. A job is a single
# attempt, so the three working states all map to ``in_progress``; only the
# terminal ones differ.
_JOB_STATUS_FOR = {
    RepositoryStatus.QUEUED: JobStatus.PENDING,
    RepositoryStatus.CLONING: JobStatus.IN_PROGRESS,
    RepositoryStatus.PROCESSING: JobStatus.IN_PROGRESS,
    RepositoryStatus.READY: JobStatus.COMPLETED,
    RepositoryStatus.FAILED: JobStatus.FAILED,
}


async def _set_status(
    repo_id: int,
    new_status: RepositoryStatus,
    *,
    job_id: int | None = None,
    detail: str = "",
) -> None:
    """Advance a repository, and the job tracking this attempt, together.

    Opens its own session because the request-scoped one is already closed
    by the time the task runs.

    Both rows move in one transaction. Committing them separately would
    leave a window where a crash could park a repository on ``failed`` with
    its job still reading ``in_progress``, and nothing would ever reconcile
    the two.

    Args:
        repo_id: Repository row to advance.
        new_status: Status to move it to.
        job_id: Ingestion job for this attempt, moved to the matching job
            status. ``None`` updates the repository alone.
        detail: Context for the log line; not persisted.
    """
    async with async_session_maker() as session:
        repo = await session.get(Repository, repo_id)
        if repo is None:
            logger.warning("Repository %s vanished during ingestion", repo_id)
            return
        repo.status = new_status

        if job_id is not None:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                # The repository outliving its job is not fatal -- status on
                # the repository is what clients poll -- but it means the
                # job history for this run is gone.
                logger.warning("Ingestion job %s vanished during ingestion", job_id)
            else:
                job.status = _JOB_STATUS_FOR[new_status]

        await session.commit()

    if detail:
        logger.info("Repository %s -> %s: %s", repo_id, new_status, detail)
    else:
        logger.info("Repository %s -> %s", repo_id, new_status)


class _IngestionStats(BaseModel):
    """Counts produced by one ingestion run.

    Attributes:
        files: Source files discovered in the clone.
        symbols: Functions, classes and methods extracted.
        chunks: AST-aware chunks produced for embedding.
        languages: Distinct languages discovered.
        skipped: Files that could not be parsed.
    """

    files: int = 0
    symbols: int = 0
    chunks: int = 0
    languages: list[str] = Field(default_factory=list)
    skipped: int = 0

    def summary(self) -> str:
        """One-line description for the status log."""
        return (
            f"{self.files} file(s), {self.symbols} symbol(s), "
            f"{self.chunks} chunk(s), {self.skipped} skipped "
            f"[{', '.join(self.languages) or 'none'}]"
        )


def _parse_repository(repo_url: str, branch: str | None) -> _IngestionStats:
    """Clone *repo_url* and parse every discovered source file.

    Performs the local half of ingestion -- clone, discover, extract symbols,
    chunk. Embedding and index population are owned by the embedding layer
    (Issues 13-15) and are not wired in here.

    A file that fails to parse is counted in ``skipped`` rather than aborting
    the run; one unparseable file in a large repository should not cost the
    whole ingest.

    Blocking: runs subprocess git and CPU-bound parsing, so callers must
    dispatch it to a worker thread.

    Args:
        repo_url: Clone URL.
        branch: Branch to clone, or None for the remote default.

    Returns:
        The counts gathered during the run.
    """
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

        for entry in manifest:
            try:
                stats.symbols += len(
                    extractor.extract_from_file(entry.path, language=entry.language)
                )
                stats.chunks += len(
                    chunker.chunk_file(entry.path, language=entry.language)
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
    """Ingest a repository in the background, recording progress.

    Drives the row through ``cloning`` and ``processing`` to ``ready``, or to
    ``failed`` with the reason logged. Never raises: a background task has no
    caller to catch anything, so every failure is written to the row instead,
    where ``GET /api/v1/repos/{id}`` can surface it.

    Args:
        repo_id: Row to update as the work progresses.
        repo_url: Clone URL.
        branch: Branch to ingest, or None for the remote default.
        job_id: Ingestion job recording this attempt, advanced in step with
            the repository. Optional so a caller can drive a repository
            without a job row.
    """
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
    """Queue *payload.repo_url* for ingestion and return immediately.

    Responds ``202 Accepted`` once the repository is recorded; the clone and
    parse happen in the background. Poll ``GET /api/v1/repos/{id}`` to watch
    ``status`` advance from ``queued`` to ``ready`` (or ``failed``).
    """
    owner = await _get_or_create_system_user(session)

    repository = Repository(
        owner_id=owner.id,
        name=_repo_name_from_url(payload.repo_url),
        url=payload.repo_url,
        status=RepositoryStatus.QUEUED,
    )
    session.add(repository)
    # Flush rather than commit so the job gets the repository's id while
    # both rows still land in one transaction; a commit here could leave a
    # repository with no job behind it.
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
    """Return a page of repositories, newest first.

    Paginated rather than returning everything: an installation that has
    ingested a few thousand repositories should not serialise all of them
    to answer a poll for the status of one recent ingest. ``total`` is
    reported alongside so a client can still show how many exist.
    """
    total = await session.scalar(select(func.count()).select_from(Repository)) or 0
    result = await session.execute(
        select(Repository)
        # Newest first, with id as the tiebreaker: rows created inside the
        # same clock tick would otherwise order arbitrarily, and a page
        # boundary landing mid-tie would drop or repeat a row.
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
    """Return a single repository by id.

    Raises:
        HTTPException: 404 when no repository has that id.
    """
    repository = await session.get(Repository, repo_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {repo_id} not found",
        )
    return RepositoryResponse.model_validate(repository)

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
  ``queued -> cloning -> processing -> ready``, or lands on ``failed`` with
  the reason recorded. That progression is the only thing a client needs to
  understand to use this API correctly.
* **The background task owns its own session.** The request-scoped
  ``AsyncSession`` from :func:`~reporag.db.session.get_db` is closed once
  the response is sent, so the task opens a fresh one from
  ``async_session_maker``. Reusing the request session would fail exactly
  when the work started succeeding.
* **Failures are recorded, not raised.** Nothing is listening when a
  background task raises. Every failure path writes ``FAILED`` and the
  error text to the row, so the state is visible through ``GET /repos``.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.db.models import Repository, RepositoryStatus, User
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
    """

    id: int
    name: str
    url: str
    status: RepositoryStatus

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    """Acknowledgement returned by ``POST /api/v1/repos/ingest``.

    Attributes:
        repository: The queued repository row.
        branch: The branch that will be ingested, echoed back.
        message: What happens next.
    """

    repository: RepositoryResponse
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


async def _set_status(
    repo_id: int, new_status: RepositoryStatus, *, detail: str = ""
) -> None:
    """Update a repository's status from a background task.

    Opens its own session because the request-scoped one is already closed
    by the time the task runs.
    """
    async with async_session_maker() as session:
        repo = await session.get(Repository, repo_id)
        if repo is None:
            logger.warning("Repository %s vanished during ingestion", repo_id)
            return
        repo.status = new_status
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


async def run_ingestion(repo_id: int, repo_url: str, branch: str | None) -> None:
    """Ingest a repository in the background, recording progress.

    Drives the row through ``cloning`` and ``processing`` to ``ready``, or to
    ``failed`` with the reason logged. Never raises: a background task has no
    caller to catch anything, so every failure is written to the row instead,
    where ``GET /api/v1/repos/{id}`` can surface it.

    Args:
        repo_id: Row to update as the work progresses.
        repo_url: Clone URL.
        branch: Branch to ingest, or None for the remote default.
    """
    try:
        await _set_status(repo_id, RepositoryStatus.CLONING)
        stats = await run_in_threadpool(_parse_repository, repo_url, branch)
        await _set_status(repo_id, RepositoryStatus.PROCESSING, detail=stats.summary())

        if stats.files == 0:
            await _set_status(
                repo_id,
                RepositoryStatus.FAILED,
                detail="no parseable source files found",
            )
            return

        await _set_status(repo_id, RepositoryStatus.READY, detail=stats.summary())
    except Exception as exc:
        logger.exception("Ingestion failed for repository %s", repo_id)
        await _set_status(repo_id, RepositoryStatus.FAILED, detail=str(exc))


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
    await session.commit()
    await session.refresh(repository)

    background_tasks.add_task(
        run_ingestion, repository.id, payload.repo_url, payload.branch
    )

    return IngestResponse(
        repository=RepositoryResponse.model_validate(repository),
        branch=payload.branch,
        message=(
            "Ingestion queued. Poll GET /api/v1/repos/"
            f"{repository.id} for status updates."
        ),
    )


@router.get(
    "",
    response_model=list[RepositoryResponse],
    summary="List ingested repositories",
    response_description="Every known repository with its ingestion status.",
)
async def list_repositories(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> list[RepositoryResponse]:
    """Return every repository with its current ingestion status."""
    result = await session.execute(select(Repository).order_by(Repository.id))
    return [RepositoryResponse.model_validate(repo) for repo in result.scalars().all()]


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

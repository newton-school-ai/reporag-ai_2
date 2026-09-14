"""Repository ingestion and management endpoints.

POST /api/v1/repos/ingest - Trigger async ingestion of a Git repository.
GET  /api/v1/repos         - List ingested repositories with status.
GET  /api/v1/repos/{id}    - Get repository details.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.models import (
    RepoIngestRequest,
    RepoIngestResponse,
    RepositoryListResponse,
    RepositoryResponse,
)
from reporag.db.models import (
    IngestionJob,
    JobStatus,
    Repository,
    RepositoryStatus,
    User,
)
from reporag.db.session import async_session_maker, get_db
from reporag.ingestion.chunker import SemanticChunker
from reporag.ingestion.cloner import RepoCloner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repos", tags=["Repositories"])


def _extract_repo_name(repo_url: str) -> str:
    """Extract a clean repository name from a git URL or local path."""
    clean_url = repo_url.strip().rstrip("/")
    if os.path.exists(clean_url):
        return Path(clean_url).name or "local-repo"
    parsed = urlparse(clean_url)
    path = parsed.path.rstrip("/")
    name = path.split("/")[-1] if path else "repository"
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repository"


async def _ensure_default_user(session: AsyncSession) -> int:
    """Ensure at least one user exists to satisfy foreign key constraints."""
    result = await session.execute(select(User).limit(1))
    user = result.scalar_one_or_none()
    if user is not None:
        return user.id

    default_user = User(
        username="system",
        email="system@reporag.ai",
        hashed_password="placeholder-hash",
    )
    session.add(default_user)
    await session.commit()
    await session.refresh(default_user)
    return default_user.id


async def run_ingestion_pipeline(
    repo_id: int,
    repo_url: str,
    branch: str | None = None,
    shallow: bool = True,
) -> None:
    """Execute the end-to-end repository ingestion pipeline in the background."""
    cloner = RepoCloner()
    cloned_path: Path | None = None

    async with async_session_maker() as session:
        try:
            # 1. Update status to processing
            repo_res = await session.execute(
                select(Repository).where(Repository.id == repo_id)
            )
            repo = repo_res.scalar_one_or_none()
            if not repo:
                logger.error(
                    "Background ingestion aborted: Repo ID %d not found", repo_id
                )
                return

            repo.status = RepositoryStatus.PROCESSING
            job_res = await session.execute(
                select(IngestionJob)
                .where(IngestionJob.repository_id == repo_id)
                .order_by(IngestionJob.id.desc())
            )
            job = job_res.scalars().first()
            if job:
                job.status = JobStatus.IN_PROGRESS
            await session.commit()

            # 2. Clone repository & discover files
            logger.info(
                "Starting clone for repo %d (%s, branch=%s)", repo_id, repo_url, branch
            )
            file_manifest = cloner.clone_and_discover(
                repo_url=repo_url,
                branch=branch,
                shallow=shallow,
            )
            cloned_path = cloner.last_clone_path
            logger.info(
                "Discovered %d source files in repo %d", len(file_manifest), repo_id
            )

            # 3. Chunk files
            chunker = SemanticChunker()
            all_chunks = []
            if cloned_path and cloned_path.exists():
                for file_entry in file_manifest:
                    full_file_path = cloned_path / file_entry.path
                    if full_file_path.exists() and full_file_path.is_file():
                        try:
                            content = full_file_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                            chunks = chunker.chunk_file(
                                file_entry.path, content, file_entry.language
                            )
                            all_chunks.extend(chunks)
                        except Exception as parse_err:  # noqa: BLE001
                            logger.warning(
                                "Failed to chunk file %s: %s",
                                file_entry.path,
                                parse_err,
                            )

            logger.info("Generated %d chunks for repo %d", len(all_chunks), repo_id)

            # 4. Mark status as ready
            repo_res = await session.execute(
                select(Repository).where(Repository.id == repo_id)
            )
            repo = repo_res.scalar_one_or_none()
            if repo:
                repo.status = RepositoryStatus.READY
            if job:
                job.status = JobStatus.COMPLETED
            await session.commit()
            logger.info("Successfully completed ingestion for repo %d", repo_id)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Ingestion failed for repo %d: %s", repo_id, exc, exc_info=True
            )
            try:
                repo_res = await session.execute(
                    select(Repository).where(Repository.id == repo_id)
                )
                repo = repo_res.scalar_one_or_none()
                if repo:
                    repo.status = RepositoryStatus.FAILED
                if job:
                    job.status = JobStatus.FAILED
                await session.commit()
            except Exception as db_err:  # noqa: BLE001
                logger.error("Failed to update failure status in DB: %s", db_err)
        finally:
            if cloned_path and cloned_path.exists():
                shutil.rmtree(cloned_path, ignore_errors=True)


@router.post(
    "/ingest",
    response_model=RepoIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a Git repository",
    description="Trigger asynchronous background ingestion of a remote or local Git repository.",
)
async def ingest_repository(
    payload: RepoIngestRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RepoIngestResponse:
    """Accept repository URL and branch, schedule async ingestion."""
    repo_name = _extract_repo_name(payload.repo_url)
    owner_id = await _ensure_default_user(db)

    # Check if a repository record with this URL already exists
    result = await db.execute(
        select(Repository).where(Repository.url == payload.repo_url)
    )
    repo = result.scalar_one_or_none()

    if repo is None:
        repo = Repository(
            owner_id=owner_id,
            name=repo_name,
            url=payload.repo_url,
            status=RepositoryStatus.QUEUED,
        )
        db.add(repo)
        await db.commit()
        await db.refresh(repo)
    else:
        repo.status = RepositoryStatus.QUEUED
        await db.commit()
        await db.refresh(repo)

    job = IngestionJob(repository_id=repo.id, status=JobStatus.PENDING)
    db.add(job)
    await db.commit()

    # Dispatch to background task runner
    background_tasks.add_task(
        run_ingestion_pipeline,
        repo_id=repo.id,
        repo_url=payload.repo_url,
        branch=payload.branch,
        shallow=payload.shallow,
    )

    return RepoIngestResponse(
        repo_id=repo.id,
        name=repo.name,
        url=repo.url,
        status=repo.status.value if hasattr(repo.status, "value") else str(repo.status),
        message="Repository ingestion queued successfully.",
    )


@router.get(
    "",
    response_model=RepositoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List ingested repositories",
    description="Retrieve all tracked repositories along with their current indexing status.",
)
async def list_repositories(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RepositoryListResponse:
    """List all ingested repositories with status."""
    result = await db.execute(select(Repository).order_by(Repository.id.desc()))
    repos = result.scalars().all()

    items = [
        RepositoryResponse(
            id=r.id,
            name=r.name,
            url=r.url,
            status=r.status.value if hasattr(r.status, "value") else str(r.status),
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in repos
    ]
    return RepositoryListResponse(repositories=items, total=len(items))


@router.get(
    "/{repo_id}",
    response_model=RepositoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get repository details",
    description="Retrieve metadata and ingestion status for a specific repository by ID.",
)
async def get_repository(
    repo_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RepositoryResponse:
    """Retrieve details for a single repository."""
    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {repo_id} not found.",
        )

    return RepositoryResponse(
        id=repo.id,
        name=repo.name,
        url=repo.url,
        status=repo.status.value if hasattr(repo.status, "value") else str(repo.status),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )

"""Query endpoints for RepoRAG.

POST /api/v1/query - Accept a natural language question about a repository,
run the full RAG pipeline, and return a cited answer with line-level references.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.models import (
    CitationResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
)
from reporag.config import settings
from reporag.db.models import QueryLog, Repository, User
from reporag.db.session import get_db
from reporag.generation.context_assembler import ContextAssembler
from reporag.generation.generator import AnswerGenerator
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.vector_search import RetrievalResult, VectorSearch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


async def _get_or_create_default_user(session: AsyncSession) -> int:
    """Ensure a user exists for logging."""
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


def _execute_retrieval(
    question: str, repo_id: int | str, top_k: int, strategy: str | None = None
) -> list[RetrievalResult]:
    """Execute search across retrieval stores with graceful degradation."""
    results: list[RetrievalResult] = []
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=1.5)
        # Probe connectivity quickly before full embed/search
        client.get_collections()
        vector_search = VectorSearch(client=client)
        results = vector_search.search(query=question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "Vector search not available or empty (%s); continuing with available context.",
            exc,
        )

    return results


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query repository with natural language",
    description="Run the full RAG pipeline (retrieval, context assembly, prompt building, and LLM generation with line-level citations).",
)
async def query_repository(
    payload: QueryRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QueryResponse:
    """Execute RAG query against repository code and return cited answer."""
    start_time = time.monotonic()

    # 1. Validate repository if numeric ID provided
    repo_db_id: int | None = None
    if isinstance(payload.repo_id, int) or (
        isinstance(payload.repo_id, str) and payload.repo_id.isdigit()
    ):
        repo_db_id = int(payload.repo_id)
        repo_res = await db.execute(
            select(Repository).where(Repository.id == repo_db_id)
        )
        repo = repo_res.scalar_one_or_none()
        if repo is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository with ID {payload.repo_id} not found.",
            )

    # 2. Retrieve relevant context chunks
    retrieved_chunks = _execute_retrieval(
        question=payload.question,
        repo_id=payload.repo_id,
        top_k=payload.top_k,
        strategy=payload.strategy,
    )

    # 3. Assemble context
    assembler = ContextAssembler(max_tokens=4000)
    context_str = assembler.assemble(retrieved_chunks)

    # 4. Build prompt
    builder = PromptBuilder()
    built_prompt = builder.build_prompt(
        query=payload.question,
        context=context_str,
    )

    # 5. Generate answer & citations
    generator = AnswerGenerator()
    if payload.include_citations:
        answered = generator.generate_with_citations(built_prompt)
        answer_text = answered.answer
        citation_report = answered.citations
        gen_result = answered.generation
    else:
        gen_result = generator.generate(built_prompt)
        answer_text = gen_result.text
        citation_report = None

    elapsed = time.monotonic() - start_time

    # Format citations
    citations_list: list[CitationResponse] = []
    coverage = 0.0
    valid_count = 0
    invalid_count = 0

    if citation_report is not None:
        coverage = citation_report.coverage
        valid_count = citation_report.valid_count
        invalid_count = citation_report.invalid_count
        for c in citation_report.citations:
            citations_list.append(
                CitationResponse(
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    snippet=c.snippet,
                    valid=c.valid,
                    raw=c.raw,
                )
            )

    # Fallback answer if LLM generation was not successful (e.g. offline dev without key)
    if not answer_text and not gen_result.success:
        answer_text = (
            f"Unable to generate live LLM answer ({gen_result.error_kind or 'unavailable'}). "
            f"Retrieved {len(retrieved_chunks)} context chunks for analysis."
        )

    # 6. Log query in database
    try:
        user_id = await _get_or_create_default_user(db)
        query_log = QueryLog(
            user_id=user_id,
            repository_id=repo_db_id,
            query_text=payload.question,
            response_text=answer_text,
        )
        db.add(query_log)
        await db.commit()
    except Exception as log_err:  # noqa: BLE001
        logger.warning("Failed to record query log in DB: %s", log_err)

    metadata = QueryMetadata(
        model=gen_result.model or generator.model,
        provider=gen_result.provider or generator.provider,
        latency_seconds=round(elapsed, 4),
        coverage=coverage,
        valid_citations=valid_count,
        invalid_citations=invalid_count,
        chunks_retrieved=len(retrieved_chunks),
        error=gen_result.error,
    )

    return QueryResponse(
        answer=answer_text,
        citations=citations_list,
        metadata=metadata,
    )

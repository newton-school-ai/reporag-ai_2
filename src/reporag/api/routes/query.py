"""Query endpoint.

POST /api/v1/query - Accept a natural language question about a repository,
run the full RAG pipeline, and return a cited answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from reporag.agent.executor import SubQueryExecutor
from reporag.agent.planner import QueryDecomposer
from reporag.agent.router import StrategyRouter
from reporag.generation.citation import CitationReport
from reporag.generation.context_assembler import ContextAssembler
from reporag.generation.generator import (
    AnsweredQuery,
    AnswerGenerator,
    GenerationResult,
)
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.bm25_search import BM25Search
from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.graph_traversal import (
    AmbiguousSymbolError,
    GraphRetriever,
    SymbolNotFoundError,
)
from reporag.retrieval.vector_search import RetrievalResult, VectorSearch

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for ``POST /api/v1/query``."""

    query: str = Field(
        ..., min_length=1, description="Natural-language question about the repo."
    )
    repository_id: int | None = Field(
        default=None,
        description=(
            "Repository to scope the query to. Reserved for future per-repo "
            "index filtering -- not yet enforced, since retrieval currently "
            "runs against a single shared index."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to retrieve per sub-query.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be a non-empty string.")
        return value.strip()


class CitationOut(BaseModel):
    """One validated citation extracted from the generated answer."""

    file_path: str
    start_line: int | None
    end_line: int | None
    valid: bool | None
    snippet: str = ""


class QueryMetadata(BaseModel):
    """Everything about *how* the answer was produced, for observability."""

    query_type: Literal["simple-lookup", "multi-hop", "exploratory"] | None = None
    needs_decomposition: bool = False
    num_sub_queries: int = 0
    strategies_used: list[str] = Field(default_factory=list)
    citation_coverage: float = 0.0
    model: str = ""
    provider: str = ""
    latency_seconds: float = 0.0
    success: bool = False
    error: str | None = None


class QueryResponse(BaseModel):
    """Response for ``POST /api/v1/query``: ``{answer, citations, metadata}``."""

    answer: str
    citations: list[CitationOut]
    metadata: QueryMetadata


# ---------------------------------------------------------------------------
# Retrieval engine: adapts BM25/Vector/Graph search to SubQueryExecutor's
# duck-typed RetrievalEngine protocol, degrading a backend to an empty
# result list (instead of failing the whole request) if it's unreachable.
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "how",
    "does",
    "do",
    "did",
    "where",
    "when",
    "why",
    "this",
    "that",
    "and",
    "or",
    "for",
    "with",
    "from",
    "into",
    "about",
    "find",
    "show",
    "list",
    "get",
    "all",
    "of",
    "in",
    "on",
    "to",
}


def _looks_like_identifier(token: str) -> bool:
    return (
        "_" in token
        or "." in token
        or (token != token.lower() and token != token.upper())
    )


def _extract_candidate_symbol(query: str) -> str | None:
    """Best-effort extraction of a symbol name to seed a graph lookup."""
    tokens = [t for t in _IDENTIFIER_RE.findall(query) if t.lower() not in _STOPWORDS]
    if not tokens:
        return None
    identifier_like = [t for t in tokens if _looks_like_identifier(t)]
    candidates = identifier_like or tokens
    return max(candidates, key=len)


class LiveRetrievalEngine:
    """Real BM25 + vector + graph retrieval, wired for :class:`SubQueryExecutor`.

    Backends are constructed lazily so instantiating an engine has no side
    effects until a search is actually performed.
    """

    def __init__(
        self,
        *,
        vector: VectorSearch | None = None,
        bm25: BM25Search | None = None,
        graph: GraphRetriever | None = None,
    ) -> None:
        self._vector = vector
        self._bm25 = bm25
        self._graph = graph

    @property
    def vector(self) -> VectorSearch:
        if self._vector is None:
            self._vector = VectorSearch()
        return self._vector

    @property
    def bm25(self) -> BM25Search:
        if self._bm25 is None:
            self._bm25 = BM25Search()
        return self._bm25

    @property
    def graph(self) -> GraphRetriever:
        if self._graph is None:
            self._graph = GraphRetriever()
        return self._graph

    def search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            return self.vector.search(query, top_k=top_k)
        except Exception:
            logger.warning(
                "Vector search unavailable; returning no results.", exc_info=True
            )
            return []

    def search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            return self.bm25.search(query, top_k=top_k)
        except Exception:
            logger.warning(
                "BM25 search unavailable; returning no results.", exc_info=True
            )
            return []

    def search_graph(self, query: str, top_k: int) -> list[RetrievalResult]:
        candidate = _extract_candidate_symbol(query)
        if not candidate:
            return []
        try:
            results = self.graph.get_neighbors(candidate, depth=2)
        except (SymbolNotFoundError, AmbiguousSymbolError):
            return []
        except Exception:
            logger.warning(
                "Graph search unavailable; returning no results.", exc_info=True
            )
            return []
        return results[:top_k]

    def search_hybrid(self, query: str, top_k: int) -> list[RetrievalResult]:
        ranked_lists = [
            results
            for results in (
                self.search_vector(query, top_k),
                self.search_bm25(query, top_k),
                self.search_graph(query, top_k),
            )
            if results
        ]
        if not ranked_lists:
            return []
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


# ---------------------------------------------------------------------------
# Pipeline orchestration: decompose -> retrieve -> assemble -> prompt ->
# generate + cite. Synchronous (blocking LLM/embedding calls); the route
# runs it via asyncio.to_thread.
# ---------------------------------------------------------------------------


def _merge_step_results(step_results: dict, top_k: int) -> list[RetrievalResult]:
    """Flatten per-step retrieval results into one ranked list via RRF."""
    ranked_lists = [sr.results for sr in step_results.values() if sr.results]
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]
    return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


def _run_query_pipeline(query: str, top_k: int = 10) -> dict:
    """Run the full RAG pipeline for *query* and return everything the route needs."""
    start = time.monotonic()

    engine = LiveRetrievalEngine()
    decomposer = QueryDecomposer()
    assembler = ContextAssembler()
    prompt_builder = PromptBuilder()
    generator = AnswerGenerator()

    plan = decomposer.decompose(query)

    executor = SubQueryExecutor(
        engine, router=StrategyRouter(use_llm=False), top_k=top_k
    )
    step_results = executor.execute(plan.steps)

    merged_results = _merge_step_results(step_results, top_k=top_k)
    context = assembler.assemble(merged_results)

    built_prompt = prompt_builder.build_prompt(
        query, query_type=plan.classification, context=context
    )
    try:
        answered = generator.generate_with_citations(built_prompt)
    except Exception as exc:
        # generate() only guards call-time failures (network, timeout, rate
        # limit) -- a construction-time one (no API key configured) still
        # raises. Treat it the same way: empty answer, error in metadata,
        # not a crashed request.
        logger.warning("Answer generation could not start: %s", exc)
        answered = AnsweredQuery(
            answer="",
            citations=CitationReport(),
            generation=GenerationResult(
                text="",
                success=False,
                error=str(exc),
                model=generator.model,
                provider=generator.provider,
            ),
        )

    strategies_used = sorted(
        {sr.strategy for sr in step_results.values() if not sr.skipped}
    )

    return {
        "answered": answered,
        "query_type": plan.classification.query_type,
        "needs_decomposition": plan.needs_decomposition,
        "num_sub_queries": len(plan.steps),
        "strategies_used": strategies_used,
        "latency_seconds": time.monotonic() - start,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the agentic RAG pipeline for ``request.query`` and return a cited answer.

    Classifies and (if needed) decomposes the query, retrieves supporting
    code via BM25/vector/graph search, assembles it into context, and asks
    the configured LLM to answer with inline citations, which are then
    validated against the retrieved context.
    """
    try:
        result = await asyncio.to_thread(
            _run_query_pipeline, request.query, request.top_k
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive: pipeline failures are unexpected
        logger.exception("Query pipeline failed for query=%r", request.query)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Query pipeline failed: {exc}",
        ) from exc

    answered: AnsweredQuery = result["answered"]
    citations = [
        CitationOut(
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            valid=c.valid,
            snippet=c.snippet,
        )
        for c in answered.citations.citations
    ]

    metadata = QueryMetadata(
        query_type=result["query_type"],
        needs_decomposition=result["needs_decomposition"],
        num_sub_queries=result["num_sub_queries"],
        strategies_used=result["strategies_used"],
        citation_coverage=answered.citations.coverage,
        model=answered.generation.model,
        provider=answered.generation.provider,
        latency_seconds=result["latency_seconds"],
        success=answered.success,
        error=answered.generation.error,
    )

    return QueryResponse(answer=answered.answer, citations=citations, metadata=metadata)

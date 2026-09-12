"""Query endpoints.

POST /api/v1/query - Accept a natural language question about a repository,
run the full RAG pipeline, and return a cited answer.

Why
---
This is where every prior milestone meets. The planner (Issues 20-21)
classifies and decomposes the question, the router and executor (Issue 22)
retrieve for each sub-query, the assembler and prompt builder (Issues 23-24)
turn results into a prompt, and the generator (Issue 25) produces an answer
whose citations are validated against the context that was actually sent.

Design
------
* **Built once, reused.** The decomposer compiles a LangGraph state machine
  and the retrieval backends hold model and network handles, so they are
  constructed lazily on first request and cached on the app state rather
  than rebuilt per call.
* **Blocking work goes to the threadpool.** Every pipeline component is
  synchronous; calling one from an ``async def`` handler would stall the
  event loop for the whole LLM round trip. The route is thin and awaits
  :func:`~fastapi.concurrency.run_in_threadpool`.
* **LLM failures map to honest status codes.** The generator reports a
  category instead of raising, so a provider rate limit surfaces as 429 and
  a timeout as 504, rather than every failure collapsing into 500.
* **Retrieval degrades, generation does not.** A dead Qdrant yields an
  answer built from fewer sources; a dead LLM has no fallback and is an
  error the caller must see.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.config import settings
from reporag.db.models import Repository
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# Generator failure category -> HTTP status. An LLM rate limit or timeout is
# an upstream condition, not a bug in this service, and 502/504/429 tell the
# caller whether retrying is worthwhile.
_ERROR_STATUS = {
    "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
    "timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "auth": status.HTTP_502_BAD_GATEWAY,
    "invalid_response": status.HTTP_502_BAD_GATEWAY,
    "api_error": status.HTTP_502_BAD_GATEWAY,
}

# Results carried into the prompt. The builder trims to a token budget
# anyway; this bounds the work it has to do on a broad question.
_MAX_CONTEXT_RESULTS = 40


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body of ``POST /api/v1/query``.

    Attributes:
        question: The natural-language question about the repository.
        repo_id: Repository to search. ``None`` searches everything indexed.
        top_k: Results retrieved per sub-query.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural language question about the repository.",
        examples=["How does the authentication flow work end to end?"],
    )
    repo_id: int | None = Field(
        default=None,
        ge=1,
        description="Repository to search. Omit to search everything indexed.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Results retrieved per sub-query. Defaults to RERANK_TOP_K.",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """Reject a question that is only whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class CitationModel(BaseModel):
    """One citation extracted from the answer and checked against the context.

    Attributes:
        file_path: Cited file.
        start_line: First cited line, if the marker carried one.
        end_line: Last cited line, if the marker carried one.
        snippet: The cited source text, when it was found in the context.
        valid: Whether the cited range was present in the context sent to
            the model. ``False`` marks a citation the model invented.
    """

    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    snippet: str = ""
    valid: bool | None = None


class QueryMetadata(BaseModel):
    """How the answer was produced.

    Attributes:
        query_type: How the planner classified the question.
        sub_queries: Sub-queries the question was decomposed into.
        strategies: Retrieval strategy used per sub-query step.
        sources: Distinct files that contributed context.
        result_count: Context chunks sent to the model.
        citation_coverage: Fraction of answer claims carrying a citation.
        valid_citations: Citations resolved against the context.
        invalid_citations: Citations that did not resolve.
        model: Model that produced the answer.
        latency_seconds: Wall-clock duration of the request.
        prompt_tokens: Size of the assembled prompt.
        truncated: Whether context had to be dropped to fit the budget.
    """

    query_type: str
    sub_queries: list[str] = Field(default_factory=list)
    strategies: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    result_count: int = 0
    citation_coverage: float = 0.0
    valid_citations: int = 0
    invalid_citations: int = 0
    model: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    truncated: bool = False


class QueryResponse(BaseModel):
    """Body of a successful ``POST /api/v1/query``.

    Attributes:
        answer: The generated answer, with inline ``[file:start-end]`` markers.
        citations: Every citation found in the answer, each flagged valid or not.
        metadata: How the answer was produced.
    """

    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    metadata: QueryMetadata


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


class _RetrievalEngineAdapter:
    """Adapt the retrieval layer to the executor's ``RetrievalEngine`` protocol.

    :class:`~reporag.agent.executor.SubQueryExecutor` expects four
    ``search_*`` methods over free text. The retrieval primitives do not
    offer that shape directly: vector and BM25 search take a query but graph
    traversal addresses nodes by symbol name, and nothing fuses the three.
    This bridges both gaps.

    Every backend is wrapped so that a dead Qdrant or Neo4j contributes an
    empty list instead of failing the request -- an answer from two backends
    is far more useful than an exception.
    """

    def __init__(
        self,
        *,
        vector_search: Any | None = None,
        bm25_search: Any | None = None,
        graph_retriever: Any | None = None,
    ) -> None:
        """Store the backends, building real ones lazily on first use."""
        self._vector = vector_search
        self._bm25 = bm25_search
        self._graph = graph_retriever

    # -- Lazy backends ------------------------------------------------------

    @property
    def vector(self) -> Any:
        """The vector searcher, constructed on first access."""
        if self._vector is None:
            from reporag.retrieval.vector_search import VectorSearch

            self._vector = VectorSearch()
        return self._vector

    @property
    def bm25(self) -> Any:
        """The BM25 searcher, constructed on first access."""
        if self._bm25 is None:
            from reporag.retrieval.bm25_search import BM25Search

            self._bm25 = BM25Search()
        return self._bm25

    @property
    def graph(self) -> Any:
        """The graph retriever, constructed on first access."""
        if self._graph is None:
            from reporag.retrieval.graph_traversal import GraphRetriever

            self._graph = GraphRetriever(neo4j_uri=settings.neo4j_uri, fallback=True)
        return self._graph

    # -- Protocol -----------------------------------------------------------

    def search_bm25(self, query: str, top_k: int) -> list[Any]:
        """Sparse keyword search over code identifiers."""
        return self._safely("bm25", lambda: self.bm25.search(query, top_k=top_k))

    def search_vector(self, query: str, top_k: int) -> list[Any]:
        """Dense semantic search over code and docstring embeddings."""
        return self._safely("vector", lambda: self.vector.search(query, top_k=top_k))

    def search_graph(self, query: str, top_k: int) -> list[Any]:
        """Structural search over the call and dependency graph.

        The graph is keyed by symbol, so identifiers are recovered from the
        query text first. A question naming no symbol retrieves nothing,
        which is the honest answer -- there is no node to traverse from.
        """
        if not settings.enable_graph_retrieval:
            return []

        symbols = _extract_symbols(query)
        if not symbols:
            return []

        def traverse() -> list[Any]:
            collected: list[Any] = []
            seen: set[tuple[str, int | None, int | None]] = set()
            for symbol in symbols:
                try:
                    neighbours = self.graph.get_neighbors(symbol, depth=2)
                except Exception as exc:
                    # A query may name something that is not in this repo;
                    # that is expected, not exceptional.
                    logger.debug("Graph traversal failed for %r: %s", symbol, exc)
                    continue
                for result in neighbours:
                    key = (result.file_path, result.start_line, result.end_line)
                    if key not in seen:
                        seen.add(key)
                        collected.append(result)
            collected.sort(key=lambda r: r.score, reverse=True)
            return collected[:top_k]

        return self._safely("graph", traverse)

    def search_hybrid(self, query: str, top_k: int) -> list[Any]:
        """Run every strategy and fuse the rankings with RRF.

        Reciprocal Rank Fusion is used rather than score averaging because
        the three backends score on incomparable scales; RRF needs only the
        ranks. The fused head is then reranked by the cross-encoder when
        ``ENABLE_RERANKER`` is set.
        """
        from reporag.retrieval.fusion import reciprocal_rank_fusion

        # Over-fetch per backend so fusion has depth to work with: the best
        # final result may sit at rank 15 in one list and rank 2 in another.
        candidate_k = max(top_k, settings.vector_search_top_k)
        ranked_lists = [
            self.search_bm25(query, candidate_k),
            self.search_vector(query, candidate_k),
            self.search_graph(query, candidate_k),
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_constant)

        if not fused or not settings.enable_reranker:
            return fused[:top_k]

        try:
            from reporag.retrieval.reranker import CrossEncoderReranker

            window = fused[: max(top_k, settings.rerank_top_k)]
            return CrossEncoderReranker().rerank(query, window, top_k=top_k)
        except Exception:
            logger.warning(
                "Reranking failed; falling back to fusion order", exc_info=True
            )
            return fused[:top_k]

    @staticmethod
    def _safely(name: str, call: Any) -> list[Any]:
        """Invoke *call*, absorbing backend failure into an empty result."""
        try:
            return list(call())
        except Exception:
            logger.warning("Retrieval strategy %s failed", name, exc_info=True)
            return []


# Identifiers, dotted or bare: ``authenticate_user``, ``Auth.verify``.
_IDENTIFIER_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"

# An identifier is "code-shaped" if it carries a naming convention prose
# does not: a dot, an underscore, or an internal capital. Without this,
# every ordinary English word in the question would be probed as a symbol.
_CODE_SHAPED_PATTERN = r"[._]|[a-z][A-Z]"


def _extract_symbols(query: str, *, limit: int = 5) -> list[str]:
    """Recover likely symbol names from natural-language *query*.

    Args:
        query: The question text.
        limit: Maximum number of candidates to return.

    Returns:
        Code-shaped identifiers in order of appearance, deduplicated. Empty
        when the question names no plausible symbol.
    """
    import re

    seen: set[str] = set()
    found: list[str] = []
    for match in re.finditer(_IDENTIFIER_PATTERN, query):
        token = match.group(0)
        if token in seen or not re.search(_CODE_SHAPED_PATTERN, token):
            continue
        seen.add(token)
        found.append(token)
    return found[:limit]


def get_pipeline(request: Request) -> dict[str, Any]:
    """Return the cached pipeline components, building them on first use.

    The decomposer compiles a LangGraph state machine and the retrieval
    backends hold model and socket handles, so rebuilding them per request
    would dominate latency. They are cached on ``app.state``.

    Args:
        request: The active request, used to reach ``app.state``.

    Returns:
        A mapping with ``engine``, ``decomposer``, ``executor``,
        ``prompt_builder`` and ``generator`` keys.

    Raises:
        HTTPException: 503 when a component cannot be constructed at all --
            most often an unconfigured LLM API key.
    """
    cached = getattr(request.app.state, "pipeline", None)
    if cached is not None:
        return cached

    from reporag.agent.executor import SubQueryExecutor
    from reporag.agent.planner import QueryDecomposer
    from reporag.generation.generator import AnswerGenerator
    from reporag.generation.prompt_builder import PromptBuilder

    try:
        engine = _RetrievalEngineAdapter()
        pipeline: dict[str, Any] = {
            "engine": engine,
            "decomposer": QueryDecomposer(),
            "executor": SubQueryExecutor(engine, top_k=settings.rerank_top_k),
            "prompt_builder": PromptBuilder(),
            "generator": AnswerGenerator(),
        }
    except Exception as exc:
        # Construction failing is a deployment problem, not a bad request:
        # an unset ANTHROPIC_API_KEY raises here, before any call is made.
        # 500 would read as a bug in this service; 503 says the deployment
        # is incomplete and retrying now will not help. The reason goes to
        # the log rather than the response -- the caller is pointed at the
        # health endpoint, which reports the same component without
        # exposing configuration detail to an unauthenticated request.
        logger.exception("Query pipeline could not be constructed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The query pipeline is not available. "
                "See GET /api/v1/health for component status."
            ),
        ) from exc

    request.app.state.pipeline = pipeline
    return pipeline


def _run_pipeline(
    components: dict[str, Any], question: str, top_k: int
) -> tuple[Any, Any, dict[str, Any]]:
    """Run plan, retrieve, prompt and generate. Blocking.

    Args:
        components: The cached pipeline components.
        question: The user's question.
        top_k: Results per sub-query.

    Returns:
        A tuple of the built prompt, the generated answer, and a mapping of
        pipeline facts for the response metadata.
    """
    from reporag.agent.executor import SubQueryExecutor
    from reporag.generation.prompt_builder import SubQueryAnswer

    plan = components["decomposer"].decompose(question)

    executor = components["executor"]
    if top_k != settings.rerank_top_k:
        # Honour a per-request top_k without mutating the shared executor.
        executor = SubQueryExecutor(components["engine"], top_k=top_k)

    try:
        step_results = executor.execute(plan.steps)
    except Exception:
        logger.warning("Sub-query execution failed", exc_info=True)
        step_results = {}

    # Steps overlap by design -- two sub-queries about one subsystem surface
    # the same functions -- so keep each chunk once, at its best rank across
    # steps rather than its first.
    best: dict[tuple[str, int | None, int | None], Any] = {}
    for step in step_results.values():
        for result in getattr(step, "results", []):
            key = (result.file_path, result.start_line, result.end_line)
            incumbent = best.get(key)
            if incumbent is None or result.score > incumbent.score:
                best[key] = result
    results = sorted(best.values(), key=lambda r: r.score, reverse=True)
    results = results[:_MAX_CONTEXT_RESULTS]

    # Feed each step's findings forward so the final prompt can see what the
    # intermediate hops discovered. Empty summaries are dropped: they would
    # spend context saying nothing.
    sub_answers = [
        SubQueryAnswer(
            step_id=step_id,
            answer=step.context_summary.strip(),
            query=next((s.query for s in plan.steps if s.id == step_id), ""),
        )
        for step_id, step in step_results.items()
        if getattr(step, "context_summary", "").strip()
    ]

    prompt = components["prompt_builder"].build_from_results(
        question, plan.classification, results, sub_answers
    )
    generated = components["generator"].generate_with_citations(prompt)

    facts = {
        "query_type": plan.classification.query_type,
        "sub_queries": [step.query for step in plan.steps],
        "strategies": {
            step_id: step.strategy for step_id, step in step_results.items()
        },
        "sources": sorted({r.file_path for r in results}),
        "result_count": len(results),
    }
    return prompt, generated, facts


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about a repository",
    response_description="A cited answer with retrieval and generation metadata.",
    responses={
        404: {"description": "No repository with that id."},
        422: {"description": "The question could not be planned or assembled."},
        429: {"description": "The LLM provider rate limited the request."},
        502: {"description": "The pipeline or the LLM provider failed."},
        503: {"description": "A pipeline component is not configured."},
        504: {"description": "The LLM provider timed out."},
    },
)
async def query(
    payload: QueryRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> QueryResponse:
    """Answer a natural-language question with citations into the source.

    Runs the full pipeline: classify and decompose the question, retrieve
    for each sub-query, assemble a prompt, generate, and validate every
    citation against the context that was actually sent to the model.

    Raises:
        HTTPException: 404 if ``repo_id`` names an unknown repository, 422
            if the question cannot be planned, 503 if a component is not
            configured, or 429/502/504 when the LLM provider fails.
    """
    if payload.repo_id is not None:
        repository = await session.get(Repository, payload.repo_id)
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository {payload.repo_id} not found",
            )

    components = get_pipeline(request)
    top_k = payload.top_k or settings.rerank_top_k

    started = time.monotonic()
    # Every pipeline component is synchronous and the LLM call dominates the
    # request, so this must not run on the event loop.
    try:
        prompt, generated, facts = await run_in_threadpool(
            _run_pipeline, components, payload.question, top_k
        )
    except ValueError as exc:
        # The planner and prompt builder raise ValueError for input they
        # cannot work with -- a question that survives field validation but
        # still cannot be planned. That is the caller's to fix, so 422
        # rather than the 500 an unhandled exception would produce.
        logger.info("Query rejected by the pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        # Retrieval already degrades to empty results internally, so getting
        # here means a component broke in a way it does not handle. 502 says
        # the failure was downstream of this service and a retry may work,
        # which the generic 500 handler cannot convey. The exception text
        # stays in the log: it routinely carries file paths and provider
        # payloads that an unauthenticated caller should not see.
        logger.exception("Query pipeline failed for question=%r", payload.question)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The query pipeline failed to produce an answer.",
        ) from exc
    elapsed = time.monotonic() - started

    if not generated.success:
        detail = generated.generation.error or "Answer generation failed"
        raise HTTPException(
            status_code=_ERROR_STATUS.get(
                generated.generation.error_kind, status.HTTP_502_BAD_GATEWAY
            ),
            detail=detail,
        )

    report = generated.citations
    return QueryResponse(
        answer=generated.answer,
        citations=[
            CitationModel(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                snippet=c.snippet,
                valid=c.valid,
            )
            for c in report.citations
        ],
        metadata=QueryMetadata(
            query_type=facts["query_type"],
            sub_queries=facts["sub_queries"],
            strategies=facts["strategies"],
            sources=facts["sources"],
            result_count=facts["result_count"],
            citation_coverage=report.coverage,
            valid_citations=report.valid_count,
            invalid_citations=report.invalid_count,
            model=generated.generation.model,
            latency_seconds=round(elapsed, 3),
            prompt_tokens=prompt.token_count,
            truncated=prompt.truncated,
        ),
    )

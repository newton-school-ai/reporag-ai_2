"""Sub-query executor.

Executes sub-queries in dependency order, passing context from earlier
steps to later ones. Orchestrates the retrieval engine based on the
strategy assigned by the router.

Why
---
A :class:`~reporag.agent.planner.DecompositionPlan` produced by Issue 21
contains an ordered set of sub-queries with dependency edges
(``step.depends_on``). The executor is the component that turns that plan
into actual retrieved results:

1. **Dependency order** -- steps that a later step depends on must run
   first. Kahn's topological sort enforces this regardless of the order
   the steps appear in the plan list.
2. **Context forwarding** -- a later step that depends on an earlier one
   benefits from knowing what the earlier step retrieved. The executor
   injects a plain-text summary of the top results into the sub-query
   text before sending it to the retrieval engine, so the retriever sees
   richer context without any schema change.
3. **Strategy dispatch** -- each step's query is sent to the retrieval
   backend chosen by :class:`~reporag.agent.router.StrategyRouter`. The
   ``hybrid`` strategy runs BM25, vector, and graph in parallel and fuses
   the results via RRF.
4. **Fail-safe** -- a single step failure (retriever error) retries once,
   then marks the step ``skipped`` and continues. Dependent steps receive
   empty context and still run -- the plan never fully aborts.

Design
------
:class:`SubQueryExecutor` is deliberately thin:

* It depends on a :class:`RetrievalEngine` *protocol* (a structural
  subtype -- no import, just duck-typing) so tests can inject a
  ``_FakeRetrievalEngine`` without any real Qdrant / Neo4j connections,
  mirroring the ``_FakeLLM`` seam in the planner tests.
* The topological sort is self-contained pure logic (no external library)
  and raises :class:`CyclicDependencyError` on a cycle rather than
  silently deadlocking.
* Context summaries are plain text (top-3 ``chunk_text`` snippets,
  truncated to 500 chars) -- intentionally simple so this module stays
  reviewable without understanding the embedding or generation layers.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reporag.agent.router import StrategyRouter
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RetrievalEngine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalEngine(Protocol):
    """Duck-typed interface the executor depends on.

    This is the test seam: tests inject a ``_FakeRetrievalEngine`` that
    returns canned results without any real Qdrant / Neo4j connections.
    Real callers pass an object that implements the four ``search_*``
    methods -- it does not need to inherit from this class.

    Each method receives a plain-text *query* string and a *top_k* limit
    and returns a list of :class:`~reporag.retrieval.vector_search.RetrievalResult`
    objects sorted by score descending, exactly as the individual
    retrievers (:class:`~reporag.retrieval.bm25_search.BM25Search`,
    :class:`~reporag.retrieval.vector_search.VectorSearch`,
    :class:`~reporag.retrieval.graph_traversal.GraphRetriever`) do.
    """

    def search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25 sparse keyword search."""
        ...

    def search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Dense vector semantic search."""
        ...

    def search_graph(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Graph-based structural traversal."""
        ...

    def search_hybrid(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Run all strategies and fuse results via RRF."""
        ...


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# Maximum number of characters kept in a context summary (forwarded to
# dependent steps).  Long enough to be useful; short enough not to blow up
# the retrieval query length.
_CONTEXT_SUMMARY_MAX_CHARS = 500

# Number of top-result snippets to include in the context summary.
_CONTEXT_SUMMARY_TOP_N = 3

# Separator between snippets in the context summary.
_CONTEXT_SNIPPET_SEP = "\n---\n"


class CyclicDependencyError(ValueError):
    """Raised when the step dependency graph contains a cycle.

    This is a ``ValueError`` subtype so callers that only catch
    ``ValueError`` (e.g. the planner's validation layer) handle it
    naturally, and callers that want to distinguish the cycle case can
    catch ``CyclicDependencyError`` specifically.
    """


@dataclass
class StepResult:
    """The outcome of executing a single :class:`~reporag.agent.planner.DecompositionStep`.

    Attributes:
        step_id: The id of the step (mirrors
            :attr:`~reporag.agent.planner.DecompositionStep.id`).
        strategy: The retrieval strategy that was used, as returned by
            :class:`~reporag.agent.router.StrategyRouter`.
        results: The retrieved results, sorted by score descending.
            Empty when the step was skipped.
        context_summary: A plain-text summary of the top
            :data:`_CONTEXT_SUMMARY_TOP_N` result snippets (truncated to
            :data:`_CONTEXT_SUMMARY_MAX_CHARS` characters).  Injected into
            the sub-queries of dependent steps as additional context.
        error: A human-readable description of the error if the step
            failed after the retry.  ``None`` on success.
        skipped: ``True`` when the step failed after the retry and was
            marked skipped so execution could continue.
        augmented_query: The full query text that was actually sent to the
            retrieval engine (original query + forwarded context from prior
            steps). Stored for debugging / observability.
    """

    step_id: str
    strategy: str = "hybrid"
    results: list[RetrievalResult] = field(default_factory=list)
    context_summary: str = ""
    error: str | None = None
    skipped: bool = False
    augmented_query: str = ""


# ---------------------------------------------------------------------------
# Internal helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


def _topological_sort(
    step_ids: list[str],
    depends_on: dict[str, tuple[str, ...]],
) -> list[str]:
    """Return *step_ids* in a valid topological order (dependencies first).

    Uses Kahn's algorithm: iteratively remove nodes with zero in-degree.

    Args:
        step_ids: All step ids in the plan (in declaration order).
        depends_on: Mapping from ``step_id`` to the ids it depends on.

    Returns:
        The step ids in an order that respects all dependency edges.

    Raises:
        CyclicDependencyError: If the dependency graph contains a cycle.
    """
    # Build adjacency and in-degree maps.
    in_degree: dict[str, int] = {sid: 0 for sid in step_ids}
    # dependents[A] = list of steps that depend on A (A must run before them).
    dependents: dict[str, list[str]] = {sid: [] for sid in step_ids}

    for sid in step_ids:
        for dep in depends_on.get(sid, ()):
            if dep in in_degree:
                in_degree[sid] += 1
                dependents[dep].append(sid)

    queue: deque[str] = deque(sid for sid in step_ids if in_degree[sid] == 0)
    order: list[str] = []

    while queue:
        sid = queue.popleft()
        order.append(sid)
        for dependent in dependents[sid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(step_ids):
        cycle_nodes = [sid for sid in step_ids if sid not in order]
        raise CyclicDependencyError(
            f"Cyclic dependency detected among steps: {cycle_nodes}. "
            "All depends_on edges must be backward-only (no forward "
            "references and no cycles)."
        )

    return order


def _build_context_summary(results: list[RetrievalResult]) -> str:
    """Build a plain-text context summary from the top retrieved results.

    Takes the top :data:`_CONTEXT_SUMMARY_TOP_N` results, joins their
    ``chunk_text`` with :data:`_CONTEXT_SNIPPET_SEP`, and truncates to
    :data:`_CONTEXT_SUMMARY_MAX_CHARS` characters.

    Returns an empty string when *results* is empty.
    """
    if not results:
        return ""
    snippets = [
        r.chunk_text.strip()
        for r in results[:_CONTEXT_SUMMARY_TOP_N]
        if r.chunk_text.strip()
    ]
    if not snippets:
        return ""
    summary = _CONTEXT_SNIPPET_SEP.join(snippets)
    return summary[:_CONTEXT_SUMMARY_MAX_CHARS]


def _augment_query(query: str, context_summaries: list[str]) -> str:
    """Append non-empty *context_summaries* to *query*.

    The augmented query is what the retrieval engine actually sees. The
    context block is clearly demarcated so the engine / embedding model
    can treat it as supplementary information rather than part of the
    primary query intent.

    Args:
        query: The original sub-query text.
        context_summaries: Context strings from prior steps. Empty strings
            are silently dropped.

    Returns:
        The augmented query string.  If no non-empty summaries are
        provided the original *query* is returned unchanged.
    """
    non_empty = [s for s in context_summaries if s and s.strip()]
    if not non_empty:
        return query
    context_block = "\n\n".join(non_empty)
    return f"{query}\n\n[Context from prior steps:\n{context_block}\n]"


# ---------------------------------------------------------------------------
# SubQueryExecutor
# ---------------------------------------------------------------------------


class SubQueryExecutor:
    """Executes a decomposition plan's steps in dependency order.

    Each step is routed to the optimal retrieval strategy by a
    :class:`~reporag.agent.router.StrategyRouter`, executed against the
    supplied :class:`RetrievalEngine`, and its results are summarised and
    forwarded as context to dependent steps.

    Args:
        engine: A :class:`RetrievalEngine`-compatible object that provides
            the four ``search_*`` methods. Tests inject a
            ``_FakeRetrievalEngine`` here (no real Qdrant / Neo4j needed).
        router: A pre-built :class:`~reporag.agent.router.StrategyRouter`.
            When ``None`` (default) a new one is constructed with
            ``use_llm=False`` (offline-safe default) so the executor works
            without any API key -- callers that want LLM-assisted routing
            should pass their own router explicitly.
        top_k: Maximum number of results to retrieve per step. Defaults
            to ``10``.

    Raises:
        TypeError: If *engine* does not implement the
            :class:`RetrievalEngine` protocol.
    """

    def __init__(
        self,
        engine: Any,
        router: StrategyRouter | None = None,
        *,
        top_k: int = 10,
    ) -> None:
        if not isinstance(engine, RetrievalEngine):
            raise TypeError(
                f"engine must implement the RetrievalEngine protocol "
                f"(search_bm25, search_vector, search_graph, search_hybrid), "
                f"got {type(engine).__name__!r}."
            )
        self._engine = engine
        self._router = router or StrategyRouter(use_llm=False)
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        steps: Sequence[Any],  # Sequence[DecompositionStep] -- avoid circular import
    ) -> dict[str, StepResult]:
        """Execute *steps* in dependency order and return per-step results.

        Args:
            steps: A sequence of
                :class:`~reporag.agent.planner.DecompositionStep` objects
                (or any objects with ``id``, ``query``, and ``depends_on``
                attributes). Typically ``plan.steps`` from a
                :class:`~reporag.agent.planner.DecompositionPlan`.

        Returns:
            A ``dict`` mapping each ``step.id`` to a :class:`StepResult`.
            Steps that were skipped after a failure have
            ``StepResult.skipped=True`` and empty ``results``.

        Raises:
            CyclicDependencyError: If the dependency graph contains a
                cycle (forwarded from :func:`_topological_sort`).
        """
        if not steps:
            return {}

        step_by_id = {step.id: step for step in steps}
        step_ids = [step.id for step in steps]
        depends_on_map = {step.id: step.depends_on for step in steps}

        # Determine execution order -- raises CyclicDependencyError on a cycle.
        ordered_ids = _topological_sort(step_ids, depends_on_map)

        completed: dict[str, StepResult] = {}

        for step_id in ordered_ids:
            step = step_by_id[step_id]
            step_result = self._execute_step(step, completed)
            completed[step_id] = step_result

        return completed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: Any,
        completed: dict[str, StepResult],
    ) -> StepResult:
        """Execute a single step, forwarding context from its dependencies.

        Retries once on engine failure. Marks the step ``skipped`` if the
        retry also fails, so execution can continue.
        """
        # Collect context summaries from already-completed dependencies.
        context_summaries = [
            completed[dep_id].context_summary
            for dep_id in step.depends_on
            if dep_id in completed and completed[dep_id].context_summary
        ]
        augmented_query = _augment_query(step.query, context_summaries)

        # Route the (un-augmented) query to pick the optimal strategy.
        routing = self._router.route(step.query, step_id=step.id)
        strategy = routing.strategy

        # Execute with one retry on failure.
        results, error = self._call_engine(augmented_query, strategy)
        if error is not None:
            logger.warning(
                "SubQueryExecutor: step %r failed (%s); retrying once.",
                step.id,
                error,
            )
            results, error = self._call_engine(augmented_query, strategy)

        if error is not None:
            logger.warning(
                "SubQueryExecutor: step %r failed after retry (%s); "
                "skipping and continuing.",
                step.id,
                error,
            )
            return StepResult(
                step_id=step.id,
                strategy=strategy,
                results=[],
                context_summary="",
                error=error,
                skipped=True,
                augmented_query=augmented_query,
            )

        context_summary = _build_context_summary(results)
        return StepResult(
            step_id=step.id,
            strategy=strategy,
            results=results,
            context_summary=context_summary,
            error=None,
            skipped=False,
            augmented_query=augmented_query,
        )

    def _call_engine(
        self,
        query: str,
        strategy: str,
    ) -> tuple[list[RetrievalResult], str | None]:
        """Dispatch *query* to the engine method matching *strategy*.

        Returns a ``(results, error)`` tuple. On success ``error`` is
        ``None``. On failure ``results`` is ``[]`` and ``error`` describes
        what went wrong.
        """
        dispatch = {
            "bm25": self._engine.search_bm25,
            "vector": self._engine.search_vector,
            "graph": self._engine.search_graph,
            "hybrid": self._engine.search_hybrid,
        }
        fn = dispatch.get(strategy, self._engine.search_hybrid)
        try:
            results = fn(query, self.top_k)
            return results, None
        except Exception as exc:  # noqa: BLE001
            return [], str(exc)

    def __repr__(self) -> str:
        return (
            f"SubQueryExecutor(engine={type(self._engine).__name__!r}, "
            f"router={self._router!r}, top_k={self.top_k})"
        )

"""Sub-query executor (Issue 22).

Executes the sub-queries of a
:class:`~reporag.agent.planner.DecompositionPlan` (Issue 21) using the
strategies assigned by :class:`~reporag.agent.router.StrategyRouter`
(Issue 22), in **dependency order**: a step only runs once every step it
``depends_on`` has finished, and the results of those earlier steps are
passed forward as additional query context so later steps can be grounded
in what earlier retrieval actually surfaced.

Design
------
The executor follows the conventions established by the rest of the
agentic-planner stack (``QueryClassifier`` / ``QueryDecomposer`` /
``StrategyRouter``):

* **Backend seam** -- the executor never reaches into Qdrant / Neo4j /
  the BM25 index directly.  Instead it consumes a
  :class:`RetrievalBackend` (a small protocol with one
  ``retrieve(strategy, query, top_k, prior_context)`` method).  A real
  backend (:class:`DefaultRetrievalBackend`) composes the existing
  :class:`~reporag.retrieval.vector_search.VectorSearch`,
  :class:`~reporag.retrieval.bm25_search.BM25Search`, and
  :class:`~reporag.retrieval.graph_traversal.GraphRetriever`, and fuses
  hybrid results with
  :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion`.  Tests inject
  a :class:`FakeRetrievalBackend`, keeping every test network-free --
  exactly the same seam philosophy as the planner's ``_FakeLLM``.
* **Topological scheduling** -- :func:`topological_order` turns the
  plan's ``depends_on`` edges into an execution order in which every
  dependency runs before its dependants.  A cycle (impossible for a valid
  plan, but defensive) raises ``ValueError``.
* **Dependency-forward context** -- when a step runs, the
  ``chunk_text`` of every dependency's results (deduplicated, capped at
  ``context_top_k`` items, joined) is appended to the sub-query text so
  the retrieval backend sees the prior steps' findings.  This is the
  "pass context forward" acceptance criterion.
* **Retry once, then skip** -- a step whose retrieval fails is retried
  *once* (when ``settings.executor_retry_failed_steps`` is True); a
  second failure skips the step, logs a warning, and records the error in
  :class:`StepResult` so downstream steps still run against whatever
  partial context exists.  Execution never aborts the whole plan on a
  single step failure.
* **Uniform output** -- every step yields a :class:`StepResult` whose
  ``results`` is a ``list[RetrievalResult]`` (possibly empty on failure),
  so downstream code can iterate ``plan_results.items()`` uniformly
  regardless of strategy or failure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from reporag.agent.planner import DecompositionPlan, DecompositionStep
from reporag.agent.router import (
    RetrievalStrategy,
    RoutingPlan,
    StrategyRouter,
)
from reporag.config import settings
from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.vector_search import RetrievalResult

if TYPE_CHECKING:
    from reporag.retrieval.bm25_search import BM25Search
    from reporag.retrieval.graph_traversal import GraphRetriever
    from reporag.retrieval.vector_search import VectorSearch

logger = logging.getLogger(__name__)

# Precompiled anchor-identifier patterns for graph retrieval.  Compiled
# once at import (mirrors ``router.py``'s ``_GRAPH_PATTERNS`` convention)
# rather than on every graph step.  See ``_extract_identifiers``.
_BACKTICK_ID_RE = re.compile(r"`([A-Za-z_][\w.]*)`")
_SNAKE_ID_RE = re.compile(r"\b([a-z]+(?:_[a-z0-9]+)+)\b")
_CAMEL_ID_RE = re.compile(r"\b([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)\b")
_PASCAL_ID_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)\b")

# Recognises "trace/find path ... from <X> to <Y> [to <Z> ...]" syntax so a
# graph-routed "trace the path from A to B" query calls ``find_paths`` with
# the two endpoints instead of degrading to a single-anchor neighbour lookup
# (or BM25 when no snake_case anchor is present).
_FROM_TO_RE = re.compile(
    r"\b(?:trace|find|follow)\b.*?\bfrom\s+(.+?)(?:\?|$)", re.IGNORECASE
)
_TO_SPLIT_RE = re.compile(r"\s+to\s+(?:the\s+)?", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """The outcome of executing a single sub-query.

    Attributes:
        step_id: The id of the :class:`~reporag.agent.planner.DecompositionStep`
            that was executed.
        strategy: The :class:`~reporag.agent.router.RetrievalStrategy` used.
        results: The retrieved :class:`~reporag.retrieval.vector_search.RetrievalResult`
            objects (possibly empty on failure / no hits).
        attempted: How many retrieval attempts were made (``2`` if the
            first attempt failed and a retry ran, otherwise ``1``).
        status: ``"ok"`` on success, ``"skipped"`` if the retry budget was
            exhausted, ``"skipped_no_deps"`` if a dependency was skipped and
            this step could not run safely.
        error: ``None`` on success, otherwise a short human-readable
            error string from the failed retrieval attempt(s).
        metadata: Free-form extras (e.g. number of prior-context items
            forwarded, hybrid component breakdown).
    """

    step_id: str
    strategy: RetrievalStrategy
    results: list[RetrievalResult]
    attempted: int = 1
    status: Literal[None, "ok", "skipped", "skipped_no_deps"] = "ok"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    """The full output of executing a whole decomposition plan.

    Attributes:
        original_query: The whole-query text (carried through from the
            :class:`DecompositionPlan` / :class:`RoutingPlan`).
        step_results: ``dict[step_id, StepResult]`` -- one entry per
            sub-query, whether it succeeded or was skipped.  Iteration
            order is the *execution* order, not the plan's declared
            order, so a caller walking ``step_results.items()`` sees
            dependencies before dependants.
        executed_order: The list of step ids in the order they were
            executed (topological order).  Skipped steps appear here so
            the caller can reconstruct the full schedule.
        routed_strategies: ``dict[step_id, RetrievalStrategy]`` echoing
            the strategy each step used -- convenience mirror of
            :class:`RoutingPlan` for callers that only hold the execution
            result.
        metadata: Free-form extras (e.g. retry count, skip count).
    """

    original_query: str
    step_results: dict[str, StepResult]
    executed_order: tuple[str, ...]
    routed_strategies: dict[str, RetrievalStrategy]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RetrievalBackend protocol + default implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalBackend(Protocol):
    """Duck-typed backend that executes one retrieval strategy.

    The executor depends only on this protocol so tests can inject a
    :class:`FakeRetrievalBackend` and never touch Qdrant / Neo4j / the
    BM25 index.  The real implementation is
    :class:`DefaultRetrievalBackend` below.
    """

    def retrieve(
        self,
        strategy: RetrievalStrategy,
        query: str,
        *,
        top_k: int,
        prior_context: str,
        hybrid_components: tuple[RetrievalStrategy, ...],
    ) -> list[RetrievalResult]:
        """Run *strategy* for *query*, returning ranked results.

        Args:
            strategy: One of :attr:`RetrievalStrategy.GRAPH`,
                :attr:`~RetrievalStrategy.VECTOR`,
                :attr:`~RetrievalStrategy.BM25`, or
                :attr:`~RetrievalStrategy.HYBRID`.
            query: The sub-query text (possibly augmented with prior-step
                context by the executor).
            top_k: Maximum number of results to return.
            prior_context: The deduplicated ``chunk_text`` of dependency
                step results, joined by newlines.  Backends may use this
                to refine the query (e.g. extract an anchor symbol);
                backends that cannot exploit it should simply ignore it.
            hybrid_components: The component strategies a ``hybrid``
                route should expand to (mirrors
                :attr:`StrategyRouter.hybrid_components`).

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects sorted by
            the backend's native score descending.  An empty list is a
            valid "no hits" result, not an error.
        """
        ...


class DefaultRetrievalBackend:
    """Retrieval backend that composes the real VectorSearch, BM25Search,
    and GraphRetriever.

    This is the production backend.  It is constructed lazily so
    ``import reporag.agent.executor`` stays cheap and the heavy retrieval
    clients are only built when the executor actually runs.

    * ``vector``  -> :meth:`VectorSearch.search`
    * ``bm25``    -> :meth:`BM25Search.search`
    * ``graph``   -> :meth:`DefaultRetrievalBackend._graph_retrieve`: a
      structural query is treated as "find callers / callees / neighbors
      of the most-salient identifier in the query", falling back to a
      BM25 lookup of the query text to discover anchor symbols when no
      explicit identifier can be extracted.
    * ``hybrid``  -> run every enabled component strategy and fuse with
      :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion`.
    """

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        bm25_search: BM25Search | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        self._vector_search = vector_search
        self._bm25_search = bm25_search
        self._graph_retriever = graph_retriever

    @property
    def vector_search(self) -> VectorSearch:
        if self._vector_search is None:
            from reporag.retrieval.vector_search import VectorSearch

            self._vector_search = VectorSearch()
        return self._vector_search

    @property
    def bm25_search(self) -> BM25Search:
        if self._bm25_search is None:
            from reporag.retrieval.bm25_search import BM25Search

            self._bm25_search = BM25Search()
        return self._bm25_search

    @property
    def graph_retriever(self) -> GraphRetriever:
        if self._graph_retriever is None:
            from reporag.retrieval.graph_traversal import GraphRetriever

            self._graph_retriever = GraphRetriever()
        return self._graph_retriever

    # ------------------------------------------------------------------
    # Identifier extraction for graph queries
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_anchor_identifier(query: str) -> str | None:
        """Pull the most-salient code identifier out of *query*.

        A graph query such as "what calls ``authenticate_user``?" needs
        an anchor symbol to traverse from.  We look for backticked
        identifiers first (the explicit form used in the issue examples),
        then fall back to ``snake_case`` / ``camelCase`` tokens.  Common
        English words are never returned -- they would produce
        nonsensical graph lookups.

        Patterns are precompiled at module level (``_BACKTICK_ID_RE``
        etc.) so the per-call cost is a search, not a compile.
        """
        # Backticked ``identifier`` -- the strongest signal.
        backtick = _BACKTICK_ID_RE.search(query)
        if backtick:
            return backtick.group(1).split(".")[-1]

        # snake_case identifier (>= 2 segments, to avoid plain English
        # words like "calls").
        snake = _SNAKE_ID_RE.search(query)
        if snake:
            return snake.group(1)

        # camelCase identifier.
        camel = _CAMEL_ID_RE.search(query)
        if camel:
            return camel.group(1)

        # PascalCase identifier (class names like DatabaseConfig).
        pascal = _PASCAL_ID_RE.search(query)
        if pascal:
            return pascal.group(1)

        return None

    @staticmethod
    def _extract_identifiers(text: str) -> list[str]:
        """Return every code identifier mentioned in *text*, in order.

        Used by :meth:`_graph_retrieve` to discover the source *and*
        target endpoints of a "trace the path from A to B" query so a
        graph-routed structural query can call ``find_paths`` rather than
        degrade to a single-anchor lookup (or BM25).  Backticked ids win
        first (in order of appearance), then snake_case, then camelCase --
        duplicates are dropped.  Dotted ``module.symbol`` segments are
        flattened to the final component.
        """
        ids: list[str] = []
        seen: set[str] = set()

        def _add(token: str) -> None:
            name = token.split(".")[-1]
            if name and name not in seen:
                seen.add(name)
                ids.append(name)

        for match in _BACKTICK_ID_RE.finditer(text):
            _add(match.group(1))
        for match in _SNAKE_ID_RE.finditer(text):
            _add(match.group(1))
        for match in _CAMEL_ID_RE.finditer(text):
            _add(match.group(1))
        for match in _PASCAL_ID_RE.finditer(text):
            _add(match.group(1))
        return ids

    @staticmethod
    def _extract_path_endpoints(query: str) -> list[str] | None:
        """Split a "trace/find path from A to B [to C ...]" query into named
        endpoints.

        Returns the ordered endpoint strings (lowercased display forms) if
        the query contains an explicit "from ... to ..." chain with at
        least two endpoints, otherwise ``None``.  The endpoints are
        matched against identifier extraction case-insensitively: each
        endpoint segment is scanned for backticked / snake_case /
        camelCase identifiers, and a segment with no extractable
        identifier contributes nothing (so a "trace from the login route
        to the session token" chain whose segments are plain English
        still returns ``None`` via :meth:`_extract_identifiers` rather
        than a garbage anchor).
        """
        match = _FROM_TO_RE.search(query)
        if match is None:
            return None
        remainder = match.group(1).rstrip("?.")
        segments = [
            segment.strip().rstrip("?.") for segment in _TO_SPLIT_RE.split(remainder)
        ]
        segments = [segment for segment in segments if segment]
        if len(segments) < 2:
            return None
        # Each segment must resolve to at least one identifier, otherwise
        # the endpoint is ungrounded (plain English) and a path lookup
        # would fail to resolve a symbol.
        endpoints: list[str] = []
        for segment in segments:
            ids = DefaultRetrievalBackend._extract_identifiers(segment)
            if not ids:
                return None
            endpoints.extend(ids)
        if len(endpoints) < 2:
            return None
        return endpoints

    def _graph_retrieve(
        self, query: str, top_k: int, prior_context: str
    ) -> list[RetrievalResult]:
        """Execute a structural (graph) retrieval.

        Strategy:

        1. If the query is a "trace/find path from A to B" shape, extract
           the endpoints and call :meth:`GraphRetriever.find_paths` -- this
           is the path traversal the structural query actually wants, and
           without it such a query would degrade to a single-anchor
           neighbour lookup (or BM25 when no snake_case anchor is
           present).
        2. Otherwise extract a single anchor identifier from the query
           (or from the prior-step context when the query names none) and
           look up its callers / callees / neighbours.
        3. If no identifier is available at all, fall back to BM25 so the
           step still returns *something* rather than failing outright --
           a graph query with no anchor is rare, but the executor's
           "never fail silently" contract means we degrade rather than
           raise.
        """
        # 1. Explicit path query -> find_paths between endpoints.
        endpoints = self._extract_path_endpoints(query)
        if endpoints is not None and len(endpoints) >= 2:
            try:
                paths = self.graph_retriever.find_paths(endpoints[0], endpoints[-1])
                # Flatten the shortest path first, then any other simple
                # paths, so the most direct traversal surfaces highest.
                flat: list[RetrievalResult] = list(paths.shortest)
                for path in paths.all_paths:
                    for node in path:
                        if node not in flat:
                            flat.append(node)
                return flat[:top_k]
            except Exception as exc:  # noqa: BLE001 - endpoint not in graph
                logger.info(
                    "DefaultRetrievalBackend: find_paths(%r -> %r) failed "
                    "(%s); falling through to anchor-based retrieval.",
                    endpoints[0],
                    endpoints[-1],
                    exc,
                )

        # 2. Single-anchor structural lookup.
        anchor = self._extract_anchor_identifier(query)
        if anchor is None and prior_context:
            # Prior step results may name a symbol the current step is
            # asking about ("what calls the function found in step-1?").
            anchor = self._extract_anchor_identifier(prior_context)

        if anchor is None:
            logger.warning(
                "DefaultRetrievalBackend: no anchor identifier in graph "
                "query %r; falling back to BM25.",
                query,
            )
            return self.bm25_search.search(query, top_k=top_k)

        lowered = query.lower()
        results: list[RetrievalResult] = []
        if "caller" in lowered or "called by" in lowered or "what calls" in lowered:
            results = self.graph_retriever.get_callers(anchor)
        elif "callees" in lowered or "what does" in lowered:
            results = self.graph_retriever.get_neighbors(
                anchor, depth=1, direction="out"
            )
        else:
            results = self.graph_retriever.get_neighbors(anchor, depth=2)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def _run_component(
        self,
        strategy: RetrievalStrategy,
        query: str,
        *,
        top_k: int,
        prior_context: str,
    ) -> list[RetrievalResult]:
        """Run a single *non-hybrid* strategy for *query*.

        This is the leaf dispatch used by :meth:`retrieve` and by the
        hybrid expansion.  It deliberately rejects
        :attr:`RetrievalStrategy.HYBRID` -- hybrid must expand to its
        components at the :meth:`retrieve` layer, so a caller-supplied
        ``hybrid_components`` tuple accidentally containing ``HYBRID``
        cannot trigger unbounded recursion through this helper.
        """
        if strategy is RetrievalStrategy.VECTOR:
            return self.vector_search.search(query, top_k=top_k)
        if strategy is RetrievalStrategy.BM25:
            return self.bm25_search.search(query, top_k=top_k)
        if strategy is RetrievalStrategy.GRAPH:
            return self._graph_retrieve(query, top_k, prior_context)
        # HYBRID is intentionally not handled here (see docstring); an
        # unknown strategy is a programming error, not a runtime fallback.
        raise ValueError(f"_run_component received a non-leaf strategy: {strategy!r}")

    def retrieve(
        self,
        strategy: RetrievalStrategy,
        query: str,
        *,
        top_k: int,
        prior_context: str,
        hybrid_components: tuple[RetrievalStrategy, ...],
    ) -> list[RetrievalResult]:
        """Run *strategy* for *query* using the composed backends.

        ``hybrid`` expands to every enabled component (via
        :meth:`_run_component`, never by recursing into ``retrieve``) and
        fuses the results with Reciprocal Rank Fusion (Issue 19).  A
        :attr:`RetrievalStrategy.HYBRID` entry inside *hybrid_components*
        is dropped defensively so the fusion loop cannot recurse into
        itself.
        """
        if strategy is RetrievalStrategy.HYBRID:
            ranked_lists: list[list[RetrievalResult]] = []
            used_components: list[str] = []
            for component in hybrid_components:
                # Defensive: a stray HYBRID in the components would
                # otherwise recurse into this same branch forever.
                if component is RetrievalStrategy.HYBRID:
                    logger.warning(
                        "DefaultRetrievalBackend: dropping hybrid-from-"
                        "hybrid component from hybrid_components %r.",
                        [c.value for c in hybrid_components],
                    )
                    continue
                try:
                    ranked_lists.append(
                        self._run_component(
                            component,
                            query,
                            top_k=top_k,
                            prior_context=prior_context,
                        )
                    )
                    used_components.append(component.value)
                except Exception as exc:  # noqa: BLE001 - per-component isolation
                    logger.warning(
                        "DefaultRetrievalBackend: hybrid component %s "
                        "failed (%s); continuing with remaining components.",
                        component.value,
                        exc,
                    )
            if not ranked_lists:
                return []
            fused = reciprocal_rank_fusion(
                ranked_lists, k=settings.rrf_constant, top_k=top_k
            )
            logger.debug(
                "DefaultRetrievalBackend: hybrid fused %d component lists "
                "into %d results.",
                len(used_components),
                len(fused),
            )
            return fused

        # Non-hybrid strategy: leaf dispatch (no recursion).
        return self._run_component(
            strategy, query, top_k=top_k, prior_context=prior_context
        )


# ---------------------------------------------------------------------------
# Topological scheduling (pure, unit-testable)
# ---------------------------------------------------------------------------


def topological_order(steps: list[DecompositionStep]) -> list[str]:
    """Return the step ids of *steps* in dependency order.

    A step only appears after every step it ``depends_on``.  The input's
    declared order is used as the tiebreaker so deterministic input
    produces deterministic output.  A cycle (which :func:`validate_steps`
    already rejects upstream) raises ``ValueError``.

    Args:
        steps: The ordered sub-queries from a
            :class:`~reporag.agent.planner.DecompositionPlan`.

    Returns:
        The step ids in a valid topological order.

    Raises:
        ValueError: If the dependency graph contains a cycle or a
            ``depends_on`` entry references an unknown step id.
    """
    by_id = {step.id: step for step in steps}
    order: list[str] = []
    done: set[str] = set()
    in_progress: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in done:
            return
        if step_id in in_progress:
            raise ValueError(f"dependency cycle detected at step {step_id!r}")
        step = by_id.get(step_id)
        if step is None:
            raise ValueError(f"step {step_id!r} is a dependency but is not in the plan")
        in_progress.add(step_id)
        for dep in step.depends_on:
            visit(dep)
        in_progress.discard(step_id)
        done.add(step_id)
        order.append(step_id)

    for step in steps:
        visit(step.id)
    return order


# ---------------------------------------------------------------------------
# SubQueryExecutor
# ---------------------------------------------------------------------------


class SubQueryExecutor:
    """Executes a :class:`DecompositionPlan`'s sub-queries in dependency order.

    Builds a :class:`RoutingPlan` with an injected
    :class:`StrategyRouter` (or one is constructed lazily), schedules the
    steps with :func:`topological_order`, and runs each step against the
    injected :class:`RetrievalBackend`, forwarding prior-step results as
    context.  Failed steps are retried once (when
    ``settings.executor_retry_failed_steps`` is True) and then skipped so
    one bad step never aborts the whole plan.

    Args:
        backend: A :class:`RetrievalBackend` (or any duck-typed object
            exposing ``retrieve(strategy, query, *, top_k, prior_context,
            hybrid_components)``).  Passing one is the supported test
            seam -- it makes the executor network-free and avoids Qdrant /
            Neo4j / model loads.  When ``None`` (default) a
            :class:`DefaultRetrievalBackend` is constructed lazily on the
            first :meth:`execute` call.
        router: A pre-built :class:`StrategyRouter` to reuse (e.g. to
            share one instance's LLM connection).  When ``None`` a new one
            is constructed lazily.
        top_k: Per-step result cap passed through to the backend.
            Defaults to ``settings.strategy_router_default_top_k``.
        context_top_k: How many results from each dependency to forward
            as context to a dependent step.  Defaults to ``top_k`` (i.e.
            forward every dependency result).
        retry_failed: When ``True`` (default, mirrors
            ``settings.executor_retry_failed_steps``) a failed step is
            retried once before being skipped.

    Raises:
        ValueError: If *top_k* or *context_top_k* is ``< 1``.
    """

    def __init__(
        self,
        backend: RetrievalBackend | None = None,
        *,
        router: StrategyRouter | None = None,
        top_k: int | None = None,
        context_top_k: int | None = None,
        retry_failed: bool | None = None,
    ) -> None:
        if top_k is None:
            top_k = settings.strategy_router_default_top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}.")
        if context_top_k is None:
            context_top_k = top_k
        if context_top_k < 1:
            raise ValueError(f"context_top_k must be >= 1, got {context_top_k!r}.")

        self._backend: RetrievalBackend | None = backend
        self._router: StrategyRouter | None = router
        self.top_k = top_k
        self.context_top_k = context_top_k
        self.retry_failed = (
            retry_failed
            if retry_failed is not None
            else settings.executor_retry_failed_steps
        )

    @property
    def backend(self) -> RetrievalBackend:
        """The retrieval backend, defaulting to DefaultRetrievalBackend."""
        if self._backend is None:
            self._backend = DefaultRetrievalBackend()
        return self._backend

    @property
    def router(self) -> StrategyRouter:
        """The strategy router, defaulting to a fresh StrategyRouter."""
        if self._router is None:
            self._router = StrategyRouter()
        return self._router

    # ------------------------------------------------------------------
    # Context assembly (pure helper)
    # ------------------------------------------------------------------

    def _build_prior_context(
        self,
        step: DecompositionStep,
        results_by_id: dict[str, StepResult],
    ) -> str:
        """Assemble the forward-context text for *step* from its dependencies.

        Collects the ``chunk_text`` of every dependency's results, dedups
        by ``(file_path, start_line, end_line)`` (so the same chunk from
        two dependencies is not repeated), caps each dependency at
        :attr:`context_top_k` results, and joins them with a header line
        per contributing dependency.  Returns ``""`` when the step has no
        dependencies, or when all dependencies have no / skipped results.
        """
        if not step.depends_on:
            return ""

        seen: set[tuple[str, int | None, int | None]] = set()
        chunks: list[str] = []
        for dep_id in step.depends_on:
            dep_result = results_by_id.get(dep_id)
            if dep_result is None or not dep_result.results:
                continue
            for r in dep_result.results[: self.context_top_k]:
                key = (r.file_path, r.start_line, r.end_line)
                if key in seen:
                    continue
                seen.add(key)
                if r.chunk_text.strip():
                    chunks.append(r.chunk_text)
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Single-step execution (with retry)
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step: DecompositionStep,
        strategy: RetrievalStrategy,
        prior_context: str,
        hybrid_components: tuple[RetrievalStrategy, ...],
    ) -> StepResult:
        """Run one retrieval attempt, retrying once on failure when enabled."""
        attempts = 0
        last_error: str | None = None
        max_attempts = 2 if self.retry_failed else 1

        while attempts < max_attempts:
            attempts += 1
            try:
                results = self.backend.retrieve(
                    strategy,
                    step.query,
                    top_k=self.top_k,
                    prior_context=prior_context,
                    hybrid_components=hybrid_components,
                )
                return StepResult(
                    step_id=step.id,
                    strategy=strategy,
                    results=results,
                    attempted=attempts,
                    status="ok",
                    metadata={
                        "context_chars": len(prior_context),
                        "context_chunks": (
                            prior_context.count("\n") + 1
                            if prior_context.strip()
                            else 0
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - any backend failure
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "SubQueryExecutor: step %r attempt %d/%d failed (%s).",
                    step.id,
                    attempts,
                    max_attempts,
                    last_error,
                )

        # Exhausted retries -> skip the step, but let the plan continue.
        return StepResult(
            step_id=step.id,
            strategy=strategy,
            results=[],
            attempted=attempts,
            status="skipped",
            error=last_error,
            metadata={
                "context_chars": len(prior_context),
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: DecompositionPlan | list[DecompositionStep],
        routing: RoutingPlan | None = None,
    ) -> ExecutionPlan:
        """Execute every step of *plan* in dependency order.

        Args:
            plan: Either a :class:`~reporag.agent.planner.DecompositionPlan`
                (produced by
                :class:`~reporag.agent.planner.QueryDecomposer`) or a bare
                list of :class:`~reporag.agent.planner.DecompositionStep`
                objects (the shape in Issue 22's "How to test locally"
                snippet -- ``executor.execute(plan.steps)``).  A bare list
                has no ``original_query`` and no classification metadata,
                so the resulting :class:`ExecutionPlan.original_query` is
                synthesised from the step queries.
            routing: An optional pre-computed :class:`RoutingPlan`.  When
                ``None`` (the common case) a new :class:`StrategyRouter`
                is used to route the plan first -- this is the
                "router then executor" flow the issue describes.  Passing
                a routing explicitly lets a caller reuse routing decisions
                (e.g. for retries / observability) without re-calling the
                LLM.

        Returns:
            An :class:`ExecutionPlan` with one :class:`StepResult` per
            step, in execution (topological) order.
        """
        if isinstance(plan, DecompositionPlan):
            original_query = plan.original_query
            steps = list(plan.steps)
        else:
            steps = list(plan)
            original_query = " ".join(step.query for step in steps)

        if not steps:
            raise ValueError("plan must contain at least one step.")

        if routing is None:
            routing = self.router.route_batch(steps, original_query=original_query)

        ordered_ids = topological_order(steps)
        steps_by_id = {step.id: step for step in steps}

        results_by_id: dict[str, StepResult] = {}
        executed_order: list[str] = []

        for step_id in ordered_ids:
            step = steps_by_id[step_id]

            # If any dependency was skipped, we cannot safely forward
            # context for this step.  Mark it skipped and continue rather
            # than running against incomplete context.
            skipped_deps = [
                dep
                for dep in step.depends_on
                if dep in results_by_id and results_by_id[dep].status != "ok"
            ]
            if skipped_deps:
                logger.warning(
                    "SubQueryExecutor: skipping step %r because "
                    "dependencies %s did not complete.",
                    step_id,
                    skipped_deps,
                )
                strategy = routing.strategy_for(step_id)
                results_by_id[step_id] = StepResult(
                    step_id=step_id,
                    strategy=strategy,
                    results=[],
                    attempted=0,
                    status="skipped_no_deps",
                    error=f"dependencies skipped: {skipped_deps}",
                )
                executed_order.append(step_id)
                continue

            strategy = routing.strategy_for(step_id)
            prior_context = self._build_prior_context(step, results_by_id)
            results_by_id[step_id] = self._run_step(
                step,
                strategy,
                prior_context,
                self.router.hybrid_components,
            )
            executed_order.append(step_id)

        ok_count = sum(1 for r in results_by_id.values() if r.status == "ok")
        skip_count = len(results_by_id) - ok_count
        return ExecutionPlan(
            original_query=original_query,
            step_results=results_by_id,
            executed_order=tuple(executed_order),
            routed_strategies={
                step_id: results_by_id[step_id].strategy for step_id in executed_order
            },
            metadata={
                "ok": ok_count,
                "skipped": skip_count,
                "retried": sum(1 for r in results_by_id.values() if r.attempted >= 2),
            },
        )

    def __repr__(self) -> str:
        return (
            f"SubQueryExecutor(top_k={self.top_k}, "
            f"retry_failed={self.retry_failed})"
        )

import logging
from dataclasses import dataclass
from typing import Any

from reporag.agent.planner import DecompositionStep
from reporag.agent.router import StrategyRouter
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEngines:
    """Container for the three core retrieval backends."""

    vector: Any
    bm25: Any
    graph: Any


class SubQueryExecutor:
    """Executes sub-queries in dependency order, orchestrating the retrieval engines."""

    def __init__(
        self,
        engines: RetrievalEngines,
        router: StrategyRouter | None = None,
    ) -> None:
        """Initialize with retrieval engines and an optional router."""
        self.engines = engines
        self.router = router or StrategyRouter()

    def execute(
        self, steps: tuple[DecompositionStep, ...]
    ) -> dict[str, list[RetrievalResult]]:
        """Execute a plan in topological order and return a dict of results per step."""
        import graphlib

        # Build dependency graph (using tuples, not sets, to ensure determinism)
        graph: dict[str, tuple[str, ...]] = {step.id: step.depends_on for step in steps}

        try:
            sorter = graphlib.TopologicalSorter(graph)
            execution_order = list(sorter.static_order())
        except graphlib.CycleError as e:
            raise ValueError(f"Cycle detected in plan dependencies: {e}") from e

        # Map step_id to step object for easy lookup
        step_map = {step.id: step for step in steps}
        results: dict[str, list[RetrievalResult]] = {}

        for step_id in execution_order:
            if step_id not in step_map:
                continue

            step = step_map[step_id]

            # 1. Build context from dependencies
            context_chunks = []
            for dep_id in step.depends_on:
                dep_results = results.get(dep_id, [])
                for r in dep_results:
                    if r.chunk_text:
                        context_chunks.append(r.chunk_text)

            augmented_query = step.query
            if context_chunks:
                joined_context = "\n".join(context_chunks)
                augmented_query = (
                    f"{step.query}\n\nContext from previous steps:\n{joined_context}"
                )

            # 2. Route
            decision = self.router.route(step.query)

            # 3. Execute
            results[step.id] = self._dispatch(step.id, decision, augmented_query)

        return results

    def _execute_backend(
        self, step_id: str, backend_name: str, func: Any, *args: Any
    ) -> list[RetrievalResult]:
        """Execute a single backend function with exactly one retry on exception."""
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                return func(*args)
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        "Step %s backend '%s' failed (attempt 1/2): %s. Retrying...",
                        step_id,
                        backend_name,
                        e,
                    )
                else:
                    logger.error(
                        "Step %s backend '%s' failed (attempt 2/2): %s. Skipping.",
                        step_id,
                        backend_name,
                        e,
                    )
        return []

    def _dispatch(
        self, step_id: str, decision: Any, augmented_query: str
    ) -> list[RetrievalResult]:
        """Dispatch to the correct backend(s) based on the routing decision."""
        from reporag.retrieval.fusion import reciprocal_rank_fusion

        if decision.strategy == "bm25":
            return self._execute_backend(
                step_id, "bm25", self.engines.bm25.search, augmented_query
            )
        elif decision.strategy == "vector":
            return self._execute_backend(
                step_id, "vector", self.engines.vector.search, augmented_query
            )
        elif decision.strategy == "graph":
            if not decision.symbol:
                # Fallback if router chose graph but failed to extract symbol
                return reciprocal_rank_fusion(
                    [
                        self._execute_backend(
                            step_id,
                            "vector",
                            self.engines.vector.search,
                            augmented_query,
                        ),
                        self._execute_backend(
                            step_id, "bm25", self.engines.bm25.search, augmented_query
                        ),
                    ]
                )
            return self._execute_backend(
                step_id, "graph", self.engines.graph.get_neighbors, decision.symbol
            )
        elif decision.strategy == "hybrid":
            lists_to_fuse = [
                self._execute_backend(
                    step_id, "vector", self.engines.vector.search, augmented_query
                ),
                self._execute_backend(
                    step_id, "bm25", self.engines.bm25.search, augmented_query
                ),
            ]
            if decision.symbol:
                lists_to_fuse.append(
                    self._execute_backend(
                        step_id,
                        "graph",
                        self.engines.graph.get_neighbors,
                        decision.symbol,
                    )
                )
            return reciprocal_rank_fusion(lists_to_fuse)
        else:
            raise ValueError(f"Unknown strategy: {decision.strategy}")

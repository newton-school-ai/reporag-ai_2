"""Sub-query executor.

Executes sub-queries in dependency order, passing context from earlier
steps to later ones. Orchestrates the retrieval engine based on the
strategy assigned by the router.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.reporag.agent.planner import DecompositionPlan, DecompositionStep
from src.reporag.agent.router import StrategyRouter
from src.reporag.config import settings
from src.reporag.retrieval.fusion import reciprocal_rank_fusion
from src.reporag.retrieval.graph_traversal import (
    AmbiguousSymbolError,
    SymbolNotFoundError,
)
from src.reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


def topological_sort(steps: tuple[DecompositionStep, ...]) -> list[DecompositionStep]:
    """Topologically sort sub-queries by dependency edges (Kahn's algorithm)."""
    adj = {step.id: [] for step in steps}
    in_degree = {step.id: 0 for step in steps}
    step_map = {step.id: step for step in steps}

    for step in steps:
        for dep in step.depends_on:
            if dep in adj:
                adj[dep].append(step.id)
                in_degree[step.id] += 1

    # Deterministic start: sort queue of nodes with in-degree 0
    queue = [step_id for step_id, deg in in_degree.items() if deg == 0]
    queue.sort()

    ordered_ids = []
    while queue:
        curr = queue.pop(0)
        ordered_ids.append(curr)
        neighbors = adj[curr]
        neighbors.sort()  # deterministic traversal
        for neighbor in neighbors:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered_ids) != len(steps):
        logger.warning(
            "Cycle or invalid dependency detected in steps; falling back to original order."
        )
        return list(steps)

    return [step_map[step_id] for step_id in ordered_ids]


def extract_symbol_from_query(query: str) -> str:
    """Extract candidate symbol/identifier from sub-query text."""
    # Look for quoted symbol names first
    quoted = re.findall(r"[`'\"]([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]", query)
    if quoted:
        return quoted[0]

    # Look for snake_case, camelCase, or PascalCase words
    words = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_\.:]*)\b", query)
    stop_words = {
        "what",
        "who",
        "where",
        "how",
        "why",
        "which",
        "function",
        "class",
        "method",
        "variable",
        "calls",
        "called",
        "caller",
        "call",
        "dependencies",
        "dependency",
        "depends",
        "import",
        "imports",
        "imported",
        "define",
        "defined",
        "definition",
        "references",
        "reference",
        "find",
        "show",
        "get",
        "lookup",
        "trace",
        "auth",
        "user",
    }
    candidates = [w for w in words if w.lower() not in stop_words]
    if candidates:
        # Prefer names containing dot, colon, underscore or camelCase transitions
        for c in candidates:
            if "_" in c or "." in c or ":" in c or re.search(r"[a-z][A-Z]", c):
                return c
        return candidates[0]

    # Fallback to the last alphanumeric word
    alnum_words = [w for w in words if w.isalnum()]
    return alnum_words[-1] if alnum_words else query


def get_retriever_from_engine(engine: Any, strategy: str) -> Any:
    """Resolve specific retriever from composite retrieval engine using duck-typing."""
    if isinstance(engine, dict):
        return engine.get(strategy)

    # Check common attribute patterns on structured engine objects
    attr_map = {
        "vector": ["vector", "vector_search"],
        "bm25": ["bm25", "bm25_search"],
        "graph": ["graph", "graph_retriever", "graph_traversal"],
    }
    attrs = attr_map.get(strategy, [])
    for attr in attrs:
        if hasattr(engine, attr):
            return getattr(engine, attr)

    # Fallback: if engine itself appears to match by class name
    class_name = engine.__class__.__name__.lower()
    if strategy in class_name:
        return engine

    return None


class SubQueryExecutor:
    """Executes sub-queries in dependency order, propagating context and handling failures.

    Orchestrates retrieval engines (vector, BM25, graph, hybrid) using strategies
    assigned by a StrategyRouter.
    """

    def __init__(
        self,
        retrieval_engine: Any,
        router: StrategyRouter | None = None,
    ) -> None:
        """Initialize the SubQueryExecutor.

        Args:
            retrieval_engine: The unified or composite search/retrieval engine.
            router: Optional StrategyRouter. If omitted, a default is created.
        """
        self.retrieval_engine = retrieval_engine
        self.router = router or StrategyRouter()

    def execute(
        self,
        plan_or_steps: DecompositionPlan | Iterable[DecompositionStep],
    ) -> dict[str, list[RetrievalResult]]:
        """Executes a list of steps or plan in topological order, returning results per step.

        Args:
            plan_or_steps: DecompositionPlan or iterable of DecompositionStep objects.

        Returns:
            A dictionary mapping step IDs to lists of RetrievalResult objects.
        """
        if hasattr(plan_or_steps, "steps"):
            steps = plan_or_steps.steps
        else:
            steps = list(plan_or_steps)

        if not steps:
            return {}

        # 1. Sort steps topologically
        ordered_steps = topological_sort(tuple(steps))

        step_results: dict[str, list[RetrievalResult]] = {}

        # 2. Iterate and execute steps
        for step in ordered_steps:
            # Inject context from dependencies into the query
            query = self._inject_context(step.query, step, step_results)

            # Determine routing strategy
            strategy = self.router.route(query)

            # Execute with failure tolerance (retry once, then skip)
            results = self._execute_step_with_retry(step, query, strategy)
            step_results[step.id] = results

        return step_results

    def _inject_context(
        self,
        query: str,
        step: DecompositionStep,
        step_results: dict[str, list[RetrievalResult]],
    ) -> str:
        """Substitute dependency step placeholders with actual extracted contexts."""
        modified_query = query

        # Find step references (e.g., step-1, step_1, step1)
        step_refs = re.findall(r"\bstep[-_]?\d+\b", modified_query, re.IGNORECASE)
        for ref in step_refs:
            normalized_ref = ref.lower().replace("_", "-")
            matched_id = None
            for step_id in step_results:
                if step_id.lower().replace("_", "-") == normalized_ref:
                    matched_id = step_id
                    break

            if matched_id and step_results[matched_id]:
                # Extract best symbol name or file path from results of that step
                results = step_results[matched_id]
                context_val = None
                for r in results:
                    if r.symbol_name:
                        context_val = r.symbol_name
                        break
                if not context_val:
                    for r in results:
                        if r.file_path:
                            context_val = Path(r.file_path).name
                            break
                if not context_val:
                    context_val = results[0].chunk_text[:50]

                if context_val:
                    modified_query = re.sub(
                        rf"\b{re.escape(ref)}\b",
                        context_val,
                        modified_query,
                        flags=re.IGNORECASE,
                    )

        # Inject extra symbol/path context metadata from dependencies if not explicitly referenced
        extra_parts = []
        for dep_id in step.depends_on:
            if dep_id in step_results and step_results[dep_id]:
                symbols = {r.symbol_name for r in step_results[dep_id] if r.symbol_name}
                paths = {r.file_path for r in step_results[dep_id] if r.file_path}
                if symbols:
                    extra_parts.append(
                        f"Context symbols from {dep_id}: " + ", ".join(symbols)
                    )
                elif paths:
                    extra_parts.append(
                        f"Context files from {dep_id}: " + ", ".join(paths)
                    )

        if extra_parts:
            modified_query += "\n" + "\n".join(extra_parts)

        return modified_query

    def _execute_step_with_retry(
        self,
        step: DecompositionStep,
        query: str,
        strategy: str,
    ) -> list[RetrievalResult]:
        """Execute query search with a single retry on failure."""
        attempts = 2
        for attempt in range(attempts):
            try:
                return self._run_strategy_retrieval(query, strategy)
            except Exception as e:
                logger.warning(
                    f"Execution attempt {attempt + 1} failed for step {step.id} "
                    f"using strategy {strategy}: {e}"
                )
                if attempt == attempts - 1:
                    logger.error(
                        f"Step {step.id} failed after {attempts} attempts. "
                        f"Skipping and continuing."
                    )
        return []

    def _run_strategy_retrieval(
        self, query: str, strategy: str
    ) -> list[RetrievalResult]:
        """Perform search using the resolved retriever for the chosen strategy."""
        if strategy == "hybrid":
            return self._run_hybrid_fusion(query)

        retriever = get_retriever_from_engine(self.retrieval_engine, strategy)
        if not retriever:
            logger.warning(
                f"No retriever found for strategy {strategy} on engine; "
                f"falling back to vector search."
            )
            retriever = get_retriever_from_engine(self.retrieval_engine, "vector")
            if not retriever:
                raise RuntimeError(
                    "No vector retriever available as ultimate fallback."
                )
            strategy = "vector"

        if strategy == "graph":
            # For graph, extract symbol name and perform structural traversal
            symbol = extract_symbol_from_query(query)
            try:
                # If query contains "calls" or "called by", fetch callers
                if any(x in query.lower() for x in ["calls", "called", "caller"]):
                    return retriever.get_callers(symbol)
                # Otherwise fetch immediate neighbors
                return retriever.get_neighbors(symbol, direction="both")
            except (SymbolNotFoundError, AmbiguousSymbolError) as e:
                logger.warning(
                    f"Graph traversal failed for symbol '{symbol}': {e}; falling back to vector."
                )
                vector_retriever = get_retriever_from_engine(
                    self.retrieval_engine, "vector"
                )
                if vector_retriever:
                    return vector_retriever.search(query)
                raise

        # For vector or bm25, execute normal query search
        return retriever.search(query)

    def _run_hybrid_fusion(self, query: str) -> list[RetrievalResult]:
        """Run BM25 and vector search concurrently/sequentially and fuse results."""
        vector_retriever = get_retriever_from_engine(self.retrieval_engine, "vector")
        bm25_retriever = get_retriever_from_engine(self.retrieval_engine, "bm25")

        vector_results = []
        if vector_retriever:
            try:
                vector_results = vector_retriever.search(query)
            except Exception as e:
                logger.warning(f"Hybrid route - Vector search failed: {e}")

        bm25_results = []
        if bm25_retriever:
            try:
                bm25_results = bm25_retriever.search(query)
            except Exception as e:
                logger.warning(f"Hybrid route - BM25 search failed: {e}")

        if not vector_results and not bm25_results:
            return []

        # Reciprocal Rank Fusion
        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=settings.rrf_constant,
            top_k=settings.vector_search_top_k,
        )

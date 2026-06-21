"""Unit tests for the agentic query planner.

Tests cover the planner module defined in src/reporag/agent/planner.py.
Currently validates the module stub; test bodies will be implemented
in Issues 20-21.
"""

import importlib

import pytest


class TestPlannerModuleExists:
    """Verify the planner module is importable and properly documented."""

    def test_planner_module_importable(self) -> None:
        """Planner module can be imported without error."""
        mod = importlib.import_module("reporag.agent.planner")
        assert mod is not None

    def test_planner_module_has_docstring(self) -> None:
        """Planner module has a module-level docstring."""
        mod = importlib.import_module("reporag.agent.planner")
        assert mod.__doc__ is not None
        assert "planner" in mod.__doc__.lower() or "query" in mod.__doc__.lower()

    def test_router_module_importable(self) -> None:
        """Router module can be imported without error."""
        mod = importlib.import_module("reporag.agent.router")
        assert mod is not None

    def test_executor_module_importable(self) -> None:
        """Executor module can be imported without error."""
        mod = importlib.import_module("reporag.agent.executor")
        assert mod is not None

    def test_synthesizer_module_importable(self) -> None:
        """Synthesizer module can be imported without error."""
        mod = importlib.import_module("reporag.agent.synthesizer")
        assert mod is not None


@pytest.mark.skip(reason="Not yet implemented -- Issue 20")
class TestQueryClassifier:
    """Query classifier should categorize queries by complexity."""

    def test_simple_lookup_classification(self) -> None:
        """Simple identifier lookup queries are classified as simple-lookup."""

    def test_multi_hop_classification(self) -> None:
        """Multi-step questions are classified as multi-hop."""

    def test_exploratory_classification(self) -> None:
        """Open-ended questions are classified as exploratory."""

    def test_low_confidence_defaults_to_multi_hop(self) -> None:
        """Queries with low classification confidence fall back to multi-hop."""

    def test_classification_returns_confidence(self) -> None:
        """Classification result includes a confidence score."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 21")
class TestQueryDecomposer:
    """Query decomposer should break complex queries into sub-queries."""

    def test_simple_query_no_decomposition(self) -> None:
        """Simple queries produce a single sub-query (no decomposition)."""

    def test_multi_hop_decomposition(self) -> None:
        """Multi-hop queries are decomposed into ordered sub-queries."""

    def test_sub_queries_have_dependency_edges(self) -> None:
        """Sub-queries include dependency edges for execution ordering."""

    def test_sub_query_has_expected_answer_type(self) -> None:
        """Each sub-query specifies an expected answer type."""

    def test_sub_query_context_from_prior_steps(self) -> None:
        """Sub-queries reference prior step IDs for context passing."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 22")
class TestStrategyRouter:
    """Router should assign retrieval strategies to sub-queries."""

    def test_identifier_query_routes_to_bm25(self) -> None:
        """Queries mentioning identifiers route to BM25 strategy."""

    def test_structural_query_routes_to_graph(self) -> None:
        """Structural queries like 'what calls X' route to graph strategy."""

    def test_semantic_query_routes_to_vector(self) -> None:
        """Semantic queries like 'how does X work' route to vector strategy."""

    def test_ambiguous_query_routes_to_hybrid(self) -> None:
        """Ambiguous queries default to hybrid strategy."""

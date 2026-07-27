"""Unit tests for GraphRetriever (Issue 18).

Fixture topology mirrors the approach in ``test_neo4j_store.py``: a small,
fully-specified graph is built via ``NetworkXGraphStore`` so no external
service is required.  Each logical concern is grouped into a named class,
matching the style used in ``test_bm25_search.py`` and ``test_neo4j_store.py``.

Call chain used throughout::

    func_login  --CALLS-->  func_auth  --CALLS-->  func_verify

``class_user`` is intentionally disconnected to test the no-path case.
"""

from __future__ import annotations

import pytest

from reporag.graph.call_graph import CallEdge
from reporag.graph.dependency_graph import DependencyEdge
from reporag.graph.neo4j_store import NetworkXGraphStore
from reporag.graph.symbol_table import SymbolRecord, SymbolTable
from reporag.retrieval.graph_traversal import GraphRetriever
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_graph_store() -> NetworkXGraphStore:
    """Fixture that builds a realistic small graph store with NetworkX fallback."""
    # 1. Create SymbolTable
    table = SymbolTable()
    records = [
        SymbolRecord(
            symbol_id="func_auth",
            name="authenticate",
            qualified_name="auth.authenticate",
            type="function",
            file_path="src/auth.py",
            module="auth",
            start_line=10,
            end_line=20,
            signature="def authenticate(token: str):",
            docstring="Authenticates user.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="func_verify",
            name="verify_token",
            qualified_name="auth.verify_token",
            type="function",
            file_path="src/auth.py",
            module="auth",
            start_line=22,
            end_line=30,
            signature="def verify_token(token: str):",
            docstring="Verifies the JWT token.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="func_login",
            name="login",
            qualified_name="api.login",
            type="function",
            file_path="src/api.py",
            module="api",
            start_line=5,
            end_line=15,
            signature="def login(req):",
            docstring="Login endpoint.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="class_user",
            name="User",
            qualified_name="models.User",
            type="class",
            file_path="src/models.py",
            module="models",
            start_line=1,
            end_line=5,
            signature="class User:",
            docstring="User model.",
            parent="",
            is_async=False,
            language="python",
        ),
    ]
    for r in records:
        table.add(r)

    # 2. Call Edges (login -> authenticate -> verify_token)
    call_edges = [
        CallEdge(
            caller="api.login",
            callee="auth.authenticate",
            caller_file="src/api.py",
            callee_file="src/auth.py",
            call_type="direct",
            resolution="exact",
            call_site_line=10,
        ),
        CallEdge(
            caller="auth.authenticate",
            callee="auth.verify_token",
            caller_file="src/auth.py",
            callee_file="src/auth.py",
            call_type="direct",
            resolution="exact",
            call_site_line=15,
        ),
    ]

    # 3. Dependency Edges (api -> auth)
    dep_edges = [
        DependencyEdge(
            source_module="api",
            target_module="auth",
            import_type="direct",
            line=1,
            resolved=True,
            source="src/api.py",
            target="src/auth.py",
        )
    ]

    # 4. Initialize store and persist
    store = NetworkXGraphStore()
    store.persist_graph(call_edges, dep_edges, table)
    return store


@pytest.fixture
def graph_retriever(test_graph_store: NetworkXGraphStore) -> GraphRetriever:
    return GraphRetriever(store=test_graph_store)


# ---------------------------------------------------------------------------
# Schema parity with vector search
# ---------------------------------------------------------------------------


class TestSchemaParity:
    """GraphRetriever must return the exact same RetrievalResult type as VectorSearch."""

    def test_returns_retrieval_result_instances(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.get_neighbors("func_auth")
        assert results
        assert isinstance(results[0], RetrievalResult)

    def test_result_has_all_expected_fields(
        self, graph_retriever: GraphRetriever
    ) -> None:
        r = graph_retriever.get_neighbors("func_auth")[0]
        assert hasattr(r, "score")
        assert hasattr(r, "file_path")
        assert hasattr(r, "start_line")
        assert hasattr(r, "end_line")
        assert hasattr(r, "symbol_name")
        assert hasattr(r, "chunk_text")
        assert hasattr(r, "metadata")

    def test_metadata_contains_node_properties(
        self, graph_retriever: GraphRetriever
    ) -> None:
        r = graph_retriever.get_neighbors("func_auth")[0]
        assert "type" in r.metadata

    def test_score_is_positive(self, graph_retriever: GraphRetriever) -> None:
        for r in graph_retriever.get_neighbors("func_auth", depth=1):
            assert r.score > 0


# ---------------------------------------------------------------------------
# chunk_text synthesis
# ---------------------------------------------------------------------------


class TestChunkTextSynthesis:
    """chunk_text must be built from signature + docstring in the correct format."""

    def test_signature_and_docstring_joined(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.get_neighbors("func_login", depth=1, direction="out")
        auth_result = next(r for r in results if r.symbol_name == "authenticate")
        expected = 'def authenticate(token: str):\n"""Authenticates user."""'
        assert auth_result.chunk_text == expected

    def test_node_with_no_signature_falls_back_to_name(
        self, test_graph_store: NetworkXGraphStore
    ) -> None:
        """A node with an empty signature but a name should use the name as chunk_text."""
        # Manually add a sparse node directly to the underlying NetworkX graph.
        test_graph_store.graph.add_node(
            "sparse_node",
            symbol_id="sparse_node",
            name="sparse_func",
            qualified_name="mod.sparse_func",
            type="function",
            file_path="src/mod.py",
            module="mod",
            start_line=1,
            end_line=2,
            signature="",  # empty signature
            docstring="",  # empty docstring
        )
        retriever = GraphRetriever(store=test_graph_store)
        # Build a one-hop neighbor edge so get_neighbors can reach it.
        test_graph_store.graph.add_edge("func_login", "sparse_node", type="CALLS")
        results = retriever.get_neighbors("func_login", depth=1, direction="out")
        sparse = next((r for r in results if r.symbol_name == "sparse_func"), None)
        assert sparse is not None
        assert sparse.chunk_text == "sparse_func"

    def test_node_with_no_name_no_signature_no_docstring_returns_empty_string(
        self, test_graph_store: NetworkXGraphStore
    ) -> None:
        """A node with no name, no signature, and no docstring produces empty chunk_text."""
        test_graph_store.graph.add_node(
            "ghost_node",
            symbol_id="ghost_node",
            type="function",
            file_path="",
        )
        test_graph_store.graph.add_edge("func_login", "ghost_node", type="CALLS")
        retriever = GraphRetriever(store=test_graph_store)
        results = retriever.get_neighbors("func_login", depth=1, direction="out")
        ghost = next((r for r in results if r.symbol_name is None), None)
        assert ghost is not None, "ghost_node must be reachable via the CALLS edge"
        assert ghost.chunk_text == ""


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    """get_neighbors must traverse call edges in both, in, and out directions."""

    def test_both_directions_returns_callers_and_callees(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.get_neighbors("func_auth", depth=1, direction="both")
        symbol_names = {r.symbol_name for r in results}
        assert "verify_token" in symbol_names
        assert "login" in symbol_names

    def test_out_direction_returns_only_callees(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.get_neighbors("func_auth", depth=1, direction="out")
        symbol_names = {r.symbol_name for r in results}
        assert "verify_token" in symbol_names
        assert "login" not in symbol_names

    def test_unknown_symbol_returns_empty_list(
        self, graph_retriever: GraphRetriever
    ) -> None:
        """NetworkXGraphStore.get_neighbors returns [] when node not in graph (line 532)."""
        assert graph_retriever.get_neighbors("nonexistent_symbol") == []

    def test_depth_1_score_greater_than_depth_2_score(
        self, graph_retriever: GraphRetriever
    ) -> None:
        """score = 1/(depth+1): depth-1 results must score higher than depth-2 results."""
        results_d1 = graph_retriever.get_neighbors(
            "func_login", depth=1, direction="out"
        )
        results_d2 = graph_retriever.get_neighbors(
            "func_login", depth=2, direction="out"
        )
        assert results_d1
        assert results_d2
        # All depth-1 results score 0.5; all depth-2 results score 0.33
        assert results_d1[0].score > results_d2[0].score


# ---------------------------------------------------------------------------
# get_callers
# ---------------------------------------------------------------------------


class TestGetCallers:
    """get_callers must follow only incoming CALLS edges."""

    def test_direct_caller_is_returned(self, graph_retriever: GraphRetriever) -> None:
        results = graph_retriever.get_callers("func_auth", depth=1)
        symbol_names = {r.symbol_name for r in results}
        assert "login" in symbol_names

    def test_callee_is_not_returned(self, graph_retriever: GraphRetriever) -> None:
        results = graph_retriever.get_callers("func_auth", depth=1)
        symbol_names = {r.symbol_name for r in results}
        assert "verify_token" not in symbol_names

    def test_deep_caller_reachable_at_depth_2(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.get_callers("func_verify", depth=2)
        symbol_names = {r.symbol_name for r in results}
        assert "authenticate" in symbol_names
        assert "login" in symbol_names

    def test_unknown_symbol_returns_empty_list(
        self, graph_retriever: GraphRetriever
    ) -> None:
        assert graph_retriever.get_callers("nonexistent_symbol") == []


# ---------------------------------------------------------------------------
# find_paths
# ---------------------------------------------------------------------------


class TestFindPaths:
    """find_paths must return the shortest path as an ordered list of RetrievalResult."""

    def test_path_contains_all_intermediate_nodes(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.find_paths("func_login", "func_verify")
        assert len(results) == 3
        names = [r.symbol_name for r in results]
        assert names == ["login", "authenticate", "verify_token"]

    def test_source_node_has_score_1(self, graph_retriever: GraphRetriever) -> None:
        results = graph_retriever.find_paths("func_login", "func_verify")
        assert results[0].score == pytest.approx(1.0)

    def test_intermediate_node_score_decreases_with_distance(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.find_paths("func_login", "func_verify")
        assert results[1].score == pytest.approx(0.5)
        assert results[2].score == pytest.approx(1 / 3, abs=0.001)

    def test_no_path_returns_empty_list(self, graph_retriever: GraphRetriever) -> None:
        """class_user is disconnected from the call chain."""
        assert graph_retriever.find_paths("func_login", "class_user") == []

    def test_unknown_source_returns_empty_list(
        self, graph_retriever: GraphRetriever
    ) -> None:
        """NetworkXGraphStore.shortest_path returns [] on unknown node (line 595)."""
        assert graph_retriever.find_paths("nonexistent", "func_verify") == []

    def test_unknown_target_returns_empty_list(
        self, graph_retriever: GraphRetriever
    ) -> None:
        assert graph_retriever.find_paths("func_login", "nonexistent") == []

    def test_source_equals_target(self, graph_retriever: GraphRetriever) -> None:
        """A path from a node to itself should return a single-node list."""
        results = graph_retriever.find_paths("func_auth", "func_auth")
        # nx.shortest_path(G, x, x) returns [x], so we expect one result at distance 0.
        assert len(results) == 1
        assert results[0].symbol_name == "authenticate"
        assert results[0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_subgraph
# ---------------------------------------------------------------------------


class TestExtractSubgraph:
    """extract_subgraph must return one RetrievalResult per requested node."""

    def test_nodes_returned_for_each_requested_id(
        self, graph_retriever: GraphRetriever
    ) -> None:
        results = graph_retriever.extract_subgraph(["func_login", "func_auth"])
        assert len(results) == 2
        names = {r.symbol_name for r in results}
        assert names == {"login", "authenticate"}

    def test_all_results_have_score_1(self, graph_retriever: GraphRetriever) -> None:
        """Subgraph nodes are always at distance 0 -> score 1.0."""
        results = graph_retriever.extract_subgraph(["func_login", "func_auth"])
        for r in results:
            assert r.score == pytest.approx(1.0)

    def test_empty_input_returns_empty_list(
        self, graph_retriever: GraphRetriever
    ) -> None:
        """subgraph([]) must not raise and must return []."""
        assert graph_retriever.extract_subgraph([]) == []

    def test_unknown_ids_silently_omitted(
        self, graph_retriever: GraphRetriever
    ) -> None:
        """NetworkXGraphStore.subgraph skips unknown ids (line 636: has_node guard)."""
        results = graph_retriever.extract_subgraph(["nonexistent_id"])
        assert results == []

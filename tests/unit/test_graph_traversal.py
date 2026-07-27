"""Unit tests for the graph retrieval engine (Issue 18)."""

from __future__ import annotations

import pytest

from reporag.graph.call_graph import CallEdge
from reporag.graph.dependency_graph import DependencyEdge
from reporag.graph.neo4j_store import GraphStore, NetworkXGraphStore
from reporag.graph.symbol_table import SymbolRecord, SymbolTable
from reporag.retrieval.graph_traversal import (
    GraphPathResult,
    GraphRetrievalResult,
    GraphRetriever,
)


def _make_record(
    *,
    symbol_id: str,
    name: str,
    qualified_name: str,
    type: str,
    file_path: str = "app.py",
    module: str = "app",
    start_line: int = 1,
    end_line: int = 10,
    signature: str | None = None,
    docstring: str | None = None,
) -> SymbolRecord:
    """Build a minimal SymbolRecord for testing."""
    return SymbolRecord(
        symbol_id=symbol_id,
        name=name,
        qualified_name=qualified_name,
        type=type,
        file_path=file_path,
        module=module,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
    )


@pytest.fixture
def populated_retriever() -> GraphRetriever:
    """Return a GraphRetriever with a pre-populated NetworkXGraphStore."""
    store = GraphStore(fallback=True)
    assert isinstance(store, NetworkXGraphStore)

    # 1. Create a SymbolTable
    table = SymbolTable()

    # Symbols
    router = _make_record(
        symbol_id="web.router.handle_request",
        name="handle_request",
        qualified_name="web.router.handle_request",
        type="function",
        file_path="web/router.py",
        module="web.router",
        start_line=5,
        end_line=15,
        signature="def handle_request(req)",
        docstring="Handles incoming web request.",
    )
    auth = _make_record(
        symbol_id="auth.authenticate_user",
        name="authenticate_user",
        qualified_name="auth.authenticate_user",
        type="function",
        file_path="auth.py",
        module="auth",
        start_line=10,
        end_line=25,
        signature="def authenticate_user(user, password)",
        docstring="Authenticates user with db.",
    )
    db = _make_record(
        symbol_id="db.connection.execute_query",
        name="execute_query",
        qualified_name="db.connection.execute_query",
        type="function",
        file_path="db/connection.py",
        module="db.connection",
        start_line=20,
        end_line=30,
        signature="def execute_query(q)",
        docstring="Executes database query.",
    )
    helper = _make_record(
        symbol_id="utils.helper_func",
        name="helper_func",
        qualified_name="utils.helper_func",
        type="function",
        file_path="utils.py",
        module="utils",
        start_line=1,
        end_line=5,
        signature="def helper_func()",
        docstring="General helper.",
    )

    table.add(router)
    table.add(auth)
    table.add(db)
    table.add(helper)

    # 2. Call edges
    calls = [
        CallEdge(
            caller="web.router.handle_request",
            callee="auth.authenticate_user",
            caller_file="web/router.py",
            callee_file="auth.py",
            call_site_line=8,
            call_type="function",
            resolution="local",
        ),
        CallEdge(
            caller="auth.authenticate_user",
            callee="db.connection.execute_query",
            caller_file="auth.py",
            callee_file="db/connection.py",
            call_site_line=18,
            call_type="function",
            resolution="local",
        ),
        CallEdge(
            caller="web.router.handle_request",
            callee="utils.helper_func",
            caller_file="web/router.py",
            callee_file="utils.py",
            call_site_line=12,
            call_type="function",
            resolution="local",
        ),
    ]

    # 3. Dependency edges
    deps = [
        DependencyEdge(
            source="web/router.py",
            target="auth.py",
            source_module="web.router",
            target_module="auth",
            import_type="import",
            line=1,
        ),
        DependencyEdge(
            source="auth.py",
            target="db/connection.py",
            source_module="auth",
            target_module="db.connection",
            import_type="import",
            line=2,
        ),
    ]

    # Ingest topology
    store.persist_graph(calls, deps, table)

    return GraphRetriever(store=store)


def test_resolve_symbol_ids(populated_retriever: GraphRetriever) -> None:
    """Verify symbol ID resolution matches simple names, qualified names, or IDs."""
    # 1. Simple name
    ids = populated_retriever._resolve_symbol_ids("authenticate_user")
    assert ids == ["auth.authenticate_user"]

    # 2. Qualified name
    ids = populated_retriever._resolve_symbol_ids("db.connection.execute_query")
    assert ids == ["db.connection.execute_query"]

    # 3. Exact ID
    ids = populated_retriever._resolve_symbol_ids("web.router.handle_request")
    assert ids == ["web.router.handle_request"]

    # 4. Unknown
    assert populated_retriever._resolve_symbol_ids("nonexistent") == []


def test_get_callers(populated_retriever: GraphRetriever) -> None:
    """Verify get_callers returns correct callers at different depths."""
    # Depth 1 caller of authenticate_user should be handle_request
    callers_1 = populated_retriever.get_callers("authenticate_user", depth=1)
    assert len(callers_1) == 1
    assert isinstance(callers_1[0], GraphRetrievalResult)
    assert callers_1[0].symbol_name == "web.router.handle_request"
    assert callers_1[0].depth == 1
    assert callers_1[0].score == 1.0
    assert "Handles incoming web request" in callers_1[0].chunk_text

    # Depth 2 callers of execute_query should be authenticate_user (depth 1) and handle_request (depth 2)
    callers_2 = populated_retriever.get_callers("execute_query", depth=2)
    assert len(callers_2) == 2
    assert isinstance(callers_2[0], GraphRetrievalResult)
    assert isinstance(callers_2[1], GraphRetrievalResult)
    assert callers_2[0].symbol_name == "auth.authenticate_user"
    assert callers_2[0].depth == 1
    assert callers_2[1].symbol_name == "web.router.handle_request"
    assert callers_2[1].depth == 2
    assert callers_2[1].score == 0.5


def test_get_callees(populated_retriever: GraphRetriever) -> None:
    """Verify get_callees returns correct callees at different depths."""
    # Callees of handle_request up to depth 2:
    # authenticate_user and helper_func (depth 1), execute_query (depth 2)
    callees = populated_retriever.get_callees("handle_request", depth=2)
    assert len(callees) == 3

    # Results should be sorted by depth ascending, then name
    assert isinstance(callees[0], GraphRetrievalResult)
    assert callees[0].symbol_name == "auth.authenticate_user"
    assert callees[0].depth == 1

    assert isinstance(callees[1], GraphRetrievalResult)
    assert callees[1].symbol_name == "utils.helper_func"
    assert callees[1].depth == 1

    assert isinstance(callees[2], GraphRetrievalResult)
    assert callees[2].symbol_name == "db.connection.execute_query"
    assert callees[2].depth == 2


def test_get_neighbors(populated_retriever: GraphRetriever) -> None:
    """Verify general neighborhood queries work."""
    # Undirected neighbors of authenticate_user (both callers and callees) at depth 1
    neighbors = populated_retriever.get_neighbors(
        "authenticate_user", depth=1, direction="both"
    )
    assert len(neighbors) == 2
    for n in neighbors:
        assert isinstance(n, GraphRetrievalResult)
    names = {n.symbol_name for n in neighbors}
    assert names == {"web.router.handle_request", "db.connection.execute_query"}


def test_find_paths_shortest(populated_retriever: GraphRetriever) -> None:
    """Verify shortest path query finds the path between source and target."""
    paths = populated_retriever.find_paths(
        "handle_request", "execute_query", max_depth=5, shortest_only=True
    )
    assert len(paths) == 1
    p = paths[0]
    assert isinstance(p, GraphPathResult)
    assert p.symbols == [
        "web.router.handle_request",
        "auth.authenticate_user",
        "db.connection.execute_query",
    ]
    assert p.score == 1.0 / 3
    assert p.symbol_name == "db.connection.execute_query"


def test_find_paths_all(populated_retriever: GraphRetriever) -> None:
    """Verify all paths query finds multiple paths if they exist."""
    # Let's add an alternative path: handle_request -> helper_func -> execute_query
    store = populated_retriever.store
    assert isinstance(store, NetworkXGraphStore)

    store.graph.add_edge(
        "utils.helper_func",
        "db.connection.execute_query",
        type="CALLS",
    )

    paths = populated_retriever.find_paths(
        "handle_request", "execute_query", max_depth=5, shortest_only=False
    )
    assert len(paths) == 2
    for p in paths:
        assert isinstance(p, GraphPathResult)
    # Paths sorted by length ascending
    assert paths[0].symbols == [
        "web.router.handle_request",
        "auth.authenticate_user",
        "db.connection.execute_query",
    ]
    assert paths[1].symbols == [
        "web.router.handle_request",
        "utils.helper_func",
        "db.connection.execute_query",
    ]


def test_find_paths_depth_cutoff(populated_retriever: GraphRetriever) -> None:
    """Verify path is not returned if depth cutoff is exceeded."""
    # Path handle_request -> execute_query requires 2 hops (3 nodes)
    # If max_depth=1 hop, it shouldn't find any path
    paths = populated_retriever.find_paths(
        "handle_request", "execute_query", max_depth=1
    )
    assert len(paths) == 0


def test_get_subgraph(populated_retriever: GraphRetriever) -> None:
    """Verify induced subgraph neighborhood extraction."""
    nodes, edges = populated_retriever.get_subgraph(["authenticate_user"], depth=1)

    # Subgraph of authenticate_user at depth 1 includes: handle_request, authenticate_user, execute_query
    node_ids = {n["symbol_id"] for n in nodes}
    assert node_ids == {
        "web.router.handle_request",
        "auth.authenticate_user",
        "db.connection.execute_query",
    }

    # Should include CALLS edges connecting them
    edge_types = {e["type"] for e in edges}
    assert "CALLS" in edge_types
    assert len(edges) == 2


def test_fallback_unreachable_neo4j() -> None:
    """Verify GraphRetriever falls back to NetworkX store if Neo4j is down/unreachable."""
    # Using an invalid local port for bolt, with fallback=True
    retriever = GraphRetriever(neo4j_uri="bolt://localhost:9999", fallback=True)
    assert isinstance(retriever.store, NetworkXGraphStore)


# ---------------------------------------------------------------------------
# Validation tests (new guards introduced in hardening pass)
# ---------------------------------------------------------------------------


def test_invalid_depth_zero_raises(populated_retriever: GraphRetriever) -> None:
    """get_callers raises ValueError for depth=0."""
    with pytest.raises(ValueError, match="depth must be >= 1"):
        populated_retriever.get_callers("authenticate_user", depth=0)


def test_invalid_depth_negative_raises(populated_retriever: GraphRetriever) -> None:
    """get_callees raises ValueError for negative depth."""
    with pytest.raises(ValueError, match="depth must be >= 1"):
        populated_retriever.get_callees("authenticate_user", depth=-1)


def test_invalid_depth_too_large_raises(populated_retriever: GraphRetriever) -> None:
    """get_neighbors raises ValueError when depth exceeds the allowed cap."""
    with pytest.raises(ValueError, match="depth must be <= 10"):
        populated_retriever.get_neighbors("authenticate_user", depth=11)


def test_invalid_direction_raises(populated_retriever: GraphRetriever) -> None:
    """get_neighbors raises ValueError for an unrecognised direction string."""
    with pytest.raises(ValueError, match="direction must be one of"):
        populated_retriever.get_neighbors("authenticate_user", direction="sideways")


def test_invalid_edge_types_empty_list_raises(
    populated_retriever: GraphRetriever,
) -> None:
    """get_neighbors raises ValueError for an empty edge_types list."""
    with pytest.raises(ValueError, match="non-empty list"):
        populated_retriever.get_neighbors("authenticate_user", edge_types=[])


def test_invalid_edge_types_blank_label_raises(
    populated_retriever: GraphRetriever,
) -> None:
    """get_neighbors raises ValueError if any edge type label is blank."""
    with pytest.raises(ValueError, match="non-empty string"):
        populated_retriever.get_neighbors("authenticate_user", edge_types=["CALLS", ""])


def test_find_paths_max_depth_zero_raises(populated_retriever: GraphRetriever) -> None:
    """find_paths raises ValueError for max_depth=0."""
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        populated_retriever.find_paths("handle_request", "execute_query", max_depth=0)


def test_find_paths_max_depth_too_large_raises(
    populated_retriever: GraphRetriever,
) -> None:
    """find_paths raises ValueError when max_depth exceeds the allowed cap."""
    with pytest.raises(ValueError, match="max_depth must be <="):
        populated_retriever.find_paths("handle_request", "execute_query", max_depth=16)


def test_unknown_symbol_returns_empty(populated_retriever: GraphRetriever) -> None:
    """Queries on an unknown symbol return empty results instead of raising."""
    assert populated_retriever.get_callers("nonexistent_func") == []
    assert populated_retriever.get_callees("nonexistent_func") == []
    assert populated_retriever.get_neighbors("nonexistent_func") == []
    assert populated_retriever.find_paths("nonexistent_func", "execute_query") == []
    assert populated_retriever.get_subgraph(["nonexistent_func"]) == ([], [])


def test_get_subgraph_empty_symbols_returns_empty(
    populated_retriever: GraphRetriever,
) -> None:
    """get_subgraph returns ([], []) for an empty symbols list."""
    nodes, edges = populated_retriever.get_subgraph([])
    assert nodes == []
    assert edges == []


def test_find_paths_empty_string_source_returns_empty(
    populated_retriever: GraphRetriever,
) -> None:
    """find_paths returns empty for empty-string source."""
    assert populated_retriever.find_paths("", "execute_query") == []

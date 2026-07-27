"""Unit tests for graph-based retrieval (Issue 18)."""

from __future__ import annotations

from reporag.graph.call_graph import CallEdge
from reporag.graph.neo4j_store import NetworkXGraphStore
from reporag.graph.symbol_table import SymbolRecord, SymbolTable
from reporag.retrieval.graph_traversal import (
    CALL_EDGE,
    GraphRetrievalResult,
    GraphRetriever,
)
from reporag.retrieval.vector_search import RetrievalResult


def _record(
    symbol_id: str,
    *,
    name: str | None = None,
    file_path: str = "app.py",
    module: str = "app",
    start_line: int = 1,
    end_line: int = 5,
) -> SymbolRecord:
    return SymbolRecord(
        symbol_id=symbol_id,
        name=name or symbol_id.rsplit(".", 1)[-1],
        qualified_name=symbol_id,
        type="function",
        file_path=file_path,
        module=module,
        start_line=start_line,
        end_line=end_line,
        signature=f"def {name or symbol_id.rsplit('.', 1)[-1]}()",
    )


def _call(caller: str, callee: str) -> CallEdge:
    return CallEdge(
        caller=caller,
        callee=callee,
        caller_file="app.py",
        callee_file="app.py",
        call_site_line=10,
        call_type="function",
        resolution="local",
    )


def _store_with_topology(*, include_direct_edge: bool = False) -> NetworkXGraphStore:
    table = SymbolTable()
    for record in [
        _record(
            "app.handle_request",
            name="handle_request",
            start_line=1,
            end_line=6,
        ),
        _record(
            "app.health_check",
            name="health_check",
            start_line=8,
            end_line=12,
        ),
        _record(
            "auth.authenticate_user",
            name="authenticate_user",
            file_path="auth.py",
            module="auth",
            start_line=3,
            end_line=9,
        ),
        _record(
            "db.load_user",
            name="load_user",
            file_path="db.py",
            module="db",
            start_line=4,
            end_line=11,
        ),
    ]:
        table.add(record)

    call_edges = [
        _call("app.handle_request", "auth.authenticate_user"),
        _call("app.health_check", "auth.authenticate_user"),
        _call("auth.authenticate_user", "db.load_user"),
    ]
    if include_direct_edge:
        call_edges.append(_call("app.handle_request", "db.load_user"))

    store = NetworkXGraphStore()
    store.persist_graph(call_edges, [], table)
    return store


def test_get_callers_returns_common_results_with_depth() -> None:
    retriever = GraphRetriever(graph_store=_store_with_topology())

    callers = retriever.get_callers("authenticate_user", depth=1)

    assert {result.symbol_name for result in callers} == {
        "app.handle_request",
        "app.health_check",
    }
    assert all(isinstance(result, RetrievalResult) for result in callers)
    assert all(isinstance(result, GraphRetrievalResult) for result in callers)
    assert {result.depth for result in callers} == {1}
    assert all(result.relationship == CALL_EDGE for result in callers)


def test_get_callees_traverses_multiple_hops() -> None:
    retriever = GraphRetriever(graph_store=_store_with_topology())

    callees = retriever.get_callees("app.handle_request", depth=2)

    by_symbol = {result.symbol_name: result for result in callees}
    assert set(by_symbol) == {"auth.authenticate_user", "db.load_user"}
    assert by_symbol["auth.authenticate_user"].depth == 1
    assert by_symbol["db.load_user"].depth == 2
    assert by_symbol["db.load_user"].score < by_symbol["auth.authenticate_user"].score


def test_find_shortest_path_between_symbols() -> None:
    retriever = GraphRetriever(graph_store=_store_with_topology())

    path = retriever.find_shortest_path(
        "handle_request",
        "load_user",
        direction="out",
        edge_types=[CALL_EDGE],
    )

    assert path is not None
    assert path.symbols == [
        "app.handle_request",
        "auth.authenticate_user",
        "db.load_user",
    ]
    assert path.depth == 2
    assert [edge["type"] for edge in path.edges] == [CALL_EDGE, CALL_EDGE]


def test_find_paths_returns_bounded_paths_shortest_first() -> None:
    retriever = GraphRetriever(
        graph_store=_store_with_topology(include_direct_edge=True)
    )

    paths = retriever.find_paths(
        "app.handle_request",
        "db.load_user",
        max_depth=2,
        direction="out",
        edge_types=[CALL_EDGE],
        limit=5,
    )

    assert [path.symbols for path in paths] == [
        ["app.handle_request", "db.load_user"],
        ["app.handle_request", "auth.authenticate_user", "db.load_user"],
    ]


def test_extract_subgraph_returns_neighborhood_and_internal_edges() -> None:
    retriever = GraphRetriever(graph_store=_store_with_topology())

    subgraph = retriever.extract_subgraph(
        ["auth.authenticate_user"],
        depth=1,
        edge_types=[CALL_EDGE],
    )

    node_ids = {node["symbol_id"] for node in subgraph.nodes}
    edge_pairs = {(edge["source"], edge["target"]) for edge in subgraph.edges}
    assert node_ids == {
        "app.handle_request",
        "app.health_check",
        "auth.authenticate_user",
        "db.load_user",
    }
    assert edge_pairs == {
        ("app.handle_request", "auth.authenticate_user"),
        ("app.health_check", "auth.authenticate_user"),
        ("auth.authenticate_user", "db.load_user"),
    }
    assert all(isinstance(result, RetrievalResult) for result in subgraph.results)


def test_search_exposes_fusion_friendly_retrieval_surface() -> None:
    retriever = GraphRetriever(graph_store=_store_with_topology())

    results = retriever.search("authenticate_user", top_k=2, edge_types=[CALL_EDGE])

    assert len(results) == 2
    assert all(isinstance(result, RetrievalResult) for result in results)
    assert {result.symbol_name for result in results} <= {
        "app.handle_request",
        "app.health_check",
        "db.load_user",
    }


def test_constructor_can_start_with_networkx_fallback() -> None:
    retriever = GraphRetriever(use_settings_uri=False)

    assert isinstance(retriever.store, NetworkXGraphStore)

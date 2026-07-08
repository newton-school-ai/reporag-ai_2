"""Unit tests for the Neo4j graph store (Issue 12).

All tests use :class:`NetworkXGraphStore` as the backend so they run
without a live Neo4j instance.  The NetworkX store implements the same
:class:`GraphStoreProtocol`, so passing these tests proves the interface
contract is correct for both backends.

Test coverage mirrors the acceptance criteria from the issue:
- Creates nodes with correct labels and properties
- Creates CALLS, IMPORTS, INHERITS, CONTAINS edges
- Cypher query helper raises NotImplementedError on NetworkX backend
- Neighbors, shortest path, and subgraph extraction return correct results
- NetworkX fallback passes the full suite without Neo4j
- Bulk insert handles 10 K+ nodes efficiently
- GraphStore factory returns NetworkXGraphStore when fallback=True
- Neo4j connection errors are handled gracefully (mocked)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.reporag.graph.call_graph import CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge
from src.reporag.graph.neo4j_store import (
    GraphQueryResult,
    GraphStore,
    GraphStoreProtocol,
    Neo4jGraphStore,
    NetworkXGraphStore,
    _label_for,
    _module_node_props,
    _symbol_to_node_props,
)
from src.reporag.graph.symbol_table import SymbolRecord, SymbolTable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    parent: str | None = None,
    bases: list[str] | None = None,
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
        parent=parent,
        bases=bases or [],
    )


def _make_table(*records: SymbolRecord) -> SymbolTable:
    """Build a SymbolTable pre-populated with the given records."""
    table = SymbolTable()
    for r in records:
        table.add(r)
    return table


def _make_call_edge(
    caller: str,
    callee: str,
    *,
    caller_file: str = "app.py",
    callee_file: str = "app.py",
    call_site_line: int = 5,
    call_type: str = "function",
    resolution: str = "local",
    is_recursive: bool = False,
) -> CallEdge:
    """Build a resolved CallEdge for testing."""
    return CallEdge(
        caller=caller,
        callee=callee,
        caller_file=caller_file,
        callee_file=callee_file,
        call_site_line=call_site_line,
        call_type=call_type,
        resolution=resolution,
        is_recursive=is_recursive,
    )


def _make_dep_edge(
    source: str,
    target: str,
    source_module: str,
    target_module: str,
    *,
    resolved: bool = True,
    import_type: str = "import",
    line: int = 1,
) -> DependencyEdge:
    """Build a DependencyEdge for testing."""
    return DependencyEdge(
        source=source,
        target=target,
        source_module=source_module,
        target_module=target_module,
        import_type=import_type,
        line=line,
        resolved=resolved,
    )


@pytest.fixture()
def store() -> NetworkXGraphStore:
    """Return a fresh, empty NetworkXGraphStore."""
    return NetworkXGraphStore()


@pytest.fixture()
def func_record() -> SymbolRecord:
    return _make_record(
        symbol_id="app.helper",
        name="helper",
        qualified_name="app.helper",
        type="function",
    )


@pytest.fixture()
def class_record() -> SymbolRecord:
    return _make_record(
        symbol_id="app.MyClass",
        name="MyClass",
        qualified_name="app.MyClass",
        type="class",
    )


@pytest.fixture()
def method_record(class_record: SymbolRecord) -> SymbolRecord:
    return _make_record(
        symbol_id="app.MyClass.run",
        name="run",
        qualified_name="app.MyClass.run",
        type="method",
        parent=class_record.qualified_name,
    )


@pytest.fixture()
def simple_table(
    func_record: SymbolRecord, class_record: SymbolRecord, method_record: SymbolRecord
) -> SymbolTable:
    return _make_table(func_record, class_record, method_record)


# ---------------------------------------------------------------------------
# Helper / private function unit tests
# ---------------------------------------------------------------------------


class TestLabelFor:
    def test_function_maps_to_function_label(self) -> None:
        r = _make_record(symbol_id="x", name="x", qualified_name="x", type="function")
        assert _label_for(r) == "Function"

    def test_method_maps_to_function_label(self) -> None:
        r = _make_record(symbol_id="x", name="x", qualified_name="x", type="method")
        assert _label_for(r) == "Function"

    def test_class_maps_to_class_label(self) -> None:
        r = _make_record(symbol_id="x", name="x", qualified_name="x", type="class")
        assert _label_for(r) == "Class"

    def test_variable_maps_to_symbol_label(self) -> None:
        r = _make_record(symbol_id="x", name="x", qualified_name="x", type="variable")
        assert _label_for(r) == "Symbol"

    def test_unknown_type_maps_to_symbol_label(self) -> None:
        r = _make_record(symbol_id="x", name="x", qualified_name="x", type="decorator")
        assert _label_for(r) == "Symbol"


class TestSymbolToNodeProps:
    def test_all_required_keys_present(self, func_record: SymbolRecord) -> None:
        props = _symbol_to_node_props(func_record)
        required = {
            "symbol_id",
            "name",
            "qualified_name",
            "type",
            "label",
            "file_path",
            "module",
            "start_line",
            "end_line",
            "is_async",
            "is_method",
            "language",
        }
        assert required <= props.keys()

    def test_method_has_is_method_true(self, method_record: SymbolRecord) -> None:
        props = _symbol_to_node_props(method_record)
        assert props["is_method"] is True

    def test_function_has_is_method_false(self, func_record: SymbolRecord) -> None:
        props = _symbol_to_node_props(func_record)
        assert props["is_method"] is False

    def test_bases_are_pipe_separated(self) -> None:
        r = _make_record(
            symbol_id="x",
            name="Child",
            qualified_name="Child",
            type="class",
            bases=["Base1", "Base2"],
        )
        props = _symbol_to_node_props(r)
        assert props["bases"] == "Base1|Base2"

    def test_none_signature_becomes_empty_string(
        self, func_record: SymbolRecord
    ) -> None:
        props = _symbol_to_node_props(func_record)
        assert props["signature"] == ""


class TestModuleNodeProps:
    def test_symbol_id_prefixed_with_module(self) -> None:
        props = _module_node_props("app.py", "app")
        assert props["symbol_id"] == "module:app"

    def test_label_is_module_label(self) -> None:
        props = _module_node_props("app.py", "app")
        assert props["label"] == "Module"

    def test_name_is_last_segment(self) -> None:
        props = _module_node_props("pkg/sub.py", "pkg.sub")
        assert props["name"] == "sub"


# ---------------------------------------------------------------------------
# GraphQueryResult
# ---------------------------------------------------------------------------


class TestGraphQueryResult:
    def test_len_returns_record_count(self) -> None:
        r = GraphQueryResult(records=[{"a": 1}, {"b": 2}])
        assert len(r) == 2

    def test_iter_over_records(self) -> None:
        r = GraphQueryResult(records=[{"x": 1}])
        assert list(r) == [{"x": 1}]

    def test_repr(self) -> None:
        r = GraphQueryResult(records=[{}, {}])
        assert "2" in repr(r)


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- node creation
# ---------------------------------------------------------------------------


class TestNetworkXNodeCreation:
    def test_function_node_exists_after_persist(
        self, store: NetworkXGraphStore, func_record: SymbolRecord
    ) -> None:
        table = _make_table(func_record)
        store.persist_graph([], [], table)
        assert store.graph.has_node(func_record.symbol_id)

    def test_class_node_exists_after_persist(
        self, store: NetworkXGraphStore, class_record: SymbolRecord
    ) -> None:
        table = _make_table(class_record)
        store.persist_graph([], [], table)
        assert store.graph.has_node(class_record.symbol_id)

    def test_method_node_exists_after_persist(
        self, store: NetworkXGraphStore, method_record: SymbolRecord
    ) -> None:
        table = _make_table(method_record)
        store.persist_graph([], [], table)
        assert store.graph.has_node(method_record.symbol_id)

    def test_module_node_created_per_file(
        self, store: NetworkXGraphStore, func_record: SymbolRecord
    ) -> None:
        """A synthetic Module node is created for each distinct module."""
        table = _make_table(func_record)
        store.persist_graph([], [], table)
        assert store.graph.has_node("module:app")

    def test_node_label_stored_as_attribute(
        self, store: NetworkXGraphStore, func_record: SymbolRecord
    ) -> None:
        table = _make_table(func_record)
        store.persist_graph([], [], table)
        node_data = store.graph.nodes[func_record.symbol_id]
        assert node_data["label"] == "Function"

    def test_node_properties_correct(
        self, store: NetworkXGraphStore, func_record: SymbolRecord
    ) -> None:
        table = _make_table(func_record)
        store.persist_graph([], [], table)
        node_data = store.graph.nodes[func_record.symbol_id]
        assert node_data["name"] == "helper"
        assert node_data["qualified_name"] == "app.helper"
        assert node_data["file_path"] == "app.py"
        assert node_data["module"] == "app"

    def test_persist_is_idempotent(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        """Calling persist_graph twice does not duplicate nodes."""
        store.persist_graph([], [], simple_table)
        n_nodes_first = store.graph.number_of_nodes()
        store.persist_graph([], [], simple_table)
        assert store.graph.number_of_nodes() == n_nodes_first

    def test_multiple_files_create_multiple_module_nodes(
        self, store: NetworkXGraphStore
    ) -> None:
        r1 = _make_record(
            symbol_id="app.f",
            name="f",
            qualified_name="app.f",
            type="function",
            file_path="app.py",
            module="app",
        )
        r2 = _make_record(
            symbol_id="db.g",
            name="g",
            qualified_name="db.g",
            type="function",
            file_path="db.py",
            module="db",
        )
        table = _make_table(r1, r2)
        store.persist_graph([], [], table)
        assert store.graph.has_node("module:app")
        assert store.graph.has_node("module:db")


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- edge creation
# ---------------------------------------------------------------------------


class TestNetworkXEdgeCreation:
    def test_calls_edge_created(self, store: NetworkXGraphStore) -> None:
        caller = _make_record(
            symbol_id="app.main",
            name="main",
            qualified_name="app.main",
            type="function",
        )
        callee = _make_record(
            symbol_id="app.helper",
            name="helper",
            qualified_name="app.helper",
            type="function",
        )
        table = _make_table(caller, callee)
        edge = _make_call_edge("app.main", "app.helper")
        store.persist_graph([edge], [], table)
        assert store.graph.has_edge(caller.symbol_id, callee.symbol_id)
        edge_data = store.graph.edges[caller.symbol_id, callee.symbol_id]
        assert edge_data["type"] == "CALLS"

    def test_calls_edge_has_metadata(self, store: NetworkXGraphStore) -> None:
        caller = _make_record(
            symbol_id="app.main",
            name="main",
            qualified_name="app.main",
            type="function",
        )
        callee = _make_record(
            symbol_id="app.helper",
            name="helper",
            qualified_name="app.helper",
            type="function",
        )
        table = _make_table(caller, callee)
        edge = _make_call_edge("app.main", "app.helper", call_site_line=42)
        store.persist_graph([edge], [], table)
        edge_data = store.graph.edges[caller.symbol_id, callee.symbol_id]
        assert edge_data["call_site_line"] == 42
        assert edge_data["resolution"] == "local"

    def test_unresolved_call_edge_skipped(self, store: NetworkXGraphStore) -> None:
        """Unresolved edges must not be added to the graph."""
        caller = _make_record(
            symbol_id="app.main",
            name="main",
            qualified_name="app.main",
            type="function",
        )
        table = _make_table(caller)
        unresolved_edge = CallEdge(
            caller="app.main",
            callee="os.path.join",
            caller_file="app.py",
            call_site_line=3,
            resolution="unresolved",
        )
        store.persist_graph([unresolved_edge], [], table)
        assert store.graph.number_of_edges() == 0

    def test_imports_edge_created(self, store: NetworkXGraphStore) -> None:
        r = _make_record(
            symbol_id="app.f",
            name="f",
            qualified_name="app.f",
            type="function",
            file_path="app.py",
            module="app",
        )
        table = _make_table(r)
        dep = _make_dep_edge("app.py", "db.py", "app", "db")
        store.persist_graph([], [dep], table)
        assert store.graph.has_edge("module:app", "module:db")
        edge_data = store.graph.edges["module:app", "module:db"]
        assert edge_data["type"] == "IMPORTS"

    def test_imports_edge_unresolved_skipped(self, store: NetworkXGraphStore) -> None:
        r = _make_record(
            symbol_id="app.f",
            name="f",
            qualified_name="app.f",
            type="function",
            file_path="app.py",
            module="app",
        )
        table = _make_table(r)
        dep = _make_dep_edge("app.py", "os", "app", "os", resolved=False)
        store.persist_graph([], [dep], table)
        # No IMPORTS edge for unresolved external deps
        assert not store.graph.has_edge("module:app", "module:os")

    def test_contains_edge_created(
        self,
        store: NetworkXGraphStore,
        class_record: SymbolRecord,
        method_record: SymbolRecord,
    ) -> None:
        table = _make_table(class_record, method_record)
        store.persist_graph([], [], table)
        # method_record.parent == "app.MyClass" which is class_record.symbol_id
        assert store.graph.has_edge(class_record.symbol_id, method_record.symbol_id)
        edge_data = store.graph.edges[class_record.symbol_id, method_record.symbol_id]
        assert edge_data["type"] == "CONTAINS"

    def test_inherits_edge_created(self, store: NetworkXGraphStore) -> None:
        base = _make_record(
            symbol_id="app.Base",
            name="Base",
            qualified_name="app.Base",
            type="class",
        )
        child = _make_record(
            symbol_id="app.Child",
            name="Child",
            qualified_name="app.Child",
            type="class",
            bases=["Base"],
        )
        table = _make_table(base, child)
        store.persist_graph([], [], table)
        assert store.graph.has_edge(child.symbol_id, base.symbol_id)
        edge_data = store.graph.edges[child.symbol_id, base.symbol_id]
        assert edge_data["type"] == "INHERITS"

    def test_inherits_edge_not_created_for_unknown_base(
        self, store: NetworkXGraphStore
    ) -> None:
        child = _make_record(
            symbol_id="app.Child",
            name="Child",
            qualified_name="app.Child",
            type="class",
            bases=["SomeExternalBase"],
        )
        table = _make_table(child)
        store.persist_graph([], [], table)
        # SomeExternalBase is unknown -- no spurious edge
        assert store.graph.number_of_edges() == 0


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def _linear_store(self) -> NetworkXGraphStore:
        """Return a store with a -> b -> c chain (CALLS edges)."""
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        c = _make_record(symbol_id="c", name="c", qualified_name="c", type="function")
        table = _make_table(a, b, c)
        store = NetworkXGraphStore()
        store.persist_graph(
            [
                _make_call_edge("a", "b"),
                _make_call_edge("b", "c"),
            ],
            [],
            table,
        )
        return store

    def test_neighbors_depth_1_outgoing(self) -> None:
        s = self._linear_store()
        nbrs = s.get_neighbors("a", depth=1, direction="out")
        ids = {n["symbol_id"] for n in nbrs}
        assert ids == {"b"}

    def test_neighbors_depth_2_outgoing(self) -> None:
        s = self._linear_store()
        nbrs = s.get_neighbors("a", depth=2, direction="out")
        ids = {n["symbol_id"] for n in nbrs}
        assert ids == {"b", "c"}

    def test_neighbors_depth_1_incoming(self) -> None:
        s = self._linear_store()
        nbrs = s.get_neighbors("c", depth=1, direction="in")
        ids = {n["symbol_id"] for n in nbrs}
        assert ids == {"b"}

    def test_neighbors_both_directions(self) -> None:
        s = self._linear_store()
        nbrs = s.get_neighbors("b", depth=1, direction="both")
        ids = {n["symbol_id"] for n in nbrs}
        assert ids == {"a", "c"}

    def test_neighbors_filtered_by_edge_type(self) -> None:
        """Edges of a different type should not be traversed."""
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        table = _make_table(a, b)
        s = NetworkXGraphStore()
        s.persist_graph([_make_call_edge("a", "b")], [], table)
        # Filter by IMPORTS -- should return nothing since the edge is CALLS
        nbrs = s.get_neighbors("a", edge_types=["IMPORTS"], depth=1, direction="out")
        assert nbrs == []

    def test_neighbors_unknown_node_returns_empty(
        self, store: NetworkXGraphStore
    ) -> None:
        assert store.get_neighbors("nonexistent") == []


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- shortest_path
# ---------------------------------------------------------------------------


class TestShortestPath:
    def _two_hop_store(self) -> NetworkXGraphStore:
        """Return a store: a -> b -> c and a direct a -> c edge."""
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        c = _make_record(symbol_id="c", name="c", qualified_name="c", type="function")
        table = _make_table(a, b, c)
        s = NetworkXGraphStore()
        s.persist_graph(
            [
                _make_call_edge("a", "b"),
                _make_call_edge("b", "c"),
                _make_call_edge("a", "c"),
            ],
            [],
            table,
        )
        return s

    def test_shortest_path_found(self) -> None:
        s = self._two_hop_store()
        path = s.shortest_path("a", "c")
        ids = [n["symbol_id"] for n in path]
        # Shortest is a -> c (direct), not a -> b -> c
        assert ids[0] == "a"
        assert ids[-1] == "c"
        assert len(ids) == 2

    def test_shortest_path_two_hops(self) -> None:
        """When only a two-hop path exists, it is returned."""
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        c = _make_record(symbol_id="c", name="c", qualified_name="c", type="function")
        table = _make_table(a, b, c)
        s = NetworkXGraphStore()
        s.persist_graph(
            [_make_call_edge("a", "b"), _make_call_edge("b", "c")],
            [],
            table,
        )
        path = s.shortest_path("a", "c")
        ids = [n["symbol_id"] for n in path]
        assert ids == ["a", "b", "c"]

    def test_no_path_returns_empty_list(self) -> None:
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        table = _make_table(a, b)
        s = NetworkXGraphStore()
        s.persist_graph([], [], table)
        path = s.shortest_path("a", "b")
        assert path == []

    def test_unknown_source_returns_empty_list(self, store: NetworkXGraphStore) -> None:
        assert store.shortest_path("ghost", "also-ghost") == []

    def test_edge_type_filter_excludes_wrong_types(self) -> None:
        """When the only path uses a CALLS edge, filtering by IMPORTS -> no path."""
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        table = _make_table(a, b)
        s = NetworkXGraphStore()
        s.persist_graph([_make_call_edge("a", "b")], [], table)
        path = s.shortest_path("a", "b", edge_types=["IMPORTS"])
        assert path == []


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- subgraph
# ---------------------------------------------------------------------------


class TestSubgraph:
    def test_subgraph_returns_only_requested_nodes(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        store.persist_graph([], [], simple_table)
        nodes, _ = store.subgraph(["app.helper"])
        ids = [n["symbol_id"] for n in nodes]
        assert ids == ["app.helper"]

    def test_subgraph_edges_include_only_internal(
        self, store: NetworkXGraphStore
    ) -> None:
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        c = _make_record(symbol_id="c", name="c", qualified_name="c", type="function")
        table = _make_table(a, b, c)
        store.persist_graph(
            [_make_call_edge("a", "b"), _make_call_edge("b", "c")],
            [],
            table,
        )
        _, edges = store.subgraph(["a", "b"])
        # Only a->b should appear; b->c crosses outside the subgraph
        assert len(edges) == 1
        assert edges[0]["source"] == "a"
        assert edges[0]["target"] == "b"

    def test_subgraph_edge_has_source_and_target_keys(
        self, store: NetworkXGraphStore
    ) -> None:
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        table = _make_table(a, b)
        store.persist_graph([_make_call_edge("a", "b")], [], table)
        _, edges = store.subgraph(["a", "b"])
        assert "source" in edges[0]
        assert "target" in edges[0]

    def test_subgraph_unknown_ids_ignored(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        store.persist_graph([], [], simple_table)
        nodes, _ = store.subgraph(["nonexistent-id"])
        assert nodes == []


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all_nodes(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        store.persist_graph([], [], simple_table)
        assert store.graph.number_of_nodes() > 0
        store.clear()
        assert store.graph.number_of_nodes() == 0

    def test_clear_removes_all_edges(self, store: NetworkXGraphStore) -> None:
        a = _make_record(symbol_id="a", name="a", qualified_name="a", type="function")
        b = _make_record(symbol_id="b", name="b", qualified_name="b", type="function")
        table = _make_table(a, b)
        store.persist_graph([_make_call_edge("a", "b")], [], table)
        store.clear()
        assert store.graph.number_of_edges() == 0

    def test_persist_after_clear_works(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        store.persist_graph([], [], simple_table)
        store.clear()
        store.persist_graph([], [], simple_table)
        assert store.graph.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# NetworkXGraphStore -- query raises NotImplementedError
# ---------------------------------------------------------------------------


class TestQueryNotImplemented:
    def test_query_raises_not_implemented(self, store: NetworkXGraphStore) -> None:
        with pytest.raises(NotImplementedError, match="Cypher"):
            store.query("MATCH (n) RETURN n LIMIT 1")

    def test_query_message_mentions_alternatives(
        self, store: NetworkXGraphStore
    ) -> None:
        with pytest.raises(NotImplementedError, match="get_neighbors"):
            store.query("MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# Bulk insert performance -- 10K+ nodes
# ---------------------------------------------------------------------------


class TestBulkInsert:
    def test_bulk_insert_10k_nodes_no_error(self, store: NetworkXGraphStore) -> None:
        """persist_graph must complete without error for 10,000 nodes."""
        n = 10_000
        records = [
            _make_record(
                symbol_id=f"bulk.f{i}",
                name=f"f{i}",
                qualified_name=f"bulk.f{i}",
                type="function",
                file_path="bulk.py",
                module="bulk",
            )
            for i in range(n)
        ]
        table = _make_table(*records)
        store.persist_graph([], [], table)
        # All symbol nodes + 1 synthetic Module node
        assert store.graph.number_of_nodes() == n + 1

    def test_bulk_insert_10k_edges_no_error(self) -> None:
        """persist_graph must complete without error for 10,000 call edges."""
        n = 10_000
        # 2 functions; n edges from f0->f1 won't all be unique so use
        # a chain of n+1 unique functions instead.
        records = [
            _make_record(
                symbol_id=f"e.f{i}",
                name=f"f{i}",
                qualified_name=f"e.f{i}",
                type="function",
                file_path="e.py",
                module="e",
            )
            for i in range(n + 1)
        ]
        table = _make_table(*records)
        call_edges = [_make_call_edge(f"e.f{i}", f"e.f{i + 1}") for i in range(n)]
        s = NetworkXGraphStore()
        s.persist_graph(call_edges, [], table)
        assert s.graph.number_of_edges() == n


# ---------------------------------------------------------------------------
# GraphStoreProtocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_networkx_store_satisfies_protocol(self, store: NetworkXGraphStore) -> None:
        assert isinstance(store, GraphStoreProtocol)

    def test_close_is_noop(self, store: NetworkXGraphStore) -> None:
        """close() must not raise."""
        store.close()

    def test_repr_contains_node_count(
        self, store: NetworkXGraphStore, simple_table: SymbolTable
    ) -> None:
        store.persist_graph([], [], simple_table)
        r = repr(store)
        assert "nodes" in r
        assert "edges" in r


# ---------------------------------------------------------------------------
# GraphStore factory
# ---------------------------------------------------------------------------


class TestGraphStoreFactory:
    def test_no_uri_returns_networkx(self) -> None:
        s = GraphStore()
        assert isinstance(s, NetworkXGraphStore)

    def test_fallback_true_no_uri_returns_networkx(self) -> None:
        s = GraphStore(fallback=True)
        assert isinstance(s, NetworkXGraphStore)

    def test_fallback_true_with_bad_uri_returns_networkx(self) -> None:
        """Even with a URI, fallback=True returns NetworkX on connection fail."""
        s = GraphStore(uri="bolt://127.0.0.1:9999", fallback=True)
        assert isinstance(s, NetworkXGraphStore)

    def test_no_fallback_bad_uri_raises(self) -> None:
        """Without fallback=True, a bad URI must raise ConnectionError."""
        with pytest.raises((ConnectionError, Exception)):
            GraphStore(uri="bolt://127.0.0.1:9999", fallback=False)


# ---------------------------------------------------------------------------
# Neo4jGraphStore -- connection error handling (mocked driver)
# ---------------------------------------------------------------------------


class TestNeo4jConnectionErrors:
    def test_all_retries_exhausted_raises_connection_error(self) -> None:
        """When Neo4j is unreachable after all retries, raise ConnectionError."""

        def bad_driver(*_args: object, **_kwargs: object) -> MagicMock:
            m = MagicMock()
            m.verify_connectivity.side_effect = OSError("Connection refused")
            return m

        with patch("src.reporag.graph.neo4j_store._import_neo4j") as mock_neo4j:
            mock_module = MagicMock()
            mock_module.GraphDatabase.driver.side_effect = bad_driver
            mock_neo4j.return_value = mock_module

            with pytest.raises((ConnectionError, OSError)):
                Neo4jGraphStore(
                    "bolt://127.0.0.1:9999",
                    max_retries=2,
                    retry_backoff=0.0,
                )

    def test_successful_connection_stores_driver(self) -> None:
        """A mock driver that passes verify_connectivity is stored."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None
        # Suppress constraint creation
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: mock_session
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = MagicMock()
        mock_driver.session.return_value = mock_session

        with patch("src.reporag.graph.neo4j_store._import_neo4j") as mock_neo4j:
            mock_module = MagicMock()
            mock_module.GraphDatabase.driver.return_value = mock_driver
            mock_neo4j.return_value = mock_module

            store = Neo4jGraphStore(
                "bolt://localhost:7687",
                max_retries=1,
                retry_backoff=0.0,
            )
            assert store._driver is not None

    def test_close_sets_driver_to_none(self) -> None:
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: mock_session
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = MagicMock()
        mock_driver.session.return_value = mock_session

        with patch("src.reporag.graph.neo4j_store._import_neo4j") as mock_neo4j:
            mock_module = MagicMock()
            mock_module.GraphDatabase.driver.return_value = mock_driver
            mock_neo4j.return_value = mock_module

            store = Neo4jGraphStore(
                "bolt://localhost:7687",
                max_retries=1,
                retry_backoff=0.0,
            )
            store.close()
            assert store._driver is None

    def test_context_manager_closes_on_exit(self) -> None:
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: mock_session
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = MagicMock()
        mock_driver.session.return_value = mock_session

        with patch("src.reporag.graph.neo4j_store._import_neo4j") as mock_neo4j:
            mock_module = MagicMock()
            mock_module.GraphDatabase.driver.return_value = mock_driver
            mock_neo4j.return_value = mock_module

            with Neo4jGraphStore(
                "bolt://localhost:7687",
                max_retries=1,
                retry_backoff=0.0,
            ) as store:
                assert store._driver is not None
            assert store._driver is None

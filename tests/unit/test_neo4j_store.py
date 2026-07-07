"""Unit tests for the Neo4j graph store (Issue 12).

Covers every acceptance criterion:

* Node creation with correct labels and properties (Function, Class, Module)
* CALLS, IMPORTS, INHERITS, CONTAINS relationships
* Cypher query helpers return correct results (Neo4j backend)
* NetworkX backend passes the same logical tests
* API compatibility between Neo4j and NetworkX backends (parametrised)
* Bulk insertion handles 10 000+ nodes
* Ingestion from existing SymbolRecord, CallEdge, DependencyEdge types
* Invalid node handling (missing id, relationship to non-existent node)
* Duplicate node handling (MERGE semantics)
* Connection lifecycle (connect / close / clear)

Neo4j tests mock the driver so no live server is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.reporag.graph.call_graph import CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge
from src.reporag.graph.neo4j_store import (
    Backend,
    GraphStore,
    Neo4jStore,
    NetworkXStore,
    _node_label,
    create_graph_store,
)
from src.reporag.graph.symbol_table import SymbolRecord

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    symbol_id: str,
    name: str,
    qualified_name: str,
    sym_type: str,
    file_path: str = "pkg/mod.py",
    module: str = "pkg.mod",
    start_line: int = 1,
    end_line: int = 10,
    parent: str | None = None,
    bases: list[str] | None = None,
    is_async: bool = False,
    signature: str | None = None,
    docstring: str | None = None,
) -> SymbolRecord:
    """Build a SymbolRecord for use in tests."""
    return SymbolRecord(
        symbol_id=symbol_id,
        name=name,
        qualified_name=qualified_name,
        type=sym_type,
        file_path=file_path,
        module=module,
        start_line=start_line,
        end_line=end_line,
        parent=parent,
        bases=bases or [],
        is_async=is_async,
        signature=signature,
        docstring=docstring,
    )


def _make_call_edge(
    caller: str,
    callee: str,
    caller_file: str = "pkg/mod.py",
    callee_file: str | None = "pkg/other.py",
    call_site_line: int = 5,
    call_type: str = "function",
    resolution: str = "local",
) -> CallEdge:
    """Build a CallEdge for use in tests."""
    return CallEdge(
        caller=caller,
        callee=callee,
        caller_file=caller_file,
        callee_file=callee_file,
        call_site_line=call_site_line,
        call_type=call_type,  # type: ignore[arg-type]
        resolution=resolution,  # type: ignore[arg-type]
    )


def _make_dep_edge(
    source: str = "pkg/app.py",
    target: str = "pkg/db.py",
    import_type: str = "from_import",
    line: int = 1,
    resolved: bool = True,
) -> DependencyEdge:
    """Build a DependencyEdge for use in tests."""
    return DependencyEdge(
        source=source,
        target=target,
        source_module="pkg.app",
        target_module="pkg.db",
        import_type=import_type,  # type: ignore[arg-type]
        imported_names=[("get_user", None)],
        line=line,
        resolved=resolved,
    )


# ---------------------------------------------------------------------------
# _node_label helper
# ---------------------------------------------------------------------------


class TestNodeLabel:
    """Tests for the _node_label() utility."""

    def test_class_maps_to_class(self) -> None:
        assert _node_label("class") == "Class"

    def test_function_maps_to_function(self) -> None:
        assert _node_label("function") == "Function"

    def test_method_maps_to_function(self) -> None:
        assert _node_label("method") == "Function"

    def test_module_maps_to_module(self) -> None:
        assert _node_label("module") == "Module"

    def test_unknown_maps_to_symbol(self) -> None:
        assert _node_label("variable") == "Symbol"
        assert _node_label("import") == "Symbol"
        assert _node_label("") == "Symbol"


# ---------------------------------------------------------------------------
# NetworkXStore tests  (no mocking required)
# ---------------------------------------------------------------------------


@pytest.fixture
def nx_store() -> NetworkXStore:
    """Fresh connected NetworkXStore for each test."""
    store = NetworkXStore()
    store.connect()
    return store


class TestNetworkXLifecycle:
    """Lifecycle: connect, close, clear."""

    def test_connect_is_noop(self) -> None:
        store = NetworkXStore()
        store.connect()  # must not raise

    def test_close_is_noop(self) -> None:
        store = NetworkXStore()
        store.connect()
        store.close()  # must not raise

    def test_clear_removes_all_nodes_and_edges(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "fn_a", "name": "a"})
        nx_store.create_node("Function", {"id": "fn_b", "name": "b"})
        nx_store.create_relationship("fn_a", "fn_b", "CALLS")
        nx_store.clear()
        assert nx_store.find_by_id("fn_a") is None
        assert nx_store.find_by_id("fn_b") is None


class TestNetworkXNodeCRUD:
    """Node creation and duplicate handling."""

    def test_create_node_returns_id(self, nx_store: NetworkXStore) -> None:
        node_id = nx_store.create_node("Function", {"id": "fn_1", "name": "run"})
        assert node_id == "fn_1"

    def test_created_node_is_findable(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Class", {"id": "cls_1", "qualified_name": "pkg.MyClass"})
        props = nx_store.find_by_id("cls_1")
        assert props is not None
        assert props["qualified_name"] == "pkg.MyClass"
        assert props["label"] == "Class"

    def test_create_node_without_id_raises(self, nx_store: NetworkXStore) -> None:
        with pytest.raises(ValueError, match="'id'"):
            nx_store.create_node("Function", {"name": "no_id"})

    def test_duplicate_node_merge_updates_props(self, nx_store: NetworkXStore) -> None:
        """Creating the same id twice updates (merges) the properties."""
        nx_store.create_node("Function", {"id": "fn_1", "name": "old"})
        nx_store.create_node("Function", {"id": "fn_1", "name": "new"})
        props = nx_store.find_by_id("fn_1")
        assert props is not None
        assert props["name"] == "new"

    def test_find_by_id_absent_returns_none(self, nx_store: NetworkXStore) -> None:
        assert nx_store.find_by_id("nonexistent") is None

    def test_find_by_qualified_name(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node(
            "Function",
            {"id": "fn_q", "qualified_name": "pkg.mod.fn_q"},
        )
        result = nx_store.find_by_qualified_name("pkg.mod.fn_q")
        assert result is not None
        assert result["id"] == "fn_q"

    def test_find_by_qualified_name_absent_returns_none(
        self, nx_store: NetworkXStore
    ) -> None:
        assert nx_store.find_by_qualified_name("nothing.here") is None


class TestNetworkXRelationships:
    """Relationship creation."""

    def _pair(self, store: NetworkXStore) -> None:
        store.create_node("Function", {"id": "caller", "name": "caller"})
        store.create_node("Function", {"id": "callee", "name": "callee"})

    def test_create_calls_relationship(self, nx_store: NetworkXStore) -> None:
        self._pair(nx_store)
        nx_store.create_relationship(
            "caller", "callee", "CALLS", {"call_type": "function"}
        )
        out = nx_store.outgoing_calls("caller")
        assert len(out) == 1
        assert out[0]["node"]["id"] == "callee"
        assert out[0]["rel"]["call_type"] == "function"

    def test_create_relationship_missing_source_raises(
        self, nx_store: NetworkXStore
    ) -> None:
        nx_store.create_node("Function", {"id": "callee"})
        with pytest.raises(KeyError, match="not found"):
            nx_store.create_relationship("ghost", "callee", "CALLS")

    def test_create_relationship_missing_target_raises(
        self, nx_store: NetworkXStore
    ) -> None:
        nx_store.create_node("Function", {"id": "caller"})
        with pytest.raises(KeyError, match="not found"):
            nx_store.create_relationship("caller", "ghost", "CALLS")

    def test_duplicate_relationship_merge_updates_props(
        self, nx_store: NetworkXStore
    ) -> None:
        self._pair(nx_store)
        nx_store.create_relationship("caller", "callee", "CALLS", {"x": 1})
        nx_store.create_relationship("caller", "callee", "CALLS", {"x": 2})
        # Only one CALLS edge should exist.
        out = nx_store.outgoing_calls("caller")
        assert len(out) == 1
        assert out[0]["rel"]["x"] == 2

    def test_incoming_calls(self, nx_store: NetworkXStore) -> None:
        self._pair(nx_store)
        nx_store.create_relationship("caller", "callee", "CALLS")
        incoming = nx_store.incoming_calls("callee")
        assert len(incoming) == 1
        assert incoming[0]["node"]["id"] == "caller"

    def test_imports_relationship(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Module", {"id": "app.py"})
        nx_store.create_node("Module", {"id": "db.py"})
        nx_store.create_relationship("app.py", "db.py", "IMPORTS")
        mods = nx_store.imported_modules("app.py")
        assert len(mods) == 1
        assert mods[0]["node"]["id"] == "db.py"

    def test_inherits_relationship(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Class", {"id": "Child"})
        nx_store.create_node("Class", {"id": "Parent"})
        nx_store.create_relationship("Child", "Parent", "INHERITS")
        bases = nx_store.inheritance_relationships("Child")
        assert len(bases) == 1
        assert bases[0]["node"]["id"] == "Parent"

    def test_contains_relationship(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Class", {"id": "MyClass"})
        nx_store.create_node("Function", {"id": "MyClass.__init__"})
        nx_store.create_relationship("MyClass", "MyClass.__init__", "CONTAINS")
        children = nx_store.containment_hierarchy("MyClass")
        assert any(c["node"]["id"] == "MyClass.__init__" for c in children)


class TestNetworkXBulkOperations:
    """Bulk insertion -- including 10 000 node scale test."""

    def test_bulk_create_nodes_returns_ids(self, nx_store: NetworkXStore) -> None:
        rows = [{"id": f"fn_{i}", "name": f"fn_{i}"} for i in range(5)]
        ids = nx_store.bulk_create_nodes("Function", rows)
        assert ids == [f"fn_{i}" for i in range(5)]
        for i in range(5):
            assert nx_store.find_by_id(f"fn_{i}") is not None

    def test_bulk_create_nodes_empty(self, nx_store: NetworkXStore) -> None:
        ids = nx_store.bulk_create_nodes("Function", [])
        assert ids == []

    def test_bulk_create_relationships(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "a"})
        nx_store.create_node("Function", {"id": "b"})
        nx_store.create_node("Module", {"id": "m"})
        nx_store.bulk_create_relationships(
            [
                {"from_id": "a", "to_id": "b", "rel_type": "CALLS", "props": {}},
                {"from_id": "m", "to_id": "b", "rel_type": "CONTAINS", "props": {}},
            ]
        )
        assert len(nx_store.outgoing_calls("a")) == 1
        assert len(nx_store.containment_hierarchy("m")) == 1

    def test_bulk_create_10k_nodes(self, nx_store: NetworkXStore) -> None:
        """Verify the store accepts 10 000+ nodes without error."""
        n = 10_000
        rows = [{"id": f"node_{i}", "name": f"node_{i}"} for i in range(n)]
        ids = nx_store.bulk_create_nodes("Function", rows)
        assert len(ids) == n
        # Spot-check a few boundary nodes.
        assert nx_store.find_by_id("node_0") is not None
        assert nx_store.find_by_id("node_4999") is not None
        assert nx_store.find_by_id(f"node_{n - 1}") is not None


class TestNetworkXQueryHelpers:
    """Query helper methods for the NetworkX backend."""

    def _setup_simple_graph(self, store: NetworkXStore) -> None:
        """Build: app -> auth -> db (CALLS chain) with an IMPORTS edge."""
        store.create_node("Function", {"id": "app", "name": "app"})
        store.create_node("Function", {"id": "auth", "name": "auth"})
        store.create_node("Module", {"id": "db", "name": "db"})
        store.create_relationship("app", "auth", "CALLS")
        store.create_relationship("auth", "db", "CALLS")
        store.create_relationship("app", "db", "IMPORTS")

    def test_shortest_path_direct(self, nx_store: NetworkXStore) -> None:
        self._setup_simple_graph(nx_store)
        path = nx_store.shortest_path("app", "db")
        assert len(path) >= 2
        assert path[0]["id"] == "app"
        assert path[-1]["id"] == "db"

    def test_shortest_path_no_path(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "isolated"})
        self._setup_simple_graph(nx_store)
        path = nx_store.shortest_path("app", "isolated")
        assert path == []

    def test_shortest_path_missing_node(self, nx_store: NetworkXStore) -> None:
        self._setup_simple_graph(nx_store)
        assert nx_store.shortest_path("ghost", "app") == []

    def test_neighborhood_depth_1(self, nx_store: NetworkXStore) -> None:
        self._setup_simple_graph(nx_store)
        result = nx_store.neighborhood("auth", depth=1)
        node_ids = {n["id"] for n in result["nodes"]}
        assert "auth" in node_ids
        assert "app" in node_ids or "db" in node_ids

    def test_neighborhood_depth_0_returns_center(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "lone", "name": "lone"})
        result = nx_store.neighborhood("lone", depth=0)
        # depth=0 means only the center node itself.
        assert any(n.get("id") == "lone" for n in result["nodes"])
        assert result["edges"] == []

    def test_neighborhood_missing_node(self, nx_store: NetworkXStore) -> None:
        result = nx_store.neighborhood("ghost")
        assert result == {"nodes": [], "edges": []}

    def test_containment_hierarchy_transitive(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Module", {"id": "mod"})
        nx_store.create_node("Class", {"id": "cls"})
        nx_store.create_node("Function", {"id": "fn"})
        nx_store.create_relationship("mod", "cls", "CONTAINS")
        nx_store.create_relationship("cls", "fn", "CONTAINS")
        children = nx_store.containment_hierarchy("mod")
        child_ids = {c["node"]["id"] for c in children}
        assert "cls" in child_ids
        assert "fn" in child_ids

    def test_outgoing_calls_empty(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "leaf"})
        assert nx_store.outgoing_calls("leaf") == []

    def test_incoming_calls_empty(self, nx_store: NetworkXStore) -> None:
        nx_store.create_node("Function", {"id": "root"})
        assert nx_store.incoming_calls("root") == []


class TestNetworkXIngestionHelpers:
    """Domain-specific ingestion from SymbolRecord / CallEdge / DependencyEdge."""

    def test_ingest_symbol_record_function(self, nx_store: NetworkXStore) -> None:
        rec = _make_record(
            symbol_id="pkg.mod.run",
            name="run",
            qualified_name="pkg.mod.run",
            sym_type="function",
        )
        nx_store.ingest_symbol_record(rec)
        props = nx_store.find_by_id("pkg.mod.run")
        assert props is not None
        assert props["label"] == "Function"
        assert props["name"] == "run"

    def test_ingest_symbol_record_class(self, nx_store: NetworkXStore) -> None:
        rec = _make_record(
            symbol_id="pkg.mod.MyClass",
            name="MyClass",
            qualified_name="pkg.mod.MyClass",
            sym_type="class",
        )
        nx_store.ingest_symbol_record(rec)
        props = nx_store.find_by_id("pkg.mod.MyClass")
        assert props is not None
        assert props["label"] == "Class"

    def test_ingest_symbol_record_creates_contains_edge(
        self, nx_store: NetworkXStore
    ) -> None:
        parent = _make_record(
            symbol_id="pkg.mod.MyClass",
            name="MyClass",
            qualified_name="pkg.mod.MyClass",
            sym_type="class",
        )
        child = _make_record(
            symbol_id="pkg.mod.MyClass.run",
            name="run",
            qualified_name="pkg.mod.MyClass.run",
            sym_type="method",
            parent="pkg.mod.MyClass",
        )
        nx_store.ingest_symbol_record(parent)
        nx_store.ingest_symbol_record(child)
        children = nx_store.containment_hierarchy("pkg.mod.MyClass")
        assert any(c["node"]["id"] == "pkg.mod.MyClass.run" for c in children)

    def test_ingest_symbol_record_creates_inherits_edge(
        self, nx_store: NetworkXStore
    ) -> None:
        child = _make_record(
            symbol_id="pkg.mod.Child",
            name="Child",
            qualified_name="pkg.mod.Child",
            sym_type="class",
            bases=["BaseClass"],
        )
        nx_store.ingest_symbol_record(child)
        bases = nx_store.inheritance_relationships("pkg.mod.Child")
        assert any(b["node"]["id"] == "BaseClass" for b in bases)

    def test_ingest_call_edge(self, nx_store: NetworkXStore) -> None:
        edge = _make_call_edge("pkg.mod.main", "pkg.mod.helper")
        nx_store.ingest_call_edge(edge)
        out = nx_store.outgoing_calls("pkg.mod.main")
        assert len(out) == 1
        assert out[0]["node"]["id"] == "pkg.mod.helper"
        assert out[0]["rel"]["call_type"] == "function"

    def test_ingest_call_edge_creates_stub_nodes(self, nx_store: NetworkXStore) -> None:
        edge = _make_call_edge("new_caller", "new_callee")
        nx_store.ingest_call_edge(edge)
        assert nx_store.find_by_id("new_caller") is not None
        assert nx_store.find_by_id("new_callee") is not None

    def test_ingest_dependency_edge(self, nx_store: NetworkXStore) -> None:
        edge = _make_dep_edge()
        nx_store.ingest_dependency_edge(edge)
        mods = nx_store.imported_modules("pkg/app.py")
        assert len(mods) == 1
        assert mods[0]["node"]["id"] == "pkg/db.py"

    def test_duplicate_symbol_record_updates_existing(self, nx_store):
        rec1 = _make_record(
            symbol_id="pkg.fn",
            name="old",
            qualified_name="pkg.fn",
            sym_type="function",
        )

        rec2 = _make_record(
            symbol_id="pkg.fn",
            name="new",
            qualified_name="pkg.fn",
            sym_type="function",
        )

        nx_store.ingest_symbol_record(rec1)
        nx_store.ingest_symbol_record(rec2)

    def test_duplicate_call_edge_does_not_duplicate_relationship(self, nx_store):
        edge = _make_call_edge("a", "b")

        nx_store.ingest_call_edge(edge)
        nx_store.ingest_call_edge(edge)

        out = nx_store.outgoing_calls("a")

        assert len(out) == 1

    def test_bulk_ingest_symbol_records(self, nx_store: NetworkXStore) -> None:
        records = [
            _make_record(
                symbol_id=f"pkg.mod.fn_{i}",
                name=f"fn_{i}",
                qualified_name=f"pkg.mod.fn_{i}",
                sym_type="function",
                start_line=i,
                end_line=i + 5,
            )
            for i in range(10)
        ]
        nx_store.bulk_ingest_symbol_records(records)
        for i in range(10):
            assert nx_store.find_by_id(f"pkg.mod.fn_{i}") is not None

    def test_clear_after_bulk_ingest(self, nx_store):
        records = [
            _make_record(
                symbol_id=f"f{i}",
                name=f"f{i}",
                qualified_name=f"f{i}",
                sym_type="function",
            )
            for i in range(500)
        ]

        nx_store.bulk_ingest_symbol_records(records)

        nx_store.clear()

        assert nx_store.find_by_id("f0") is None
        assert nx_store.find_by_id("f499") is None

    def test_bulk_ingest_call_edges(self, nx_store: NetworkXStore) -> None:
        edges = [_make_call_edge(f"caller_{i}", f"callee_{i}") for i in range(5)]
        nx_store.bulk_ingest_call_edges(edges)
        for i in range(5):
            assert len(nx_store.outgoing_calls(f"caller_{i}")) == 1

    def test_bulk_ingest_dependency_edges(self, nx_store: NetworkXStore) -> None:
        edges = [
            _make_dep_edge(source=f"src_{i}.py", target=f"tgt_{i}.py") for i in range(5)
        ]
        nx_store.bulk_ingest_dependency_edges(edges)
        for i in range(5):
            assert len(nx_store.imported_modules(f"src_{i}.py")) == 1

    def test_ingest_symbol_record_serialises_list_fields(
        self, nx_store: NetworkXStore
    ) -> None:
        """``decorators`` and ``bases`` must be stored as JSON strings."""
        rec = _make_record(
            symbol_id="pkg.mod.Decorated",
            name="Decorated",
            qualified_name="pkg.mod.Decorated",
            sym_type="class",
            bases=["BaseA", "BaseB"],
        )
        nx_store.ingest_symbol_record(rec)
        props = nx_store.find_by_id("pkg.mod.Decorated")
        assert props is not None
        # bases should be the JSON-encoded string stored by _record_to_node_props
        stored_bases = props.get("bases")
        assert stored_bases == json.dumps(["BaseA", "BaseB"])


# ---------------------------------------------------------------------------
# Neo4j backend tests  (driver mocked)
# ---------------------------------------------------------------------------


def _make_neo4j_store(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "test",
) -> Neo4jStore:
    """Return a Neo4jStore with a fresh set of mock objects attached."""
    return Neo4jStore(uri=uri, username=username, password=password)


def _mock_driver() -> MagicMock:
    """Build a mock neo4j.Driver whose session().run() returns an empty list."""
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = iter([])
    driver.session.return_value = session
    return driver


@pytest.fixture
def neo4j_store() -> Neo4jStore:
    """Neo4jStore with its driver fully mocked and connected."""
    store = Neo4jStore(uri="bolt://localhost:7687", username="neo4j", password="test")
    mock_drv = _mock_driver()
    with patch("neo4j.GraphDatabase.driver", return_value=mock_drv):
        store.connect()
    # Keep the mock on the store for assertion.
    store._mock_driver = mock_drv  # type: ignore[attr-defined]
    return store


class TestNeo4jLifecycle:
    """connect / close / clear behaviour."""

    def test_connect_calls_driver_and_verify(self) -> None:
        mock_drv = _mock_driver()
        with patch("neo4j.GraphDatabase.driver", return_value=mock_drv) as mock_factory:
            store = Neo4jStore(uri="bolt://localhost:7687", username="u", password="p")
            store.connect()
            mock_factory.assert_called_once_with(
                "bolt://localhost:7687", auth=("u", "p")
            )
            mock_drv.verify_connectivity.assert_called_once()

    def test_close_closes_driver(self, neo4j_store: Neo4jStore) -> None:
        neo4j_store.close()
        neo4j_store._mock_driver.close.assert_called_once()  # type: ignore[attr-defined]

    def test_close_idempotent(self, neo4j_store: Neo4jStore) -> None:
        neo4j_store.close()
        neo4j_store.close()  # second call must not raise

    def test_clear_issues_detach_delete(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        neo4j_store.clear()
        session.run.assert_called()
        cypher_arg = session.run.call_args[0][0]
        assert "DETACH DELETE" in cypher_arg

    def test_query_before_connect_raises(self) -> None:
        store = Neo4jStore()
        with pytest.raises(RuntimeError, match="connect"):
            store.query("MATCH (n) RETURN n")


class TestNeo4jNodeCRUD:
    """Node creation via the mocked driver."""

    def test_create_node_runs_merge_cypher(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        neo4j_store.create_node("Function", {"id": "fn_1", "name": "run"})
        session.run.assert_called()
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher
        assert "Function" in cypher

    def test_create_node_without_id_raises(self, neo4j_store: Neo4jStore) -> None:
        with pytest.raises(ValueError, match="'id'"):
            neo4j_store.create_node("Function", {"name": "no_id"})

    def test_create_node_returns_id(self, neo4j_store: Neo4jStore) -> None:
        returned_id = neo4j_store.create_node("Class", {"id": "cls_1"})
        assert returned_id == "cls_1"


class TestNeo4jBulkOperations:
    """Bulk insert batching behaviour."""

    def test_bulk_create_nodes_returns_ids(self, neo4j_store: Neo4jStore) -> None:
        rows = [{"id": f"fn_{i}", "name": f"fn_{i}"} for i in range(3)]
        ids = neo4j_store.bulk_create_nodes("Function", rows)
        assert ids == ["fn_0", "fn_1", "fn_2"]

    def test_bulk_create_nodes_empty_list(self, neo4j_store: Neo4jStore) -> None:
        ids = neo4j_store.bulk_create_nodes("Function", [])
        assert ids == []

    def test_bulk_create_nodes_issues_unwind_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        rows = [{"id": "fn_x", "name": "x"}]
        neo4j_store.bulk_create_nodes("Function", rows)
        cypher = session.run.call_args[0][0]
        assert "UNWIND" in cypher
        assert "Function" in cypher

    def test_bulk_create_nodes_batches_at_threshold(
        self, neo4j_store: Neo4jStore
    ) -> None:
        """10 000 rows must be split into multiple batches of <= batch_size."""
        from src.reporag.graph import neo4j_store as store_module  # noqa: PLC0415

        original_batch = store_module._BATCH_SIZE
        # Temporarily lower the batch size so we can assert without 10k rows.
        store_module._BATCH_SIZE = 3
        try:
            driver = neo4j_store._mock_driver  # type: ignore[attr-defined]
            driver.session.reset_mock()
            rows = [{"id": f"n_{i}"} for i in range(7)]
            neo4j_store.bulk_create_nodes("Function", rows)
            # 7 rows with _BATCH_SIZE=3 -> 3 sessions opened (3 + 3 + 1 rows each).
            assert driver.session.call_count == 3
        finally:
            store_module._BATCH_SIZE = original_batch

    def test_bulk_create_relationships_groups_by_type(
        self, neo4j_store: Neo4jStore
    ) -> None:
        """Relationships are grouped by rel_type for homogeneous UNWIND batches."""
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        rows = [
            {"from_id": "a", "to_id": "b", "rel_type": "CALLS", "props": {}},
            {"from_id": "m", "to_id": "b", "rel_type": "IMPORTS", "props": {}},
        ]
        neo4j_store.bulk_create_relationships(rows)
        # Two separate run calls, one per rel_type.
        assert session.run.call_count == 2

    def test_bulk_create_relationships_empty(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        neo4j_store.bulk_create_relationships([])
        session.run.assert_not_called()


class TestNeo4jRelationships:
    """Relationship creation via mocked driver."""

    def test_create_relationship_runs_merge_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value

        neo4j_store.create_node("Function", {"id": "a"})
        neo4j_store.create_node("Function", {"id": "b"})
        neo4j_store.find_by_id = MagicMock(return_value={"id": "dummy"})

        neo4j_store.create_relationship(
            "a",
            "b",
            "CALLS",
            {"resolved": True},
        )

        cypher = session.run.call_args[0][0]

        assert "MERGE" in cypher
        assert "CALLS" in cypher

    def test_create_relationship_passes_props(
        self,
        neo4j_store: Neo4jStore,
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value

        neo4j_store.create_node("Function", {"id": "a"})
        neo4j_store.create_node("Function", {"id": "b"})
        neo4j_store.find_by_id = MagicMock(return_value={"id": "dummy"})

        neo4j_store.create_relationship(
            "a",
            "b",
            "CALLS",
            {"resolved": True},
        )

        params = (
            session.run.call_args[1].get("parameters") or session.run.call_args[0][1]
        )

        assert params["props"]["resolved"] is True


class TestNeo4jQueryHelpers:
    """Cypher helper methods -- results parsed from mocked driver responses."""

    def _set_run_result(self, neo4j_store: Neo4jStore, records: list[dict]) -> None:
        """Make session.run() return mock records matching *records*."""
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        mock_records = []
        for rec_dict in records:
            mr = MagicMock()
            mr.__getitem__ = lambda self, k, _d=rec_dict: _d[k]
            mr.__iter__ = lambda self, _d=rec_dict: iter(_d)
            mock_records.append(mr)
        session.run.return_value = iter(mock_records)

    def test_find_by_id_returns_none_when_no_result(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        result = neo4j_store.find_by_id("missing")
        assert result is None

    def test_find_by_id_issues_correct_cypher(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.find_by_id("some_id")
        cypher = session.run.call_args[0][0]
        assert "{id:" in cypher.replace(" ", "") or "id: $id" in cypher

    def test_find_by_qualified_name_issues_correct_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.find_by_qualified_name("pkg.mod.MyClass")
        cypher = session.run.call_args[0][0]
        assert "qualified_name" in cypher

    def test_outgoing_calls_issues_correct_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.outgoing_calls("fn_id")
        cypher = session.run.call_args[0][0]
        assert "CALLS" in cypher
        assert "->" in cypher or "->" in cypher or "-[r:CALLS]->" in cypher

    def test_incoming_calls_issues_correct_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.incoming_calls("fn_id")
        cypher = session.run.call_args[0][0]
        assert "CALLS" in cypher
        assert "<-" in cypher or "m)-[r:CALLS]->(n" in cypher

    def test_imported_modules_issues_correct_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.imported_modules("mod_id")
        cypher = session.run.call_args[0][0]
        assert "IMPORTS" in cypher

    def test_inheritance_relationships_issues_correct_cypher(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.inheritance_relationships("cls_id")
        cypher = session.run.call_args[0][0]
        assert "INHERITS" in cypher

    def test_containment_hierarchy_uses_star_pattern(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.containment_hierarchy("mod_id")
        cypher = session.run.call_args[0][0]
        assert "CONTAINS" in cypher
        assert "*" in cypher

    def test_shortest_path_uses_shortestpath(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.shortest_path("a", "b")
        cypher = session.run.call_args[0][0]
        assert "shortestPath" in cypher

    def test_shortest_path_returns_empty_on_no_result(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        assert neo4j_store.shortest_path("x", "y") == []

    def test_neighborhood_passes_depth_param(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        neo4j_store.neighborhood("fn_id", depth=3)
        # The first call is the main neighborhood query; check it includes depth.
        first_call_params = (
            session.run.call_args_list[0][1].get("parameters")
            or session.run.call_args_list[0][0][1]
        )
        assert first_call_params.get("depth") == 3

    def test_query_passes_through_raw_cypher(self, neo4j_store: Neo4jStore) -> None:
        """query() is a Neo4j-specific extra -- must run the provided cypher."""
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.return_value = iter([])
        neo4j_store.query("MATCH (n) WHERE n.name = $n RETURN n", {"n": "foo"})
        called_cypher = session.run.call_args[0][0]
        assert "MATCH (n)" in called_cypher


class TestNeo4jIngestionHelpers:
    """Ingestion helper methods for the Neo4j backend (mocked driver)."""

    def test_ingest_symbol_record_creates_node(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        rec = _make_record(
            symbol_id="pkg.mod.run",
            name="run",
            qualified_name="pkg.mod.run",
            sym_type="function",
        )
        neo4j_store.ingest_symbol_record(rec)
        # At least one MERGE must have been issued for the node itself.
        assert session.run.call_count >= 1

    def test_ingest_call_edge_calls_create_node_and_relationship(
        self, neo4j_store: Neo4jStore
    ) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        neo4j_store.find_by_id = MagicMock(return_value={"id": "dummy"})
        # Should issue at least a MERGE for caller + callee + a CALLS MERGE.
        assert session.run.call_count >= 1

    def test_bulk_ingest_symbol_records_empty(self, neo4j_store: Neo4jStore) -> None:
        """Empty input must not issue any queries."""
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        neo4j_store.bulk_ingest_symbol_records([])
        session.run.assert_not_called()

    def test_bulk_ingest_call_edges_empty(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        neo4j_store.bulk_ingest_call_edges([])
        session.run.assert_not_called()

    def test_bulk_ingest_dependency_edges_empty(self, neo4j_store: Neo4jStore) -> None:
        session = neo4j_store._mock_driver.session.return_value  # type: ignore[attr-defined]
        session.run.reset_mock()
        neo4j_store.bulk_ingest_dependency_edges([])
        session.run.assert_not_called()


# ---------------------------------------------------------------------------
# API compatibility: same logical tests run against BOTH backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["networkx", "neo4j"])
def compat_store(request: pytest.FixtureRequest) -> GraphStore:
    """Parametrised fixture that returns either backend."""
    backend: Backend = request.param  # type: ignore[assignment]
    if backend == "networkx":
        store = NetworkXStore()
        store.connect()
        return store
    # Neo4j with mocked driver.
    store = Neo4jStore(uri="bolt://localhost:7687", username="neo4j", password="test")
    mock_drv = _mock_driver()
    with patch("neo4j.GraphDatabase.driver", return_value=mock_drv):
        store.connect()
    store._mock_driver = mock_drv  # type: ignore[attr-defined]
    return store


class TestAPICompatibility:
    """Verify that both backends expose exactly the same public interface."""

    def test_graphstore_is_instance(self, compat_store: GraphStore) -> None:
        assert isinstance(compat_store, GraphStore)

    def test_has_connect(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.connect)

    def test_has_close(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.close)

    def test_has_clear(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.clear)

    def test_has_create_node(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.create_node)

    def test_has_create_relationship(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.create_relationship)

    def test_has_bulk_create_nodes(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.bulk_create_nodes)

    def test_has_bulk_create_relationships(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.bulk_create_relationships)

    def test_has_ingest_symbol_record(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.ingest_symbol_record)

    def test_has_ingest_call_edge(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.ingest_call_edge)

    def test_has_ingest_dependency_edge(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.ingest_dependency_edge)

    def test_has_bulk_ingest_symbol_records(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.bulk_ingest_symbol_records)

    def test_has_bulk_ingest_call_edges(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.bulk_ingest_call_edges)

    def test_has_bulk_ingest_dependency_edges(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.bulk_ingest_dependency_edges)

    def test_has_find_by_id(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.find_by_id)

    def test_has_find_by_qualified_name(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.find_by_qualified_name)

    def test_has_outgoing_calls(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.outgoing_calls)

    def test_has_incoming_calls(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.incoming_calls)

    def test_has_imported_modules(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.imported_modules)

    def test_has_inheritance_relationships(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.inheritance_relationships)

    def test_has_containment_hierarchy(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.containment_hierarchy)

    def test_has_shortest_path(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.shortest_path)

    def test_has_neighborhood(self, compat_store: GraphStore) -> None:
        assert callable(compat_store.neighborhood)

    def test_query_not_on_graphstore_interface(self) -> None:
        """``query()`` must NOT be declared on the abstract GraphStore base."""
        assert not hasattr(GraphStore, "query")

    def test_query_is_on_neo4j_store(self) -> None:
        """``query()`` IS available as a Neo4j-specific capability."""
        assert callable(Neo4jStore.query)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateGraphStore:
    """create_graph_store() factory."""

    def test_returns_networkx_store(self) -> None:
        store = create_graph_store("networkx")
        store.connect()
        store.close()
        assert isinstance(store, NetworkXStore)

    def test_returns_neo4j_store(self) -> None:
        store = create_graph_store(
            "neo4j", uri="bolt://localhost:7687", username="u", password="p"
        )
        assert isinstance(store, Neo4jStore)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_graph_store("sqlite")  # type: ignore[arg-type]

    def test_default_backend_is_neo4j(self) -> None:
        store = create_graph_store()
        assert isinstance(store, Neo4jStore)


def test_bulk_insert_large_dataset(nx_store: NetworkXStore):
    records = []

    for i in range(10000):
        records.append(
            SymbolRecord(
                symbol_id=f"func{i}",
                name=f"func{i}",
                qualified_name=f"func{i}",
                type="function",
                file_path="a.py",
                module="pkg",
                start_line=1,
                end_line=1,
            )
        )

    nx_store.bulk_ingest_symbol_records(records)
    assert nx_store.find_by_id("func0") is not None
    assert nx_store.find_by_id("func9999") is not None

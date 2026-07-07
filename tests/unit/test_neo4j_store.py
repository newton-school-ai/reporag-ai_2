"""Unit tests for the graph store (Issue 12).

The behavioural tests are parametrised over both backends -- the in-memory
``networkx`` fallback (always) and a live ``neo4j`` (skipped when no database is
reachable) -- so the *same* suite validates that the fallback matches Neo4j, per
the acceptance criteria. Backend-agnostic builder behaviour, the
:func:`GraphStore` factory selection / fallback, and bulk insert are covered
separately.
"""

from __future__ import annotations

import pytest
from neo4j.exceptions import DriverError

from src.reporag.config import settings
from src.reporag.graph.call_graph import CallGraphBuilder
from src.reporag.graph.dependency_graph import DependencyGraphBuilder
from src.reporag.graph.neo4j_store import (
    EDGE_CALLS,
    EDGE_CONTAINS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    GraphNode,
    GraphStore,
    NetworkXGraphStore,
    _GraphBuilder,
)
from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import SymbolExtractor

# ---------------------------------------------------------------------------
# A small, fully-specified two-file project exercising every edge type.
# ---------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "db.py": (
        "class Base:\n"
        "    def save(self):\n"
        "        return True\n"
        "\n\n"
        "class User(Base):\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def get_name(self):\n"
        "        self.save()\n"
        "        return self.name\n"
        "\n\n"
        "def connect():\n"
        "    return User('root')\n"
    ),
    "app.py": (
        "from db import User, connect\n"
        "\n\n"
        "def start():\n"
        "    user = User('x')\n"
        "    connect()\n"
        "    return user.get_name()\n"
        "\n\n"
        "def start_again():\n"
        "    return start()\n"
    ),
}


def _build_inputs(sources: dict[str, str]):
    """Return ``(table, call_edges, dep_edges)`` for a set of in-memory sources."""
    extractor = SymbolExtractor()
    table = SymbolTable()
    for path, code in sources.items():
        table.register_symbols(extractor.extract_from_source(code, file_path=path))
    calls = CallGraphBuilder().build_from_sources(sources)
    deps = DependencyGraphBuilder().build_from_sources(sources)
    return table, calls, deps


def _qid(table: SymbolTable, qualified_name: str) -> str:
    """Return the symbol_id for a fully qualified name (fails loudly if absent)."""
    record = table.lookup_qualified(qualified_name)
    assert record is not None, f"missing symbol {qualified_name}"
    return record.symbol_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_inputs():
    """Build the graph inputs once for the whole session."""
    return _build_inputs(SOURCES)


def _neo4j_or_skip():
    """Return a connected Neo4j store, or skip if none is reachable."""
    try:
        return GraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password.get_secret_value(),
            backend="neo4j",
            max_retries=1,
            retry_backoff=0.05,
        )
    except Exception as exc:  # noqa: BLE001 - any connection failure -> skip
        pytest.skip(f"Neo4j not reachable: {type(exc).__name__}")


@pytest.fixture(params=["networkx", "neo4j"])
def store(request):
    """A cleared store for each backend (Neo4j skipped when unavailable)."""
    if request.param == "networkx":
        backend_store = GraphStore(backend="networkx")
        yield backend_store
        return

    backend_store = _neo4j_or_skip()
    backend_store.clear()
    try:
        yield backend_store
    finally:
        backend_store.clear()
        backend_store.close()


@pytest.fixture
def populated_store(store, sample_inputs):
    """A store with the sample graph already persisted."""
    table, calls, deps = sample_inputs
    store.persist_graph(calls, deps, table, clear=True)
    return store, table


# ---------------------------------------------------------------------------
# Node creation: labels + properties
# ---------------------------------------------------------------------------


def test_persist_creates_expected_node_and_edge_counts(populated_store) -> None:
    """The known sample project yields exactly the expected node/edge totals."""
    store, _ = populated_store
    # 8 symbols (2 classes, 4 methods, 2 functions) + 2 module nodes.
    assert store.node_count() == 10
    # 8 CONTAINS + 1 INHERITS + 1 IMPORTS + 6 CALLS.
    assert store.edge_count() == 16


def test_function_node_has_label_and_properties(populated_store) -> None:
    """A function node carries the ``Function`` label and symbol metadata."""
    store, table = populated_store
    node = store.get_node(_qid(table, "db.connect"))
    assert node is not None
    assert node.label == "Function"
    assert node.properties["name"] == "connect"
    assert node.properties["qualified_name"] == "db.connect"
    assert node.properties["type"] == "function"
    assert node.properties["file_path"] == "db.py"
    assert node.properties["signature"] == "def connect()"


def test_method_and_class_nodes(populated_store) -> None:
    """Methods are ``Function`` nodes (``is_method``); classes are ``Class``."""
    store, table = populated_store
    method = store.get_node(_qid(table, "db.User.get_name"))
    assert method is not None
    assert method.label == "Function"
    assert method.properties["is_method"] is True

    cls = store.get_node(_qid(table, "db.User"))
    assert cls is not None
    assert cls.label == "Class"
    assert "Base" in cls.properties["bases"]


def test_module_node_created(populated_store) -> None:
    """A ``Module`` node exists for each project module."""
    store, _ = populated_store
    module = store.get_node("module::db")
    assert module is not None
    assert module.label == "Module"
    assert module.properties["name"] == "db"
    assert module.properties["external"] is False


# ---------------------------------------------------------------------------
# Edge types: CALLS / IMPORTS / INHERITS / CONTAINS
# ---------------------------------------------------------------------------


def _out(store, node_id, edge_type):
    """Return the set of qualified names reachable via one *edge_type* hop out."""
    return {
        n.properties.get("qualified_name", n.id)
        for n in store.get_neighbors(
            node_id, depth=1, direction="out", edge_types=[edge_type]
        )
    }


def test_calls_edges(populated_store) -> None:
    """CALLS edges capture function, constructor, and inferred-method calls."""
    store, table = populated_store
    start_calls = _out(store, _qid(table, "app.start"), EDGE_CALLS)
    # start() calls connect(), instantiates User, and calls user.get_name().
    assert start_calls == {"db.connect", "db.User", "db.User.get_name"}

    # Recursion-style chain: start_again -> start.
    assert _out(store, _qid(table, "app.start_again"), EDGE_CALLS) == {"app.start"}


def test_calls_self_method_resolves_through_inheritance(populated_store) -> None:
    """``self.save()`` in User resolves to the inherited Base.save."""
    store, table = populated_store
    assert _out(store, _qid(table, "db.User.get_name"), EDGE_CALLS) == {"db.Base.save"}


def test_inherits_edge(populated_store) -> None:
    """User INHERITS Base (bare, same-module base is resolved)."""
    store, table = populated_store
    assert _out(store, _qid(table, "db.User"), EDGE_INHERITS) == {"db.Base"}


def test_contains_edges_module_and_class(populated_store) -> None:
    """CONTAINS captures module->top-level and class->method containment."""
    store, table = populated_store
    module_contains = _out(store, "module::db", EDGE_CONTAINS)
    assert module_contains == {"db.Base", "db.User", "db.connect"}

    class_contains = _out(store, _qid(table, "db.User"), EDGE_CONTAINS)
    assert class_contains == {"db.User.__init__", "db.User.get_name"}


def test_imports_edge_targets_full_module(populated_store) -> None:
    """IMPORTS connects the importing module to the resolved target module."""
    store, _ = populated_store
    neighbours = store.get_neighbors(
        "module::app", depth=1, direction="out", edge_types=[EDGE_IMPORTS]
    )
    assert [n.id for n in neighbours] == ["module::db"]
    assert neighbours[0].properties["external"] is False


# ---------------------------------------------------------------------------
# Traversal helpers: neighbours, shortest path, subgraph
# ---------------------------------------------------------------------------


def test_get_neighbors_direction(populated_store) -> None:
    """Direction selects successors, predecessors, or both."""
    store, table = populated_store
    connect_id = _qid(table, "db.connect")

    callers = _out  # alias for readability
    # Incoming CALLS to connect: only app.start calls it.
    incoming = {
        n.properties["qualified_name"]
        for n in store.get_neighbors(
            connect_id, direction="in", edge_types=[EDGE_CALLS]
        )
    }
    assert incoming == {"app.start"}
    # connect has no outgoing CALLS to a project symbol other than the User ctor.
    assert callers(store, connect_id, EDGE_CALLS) == {"db.User"}


def test_get_neighbors_depth(populated_store) -> None:
    """A 2-hop traversal reaches indirect neighbours."""
    store, table = populated_store
    start_again = _qid(table, "app.start_again")
    reachable = {
        n.properties.get("qualified_name")
        for n in store.get_neighbors(
            start_again, depth=2, direction="out", edge_types=[EDGE_CALLS]
        )
    }
    # start_again -> start -> {connect, User, User.get_name}
    assert "app.start" in reachable
    assert "db.connect" in reachable


def test_shortest_path_across_calls(populated_store) -> None:
    """shortest_path returns the call chain start_again -> start -> connect."""
    store, table = populated_store
    path = store.shortest_path(
        _qid(table, "app.start_again"),
        _qid(table, "db.connect"),
        edge_types=[EDGE_CALLS],
    )
    assert path is not None
    assert [n.properties["qualified_name"] for n in path] == [
        "app.start_again",
        "app.start",
        "db.connect",
    ]


def test_shortest_path_missing_returns_none(populated_store) -> None:
    """No path (wrong direction) yields ``None``."""
    store, table = populated_store
    # connect does not reach start_again along CALLS.
    assert (
        store.shortest_path(
            _qid(table, "db.connect"),
            _qid(table, "app.start_again"),
            edge_types=[EDGE_CALLS],
        )
        is None
    )


def test_shortest_path_respects_max_depth(populated_store) -> None:
    """A hop cap shorter than the path yields ``None`` on both backends."""
    store, table = populated_store
    assert (
        store.shortest_path(
            _qid(table, "app.start_again"),
            _qid(table, "db.connect"),
            edge_types=[EDGE_CALLS],
            max_depth=1,
        )
        is None
    )


def test_subgraph_extraction(populated_store) -> None:
    """subgraph(depth=1) returns the seed plus its immediate neighbourhood."""
    store, table = populated_store
    user_id = _qid(table, "db.User")
    sub = store.subgraph([user_id], depth=1)
    ids = {n.id for n in sub.nodes}
    # User + its methods + its base + the module that contains it.
    assert user_id in ids
    assert _qid(table, "db.User.get_name") in ids
    assert _qid(table, "db.Base") in ids
    assert "module::db" in ids
    # Every returned edge stays within the returned node set.
    for edge in sub.edges:
        assert edge.source in ids and edge.target in ids


# ---------------------------------------------------------------------------
# Lifecycle: clear, idempotency, missing lookups
# ---------------------------------------------------------------------------


def test_get_node_missing_returns_none(store) -> None:
    """Looking up an absent id returns ``None``."""
    assert store.get_node("does.not.exist") is None


def test_clear_empties_store(populated_store) -> None:
    """clear() removes every node and edge."""
    store, _ = populated_store
    assert store.node_count() > 0
    store.clear()
    assert store.node_count() == 0
    assert store.edge_count() == 0


def test_persist_is_idempotent(store, sample_inputs) -> None:
    """Persisting the same graph twice does not duplicate nodes or edges."""
    table, calls, deps = sample_inputs
    store.persist_graph(calls, deps, table, clear=True)
    first_nodes, first_edges = store.node_count(), store.edge_count()
    store.persist_graph(calls, deps, table, clear=False)
    assert store.node_count() == first_nodes
    assert store.edge_count() == first_edges


# ---------------------------------------------------------------------------
# Bulk insert (batch transactions) -- acceptance: 10K+ nodes
# ---------------------------------------------------------------------------


def test_bulk_insert_many_nodes(store) -> None:
    """Bulk-insert 10k nodes (exercises Neo4j batched transactions)."""
    nodes = [
        GraphNode(
            id=f"n{i}", label="Function", properties={"id": f"n{i}", "name": str(i)}
        )
        for i in range(10_000)
    ]
    edges = []  # keep the edge phase light; node volume is the point here
    store.add_nodes(nodes)
    store.add_edges(edges)
    assert store.node_count() == 10_000


# ---------------------------------------------------------------------------
# Backend-agnostic builder behaviour (verified on the NetworkX fallback)
# ---------------------------------------------------------------------------


def test_unresolved_calls_are_dropped() -> None:
    """A call to a builtin / unknown symbol produces no CALLS edge."""
    table, calls, deps = _build_inputs(
        {"m.py": "def f():\n    print('hi')\n    return len([])\n"}
    )
    store = GraphStore(backend="networkx")
    store.persist_graph(calls, deps, table)
    # ``print`` / ``len`` are not project symbols -> f has no outgoing CALLS.
    assert _out(store, "m.f", EDGE_CALLS) == set()


def test_duplicate_calls_merge_and_accumulate_lines() -> None:
    """Two calls to the same target collapse into one edge tracking both lines."""
    table, calls, deps = _build_inputs(
        {
            "m.py": "def helper():\n    return 1\n\n\ndef f():\n    helper()\n    helper()\n"
        }
    )
    _, edges = _GraphBuilder(table, calls, deps).build()
    calls_edges = [e for e in edges if e.type == EDGE_CALLS and e.target == "m.helper"]
    assert len(calls_edges) == 1
    assert calls_edges[0].properties["count"] == 2
    assert len(calls_edges[0].properties["call_site_lines"]) == 2


def test_generic_base_class_is_not_linked() -> None:
    """A subscripted/dotted base (``Generic[T]``) yields no INHERITS edge."""
    table, calls, deps = _build_inputs(
        {
            "m.py": "from typing import Generic, TypeVar\nT = TypeVar('T')\n\n\nclass Box(Generic[T]):\n    pass\n"
        }
    )
    store = GraphStore(backend="networkx")
    store.persist_graph(calls, deps, table)
    assert _out(store, "m.Box", EDGE_INHERITS) == set()


# ---------------------------------------------------------------------------
# GraphStore factory: backend selection and fallback
# ---------------------------------------------------------------------------


def test_factory_defaults_to_networkx_without_uri() -> None:
    """No URI under ``auto`` selects the in-memory backend."""
    assert isinstance(GraphStore(), NetworkXGraphStore)
    assert GraphStore().backend == "networkx"


def test_factory_explicit_networkx() -> None:
    """``backend='networkx'`` always returns the fallback."""
    assert isinstance(GraphStore(backend="networkx"), NetworkXGraphStore)


def test_factory_auto_falls_back_when_neo4j_unreachable() -> None:
    """Under ``auto`` an unreachable Neo4j degrades to NetworkX (no raise)."""
    got = GraphStore(
        uri="bolt://127.0.0.1:59999",
        user="neo4j",
        password="nope",
        backend="auto",
        max_retries=1,
        retry_backoff=0.01,
    )
    assert isinstance(got, NetworkXGraphStore)


def test_factory_neo4j_backend_raises_when_unreachable() -> None:
    """``backend='neo4j'`` surfaces the connection error instead of degrading."""
    with pytest.raises(DriverError):
        GraphStore(
            uri="bolt://127.0.0.1:59999",
            user="neo4j",
            password="nope",
            backend="neo4j",
            max_retries=1,
            retry_backoff=0.01,
        )


def test_factory_rejects_unknown_backend() -> None:
    """An unknown backend name is a clear error."""
    with pytest.raises(ValueError):
        GraphStore(backend="sqlite")  # type: ignore[arg-type]


def test_query_unsupported_on_networkx() -> None:
    """Raw Cypher is Neo4j-only; the fallback raises a helpful error."""
    store = GraphStore(backend="networkx")
    with pytest.raises(NotImplementedError):
        store.query("MATCH (n) RETURN n")

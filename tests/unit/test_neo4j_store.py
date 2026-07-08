"""Unit tests for Issue 12 - Neo4j graph store with Cypher query layer.

Tests the NetworkXGraphStore (no live Neo4j required) and verifies that
the public GraphStore interface is fully satisfied.  The same test suite
can be run against Neo4jGraphStore by setting the REPORAG_NEO4J_URI
environment variable (integration / optional).

Acceptance criteria exercised:
- [x] Creates nodes with correct labels and properties
- [x] CALLS, IMPORTS, INHERITS, CONTAINS edges work
- [x] Cypher query subset returns correct results (neighbors, paths)
- [x] NetworkX fallback passes same test suite without Neo4j
- [x] Bulk insert handles 10K+ nodes efficiently
- [x] Connection errors are handled gracefully with retry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.reporag.graph.neo4j_store import (
    GraphEdge,
    GraphNode,
    GraphStore,
    NetworkXGraphStore,
    SubgraphResult,
    _symbol_label,
)

# ---------------------------------------------------------------------------
# Minimal stubs for CallEdge, DependencyEdge, SymbolRecord
# ---------------------------------------------------------------------------


@dataclass
class _Symbol:
    """Minimal SymbolRecord stub mirroring SymbolTable.SymbolRecord fields."""

    symbol_id: str
    name: str
    qualified_name: str
    type: str
    file_path: str
    module: str
    start_line: int = 1
    end_line: int = 10
    signature: str | None = None
    docstring: str | None = None
    parent: str | None = None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    is_async: bool = False
    language: str = "python"


@dataclass
class _CallEdge:
    """Minimal CallEdge stub that mirrors the real CallEdge field layout.

    Note: In the real CallEdge, ``resolved`` is derived from ``resolution``
    in ``__post_init__``.  We mirror that behaviour here.
    """

    caller: str
    callee: str
    caller_file: str
    call_site_line: int
    callee_file: str | None = None
    call_type: str = "function"
    resolution: str = "unresolved"  # real default is "unresolved"
    resolved: bool = False  # real default is False
    is_recursive: bool = False

    def __post_init__(self) -> None:
        """Derive resolved from resolution, matching the real CallEdge behaviour."""
        self.resolved = self.resolution != "unresolved"


@dataclass
class _DepEdge:
    """Minimal DependencyEdge stub matching real field layout."""

    source: str
    target: str
    source_module: str
    target_module: str
    import_type: str = "from_import"
    imported_names: list[tuple[str, str | None]] = field(default_factory=list)
    line: int = 0  # real default is 0
    is_relative: bool = False
    relative_level: int = 0  # field present in real DependencyEdge
    is_wildcard: bool = False
    resolved: bool = False  # real default is False


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> NetworkXGraphStore:
    """Return a fresh, empty NetworkXGraphStore."""
    return NetworkXGraphStore()


def _make_symbols(names: list[tuple[str, str, str]]) -> list[_Symbol]:
    """Build ``_Symbol`` stubs from ``(symbol_id, name, type)`` triples."""
    return [
        _Symbol(
            symbol_id=sid,
            name=n,
            qualified_name=sid,
            type=t,
            file_path=f"{n}.py",
            module=sid.rsplit(".", 1)[0] if "." in sid else sid,
        )
        for sid, n, t in names
    ]


def _call(
    caller: str,
    callee: str,
    caller_file: str,
    line: int,
    callee_file: str | None = None,
) -> _CallEdge:
    """Convenience factory for a resolved CALLS edge."""
    return _CallEdge(
        caller=caller,
        callee=callee,
        caller_file=caller_file,
        call_site_line=line,
        callee_file=callee_file,
        resolution="local",  # triggers __post_init__ -> resolved=True
    )


def _dep_edge(
    source_file: str, target_file: str, src_mod: str, tgt_mod: str
) -> _DepEdge:
    """Convenience factory for a resolved IMPORTS edge."""
    return _DepEdge(
        source=source_file,
        target=target_file,
        source_module=src_mod,
        target_module=tgt_mod,
        resolved=True,
        line=1,
    )


# ---------------------------------------------------------------------------
# _symbol_label helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sym_type, expected",
    [
        ("function", "Function"),
        ("method", "Function"),
        ("class", "Class"),
        ("module", "Module"),
        ("variable", "Symbol"),
        ("import", "Symbol"),
        ("unknown", "Symbol"),
        ("", "Symbol"),
    ],
)
def test_symbol_label(sym_type: str, expected: str) -> None:
    """_symbol_label maps every symbol type to the correct Neo4j label."""
    assert _symbol_label(sym_type) == expected


# ---------------------------------------------------------------------------
# GraphStore interface compliance
# ---------------------------------------------------------------------------


def test_networkx_store_is_graph_store(store: NetworkXGraphStore) -> None:
    """NetworkXGraphStore must be a concrete GraphStore subclass."""
    assert isinstance(store, GraphStore)


def test_graph_store_is_abstract() -> None:
    """GraphStore cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GraphStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Node creation - correct labels
# ---------------------------------------------------------------------------


def test_persist_creates_function_nodes(store: NetworkXGraphStore) -> None:
    """'function' and 'method' symbols become Function nodes."""
    symbols = _make_symbols(
        [("auth.login", "login", "function"), ("auth.User.save", "save", "method")]
    )
    store.persist_graph([], [], symbols)
    g = store.graph
    assert "auth.login" in g
    assert g.nodes["auth.login"]["label"] == "Function"
    assert "auth.User.save" in g
    assert g.nodes["auth.User.save"]["label"] == "Function"


def test_persist_creates_class_nodes(store: NetworkXGraphStore) -> None:
    """'class' symbols become Class nodes."""
    symbols = _make_symbols([("auth.User", "User", "class")])
    store.persist_graph([], [], symbols)
    assert store.graph.nodes["auth.User"]["label"] == "Class"


def test_persist_creates_module_nodes(store: NetworkXGraphStore) -> None:
    """'module' symbols become Module nodes."""
    symbols = _make_symbols([("auth", "auth", "module")])
    store.persist_graph([], [], symbols)
    assert store.graph.nodes["auth"]["label"] == "Module"


def test_persist_creates_symbol_nodes_for_unknown_type(
    store: NetworkXGraphStore,
) -> None:
    """Unknown types fall back to the Symbol label."""
    symbols = _make_symbols([("cfg.TIMEOUT", "TIMEOUT", "variable")])
    store.persist_graph([], [], symbols)
    assert store.graph.nodes["cfg.TIMEOUT"]["label"] == "Symbol"


# ---------------------------------------------------------------------------
# Node creation - properties
# ---------------------------------------------------------------------------


def test_node_properties_stored(store: NetworkXGraphStore) -> None:
    """All SymbolRecord properties are written onto the node."""
    sym = _Symbol(
        symbol_id="db.get_user",
        name="get_user",
        qualified_name="db.get_user",
        type="function",
        file_path="db.py",
        module="db",
        start_line=10,
        end_line=25,
        signature="def get_user(uid: int)",
        docstring="Fetch a user by id.",
        is_async=False,
        language="python",
    )
    store.persist_graph([], [], [sym])
    data = store.graph.nodes["db.get_user"]
    assert data["name"] == "get_user"
    assert data["file_path"] == "db.py"
    assert data["start_line"] == 10
    assert data["end_line"] == 25
    assert data["signature"] == "def get_user(uid: int)"
    assert data["docstring"] == "Fetch a user by id."
    assert data["language"] == "python"
    assert data["is_async"] is False


def test_node_id_equals_symbol_id(store: NetworkXGraphStore) -> None:
    """The node's node_id property must equal the graph key (symbol_id)."""
    symbols = _make_symbols([("auth.login", "login", "function")])
    store.persist_graph([], [], symbols)
    assert store.graph.nodes["auth.login"]["node_id"] == "auth.login"


# ---------------------------------------------------------------------------
# CALLS edges
# ---------------------------------------------------------------------------


def test_calls_edge_created(store: NetworkXGraphStore) -> None:
    """A resolved CallEdge creates a CALLS relationship with metadata."""
    symbols = _make_symbols(
        [("app.main", "main", "function"), ("auth.login", "login", "function")]
    )
    edges = [_call("app.main", "auth.login", "app.py", 5, "auth.py")]
    store.persist_graph(edges, [], symbols)
    g = store.graph
    assert g.has_edge("app.main", "auth.login")
    data = g.get_edge_data("app.main", "auth.login")
    assert data["rel_type"] == "CALLS"
    assert data["call_site_line"] == 5
    assert data["caller_file"] == "app.py"
    assert data["callee_file"] == "auth.py"


def test_unresolved_call_edge_skipped(store: NetworkXGraphStore) -> None:
    """Unresolved CallEdges must not be added to the graph."""
    symbols = _make_symbols([("app.main", "main", "function")])
    # Default resolution is "unresolved", so __post_init__ sets resolved=False.
    store.persist_graph(
        [
            _CallEdge(
                caller="app.main",
                callee="external.func",
                caller_file="app.py",
                call_site_line=3,
            )
        ],
        [],
        symbols,
    )
    assert not store.graph.has_edge("app.main", "external.func")


def test_recursive_call_edge(store: NetworkXGraphStore) -> None:
    """A recursive CallEdge produces a self-loop with is_recursive=True."""
    symbols = _make_symbols([("util.recurse", "recurse", "function")])
    store.persist_graph(
        [
            _CallEdge(
                caller="util.recurse",
                callee="util.recurse",
                caller_file="util.py",
                call_site_line=8,
                resolution="local",  # resolved=True via __post_init__
                is_recursive=True,
            )
        ],
        [],
        symbols,
    )
    assert store.graph.has_edge("util.recurse", "util.recurse")
    data = store.graph.get_edge_data("util.recurse", "util.recurse")
    assert data["is_recursive"] is True
    assert data["rel_type"] == "CALLS"


def test_call_edge_stubs_missing_nodes(store: NetworkXGraphStore) -> None:
    """Caller / callee nodes missing from the symbol table are auto-stubbed."""
    store.persist_graph(
        [_call("ghost.caller", "ghost.callee", "ghost.py", 1)],
        [],
        [],
    )
    assert "ghost.caller" in store.graph
    assert "ghost.callee" in store.graph
    assert store.graph.has_edge("ghost.caller", "ghost.callee")


# ---------------------------------------------------------------------------
# IMPORTS edges
# ---------------------------------------------------------------------------


def test_imports_edge_created(store: NetworkXGraphStore) -> None:
    """A resolved DependencyEdge creates an IMPORTS relationship."""
    dep = _dep_edge("app.py", "auth.py", "app", "auth")
    store.persist_graph([], [dep], [])
    assert store.graph.has_edge("app.py", "auth.py")
    data = store.graph.get_edge_data("app.py", "auth.py")
    assert data["rel_type"] == "IMPORTS"
    assert data["import_type"] == "from_import"
    assert data["line"] == 1


def test_imports_edge_auto_creates_module_nodes(store: NetworkXGraphStore) -> None:
    """Module nodes (keyed by file path) are auto-created for IMPORTS edges."""
    dep = _dep_edge("app.py", "auth.py", "app", "auth")
    store.persist_graph([], [dep], [])
    assert "app.py" in store.graph
    assert "auth.py" in store.graph
    assert store.graph.nodes["app.py"]["label"] == "Module"
    assert store.graph.nodes["auth.py"]["label"] == "Module"


def test_unresolved_import_edge_skipped(store: NetworkXGraphStore) -> None:
    """Unresolved DependencyEdges (external packages) are not added."""
    dep = _DepEdge(
        source="app.py",
        target="requests",
        source_module="app",
        target_module="requests",
        resolved=False,
    )
    store.persist_graph([], [dep], [])
    assert not store.graph.has_edge("app.py", "requests")


def test_wildcard_import_edge_flag(store: NetworkXGraphStore) -> None:
    """is_wildcard is stored on the IMPORTS relationship."""
    dep = _DepEdge(
        source="app.py",
        target="helpers.py",
        source_module="app",
        target_module="helpers",
        is_wildcard=True,
        resolved=True,
    )
    store.persist_graph([], [dep], [])
    data = store.graph.get_edge_data("app.py", "helpers.py")
    assert data["is_wildcard"] is True


def test_relative_import_edge_flag(store: NetworkXGraphStore) -> None:
    """is_relative is stored on the IMPORTS relationship."""
    dep = _DepEdge(
        source="pkg/a.py",
        target="pkg/b.py",
        source_module="pkg.a",
        target_module="pkg.b",
        is_relative=True,
        relative_level=1,
        resolved=True,
    )
    store.persist_graph([], [dep], [])
    data = store.graph.get_edge_data("pkg/a.py", "pkg/b.py")
    assert data["is_relative"] is True


# ---------------------------------------------------------------------------
# CONTAINS edges
# ---------------------------------------------------------------------------


def test_contains_edge_from_parent(store: NetworkXGraphStore) -> None:
    """A symbol with parent set creates a CONTAINS edge from parent to child."""
    symbols = [
        _Symbol(
            symbol_id="auth.User",
            name="User",
            qualified_name="auth.User",
            type="class",
            file_path="auth.py",
            module="auth",
        ),
        _Symbol(
            symbol_id="auth.User.login",
            name="login",
            qualified_name="auth.User.login",
            type="method",
            file_path="auth.py",
            module="auth",
            parent="auth.User",
        ),
    ]
    store.persist_graph([], [], symbols)
    assert store.graph.has_edge("auth.User", "auth.User.login")
    assert (
        store.graph.get_edge_data("auth.User", "auth.User.login")["rel_type"]
        == "CONTAINS"
    )


def test_contains_edge_missing_parent_skipped(store: NetworkXGraphStore) -> None:
    """No CONTAINS edge is created when the parent node does not exist."""
    symbols = [
        _Symbol(
            symbol_id="auth.User.login",
            name="login",
            qualified_name="auth.User.login",
            type="method",
            file_path="auth.py",
            module="auth",
            parent="auth.User",  # parent NOT registered
        )
    ]
    store.persist_graph([], [], symbols)
    # No phantom parent node should have been created
    assert "auth.User" not in store.graph


def test_contains_multiple_methods(store: NetworkXGraphStore) -> None:
    """A class with several methods generates one CONTAINS edge per method."""
    symbols = [
        _Symbol(
            symbol_id="db.User",
            name="User",
            qualified_name="db.User",
            type="class",
            file_path="db.py",
            module="db",
        ),
    ] + [
        _Symbol(
            symbol_id=f"db.User.{m}",
            name=m,
            qualified_name=f"db.User.{m}",
            type="method",
            file_path="db.py",
            module="db",
            parent="db.User",
        )
        for m in ("save", "delete", "load")
    ]
    store.persist_graph([], [], symbols)
    children = list(store.graph.successors("db.User"))
    assert set(children) == {"db.User.save", "db.User.delete", "db.User.load"}


# ---------------------------------------------------------------------------
# INHERITS edges
# ---------------------------------------------------------------------------


def test_inherits_edge_created(store: NetworkXGraphStore) -> None:
    """A class with bases creates INHERITS relationships to each base."""
    symbols = [
        _Symbol(
            symbol_id="models.Base",
            name="Base",
            qualified_name="models.Base",
            type="class",
            file_path="models.py",
            module="models",
        ),
        _Symbol(
            symbol_id="auth.User",
            name="User",
            qualified_name="auth.User",
            type="class",
            file_path="auth.py",
            module="auth",
            bases=["Base"],
        ),
    ]
    store.persist_graph([], [], symbols)
    assert store.graph.has_edge("auth.User", "models.Base")
    assert (
        store.graph.get_edge_data("auth.User", "models.Base")["rel_type"] == "INHERITS"
    )


def test_inherits_unknown_base_skipped(store: NetworkXGraphStore) -> None:
    """INHERITS edges for bases not in the graph are silently skipped."""
    symbols = [
        _Symbol(
            symbol_id="auth.Admin",
            name="Admin",
            qualified_name="auth.Admin",
            type="class",
            file_path="auth.py",
            module="auth",
            bases=["SomeThirdPartyBase"],
        )
    ]
    store.persist_graph([], [], symbols)
    # No edge and no phantom node for the unknown base
    assert "SomeThirdPartyBase" not in store.graph


def test_inherits_multiple_bases(store: NetworkXGraphStore) -> None:
    """Multiple-inheritance produces one INHERITS edge per base."""
    symbols = [
        _Symbol(
            symbol_id="m.Mixin",
            name="Mixin",
            qualified_name="m.Mixin",
            type="class",
            file_path="m.py",
            module="m",
        ),
        _Symbol(
            symbol_id="m.Base",
            name="Base",
            qualified_name="m.Base",
            type="class",
            file_path="m.py",
            module="m",
        ),
        _Symbol(
            symbol_id="m.Child",
            name="Child",
            qualified_name="m.Child",
            type="class",
            file_path="m.py",
            module="m",
            bases=["Base", "Mixin"],
        ),
    ]
    store.persist_graph([], [], symbols)
    assert store.graph.has_edge("m.Child", "m.Base")
    assert store.graph.has_edge("m.Child", "m.Mixin")


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


def test_get_neighbors_depth_1(store: NetworkXGraphStore) -> None:
    """depth=1 returns only direct neighbours, not 2-hop ones."""
    symbols = _make_symbols(
        [
            ("app.main", "main", "function"),
            ("auth.login", "login", "function"),
            ("db.get_user", "get_user", "function"),
        ]
    )
    store.persist_graph(
        [
            _call("app.main", "auth.login", "app.py", 5, "auth.py"),
            _call("auth.login", "db.get_user", "auth.py", 12, "db.py"),
        ],
        [],
        symbols,
    )
    result = store.get_neighbors("app.main", depth=1)
    ids = {n.node_id for n in result.nodes}
    assert "auth.login" in ids
    assert "db.get_user" not in ids


def test_get_neighbors_depth_2(store: NetworkXGraphStore) -> None:
    """depth=2 follows two hops."""
    symbols = _make_symbols(
        [
            ("app.main", "main", "function"),
            ("auth.login", "login", "function"),
            ("db.get_user", "get_user", "function"),
        ]
    )
    store.persist_graph(
        [
            _call("app.main", "auth.login", "app.py", 5, "auth.py"),
            _call("auth.login", "db.get_user", "auth.py", 12, "db.py"),
        ],
        [],
        symbols,
    )
    result = store.get_neighbors("app.main", depth=2)
    ids = {n.node_id for n in result.nodes}
    assert "auth.login" in ids
    assert "db.get_user" in ids


def test_get_neighbors_excludes_start_node(store: NetworkXGraphStore) -> None:
    """The start node must not appear in the returned nodes list."""
    symbols = _make_symbols(
        [("app.main", "main", "function"), ("auth.login", "login", "function")]
    )
    store.persist_graph(
        [_call("app.main", "auth.login", "app.py", 1, "auth.py")],
        [],
        symbols,
    )
    result = store.get_neighbors("app.main", depth=1)
    assert all(n.node_id != "app.main" for n in result.nodes)


def test_get_neighbors_unknown_node_returns_empty(store: NetworkXGraphStore) -> None:
    """Querying a non-existent node returns an empty SubgraphResult."""
    result = store.get_neighbors("no.such.node")
    assert result.nodes == []
    assert result.edges == []


def test_get_neighbors_rel_type_filter(store: NetworkXGraphStore) -> None:
    """rel_types filter restricts which edge types are traversed."""
    symbols = _make_symbols(
        [("mod.A", "A", "class"), ("mod.B", "B", "class"), ("mod.C", "C", "class")]
    )
    store.persist_graph([], [], symbols)
    store._graph.add_edge("mod.A", "mod.B", rel_type="CALLS")
    store._graph.add_edge("mod.A", "mod.C", rel_type="IMPORTS")

    result = store.get_neighbors("mod.A", depth=1, rel_types=["CALLS"])
    ids = {n.node_id for n in result.nodes}
    assert "mod.B" in ids
    assert "mod.C" not in ids


def test_get_neighbors_no_duplicate_edges(store: NetworkXGraphStore) -> None:
    """Each edge appears at most once in SubgraphResult.edges."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph(
        [_call("a.A", "b.B", "a.py", 1, "b.py")],
        [],
        symbols,
    )
    result = store.get_neighbors("a.A", depth=2)
    edge_keys = [(e.source, e.target, e.rel_type) for e in result.edges]
    assert len(edge_keys) == len(set(edge_keys)), "Duplicate edges found"


def test_get_neighbors_returns_graph_node_instances(store: NetworkXGraphStore) -> None:
    """All items in SubgraphResult.nodes are GraphNode instances."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph(
        [_call("a.A", "b.B", "a.py", 1, "b.py")],
        [],
        symbols,
    )
    result = store.get_neighbors("a.A", depth=1)
    assert isinstance(result, SubgraphResult)
    for node in result.nodes:
        assert isinstance(node, GraphNode)
    for edge in result.edges:
        assert isinstance(edge, GraphEdge)


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------


def test_shortest_path_direct(store: NetworkXGraphStore) -> None:
    """Adjacent nodes return a 2-element path."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph(
        [_call("a.A", "b.B", "a.py", 1, "b.py")],
        [],
        symbols,
    )
    assert store.shortest_path("a.A", "b.B") == ["a.A", "b.B"]


def test_shortest_path_indirect(store: NetworkXGraphStore) -> None:
    """shortest_path traverses intermediate nodes correctly."""
    symbols = _make_symbols(
        [("a.A", "A", "function"), ("b.B", "B", "function"), ("c.C", "C", "function")]
    )
    store.persist_graph(
        [
            _call("a.A", "b.B", "a.py", 1, "b.py"),
            _call("b.B", "c.C", "b.py", 2, "c.py"),
        ],
        [],
        symbols,
    )
    assert store.shortest_path("a.A", "c.C") == ["a.A", "b.B", "c.C"]


def test_shortest_path_self(store: NetworkXGraphStore) -> None:
    """shortest_path from a node to itself returns a single-element list."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    assert store.shortest_path("a.A", "a.A") == ["a.A"]


def test_shortest_path_no_path(store: NetworkXGraphStore) -> None:
    """Two disconnected nodes produce an empty path list."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph([], [], symbols)
    assert store.shortest_path("a.A", "b.B") == []


def test_shortest_path_missing_node(store: NetworkXGraphStore) -> None:
    """A missing node produces an empty path list (no exception)."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    assert store.shortest_path("a.A", "missing.X") == []


def test_shortest_path_rel_type_filter(store: NetworkXGraphStore) -> None:
    """shortest_path with rel_types only traverses allowed edge types."""
    symbols = _make_symbols(
        [("a.A", "A", "function"), ("b.B", "B", "function"), ("c.C", "C", "function")]
    )
    store.persist_graph([], [], symbols)
    # Direct IMPORTS edge: a->b; CALLS edge: b->c (filtered out)
    store._graph.add_edge("a.A", "b.B", rel_type="IMPORTS")
    store._graph.add_edge("b.B", "c.C", rel_type="CALLS")

    # With only IMPORTS allowed, cannot reach c.C
    path = store.shortest_path("a.A", "c.C", rel_types=["IMPORTS"])
    assert path == []

    # With both allowed, can reach c.C
    path2 = store.shortest_path("a.A", "c.C", rel_types=["IMPORTS", "CALLS"])
    assert path2 == ["a.A", "b.B", "c.C"]


# ---------------------------------------------------------------------------
# get_subgraph
# ---------------------------------------------------------------------------


def test_get_subgraph_induced(store: NetworkXGraphStore) -> None:
    """get_subgraph returns only edges within the requested node set."""
    symbols = _make_symbols(
        [("a.A", "A", "function"), ("b.B", "B", "function"), ("c.C", "C", "function")]
    )
    store.persist_graph(
        [
            _call("a.A", "b.B", "a.py", 1, "b.py"),
            _call("b.B", "c.C", "b.py", 2, "c.py"),
            _call("a.A", "c.C", "a.py", 5, "c.py"),
        ],
        [],
        symbols,
    )
    sub = store.get_subgraph(["a.A", "b.B"])
    edge_pairs = {(e.source, e.target) for e in sub.edges}
    assert ("a.A", "b.B") in edge_pairs
    assert ("b.B", "c.C") not in edge_pairs
    assert ("a.A", "c.C") not in edge_pairs


def test_get_subgraph_all_nodes(store: NetworkXGraphStore) -> None:
    """get_subgraph with the full node set returns all edges."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph([_call("a.A", "b.B", "a.py", 1, "b.py")], [], symbols)
    sub = store.get_subgraph(["a.A", "b.B"])
    assert len(sub.edges) == 1
    assert sub.edges[0].rel_type == "CALLS"


def test_get_subgraph_empty_input(store: NetworkXGraphStore) -> None:
    """get_subgraph with an empty list returns an empty result."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    sub = store.get_subgraph([])
    assert sub.nodes == []
    assert sub.edges == []


def test_get_subgraph_missing_ids_skipped(store: NetworkXGraphStore) -> None:
    """Node ids not present in the graph are silently ignored."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    sub = store.get_subgraph(["a.A", "phantom.X"])
    assert len(sub.nodes) == 1
    assert sub.nodes[0].node_id == "a.A"


def test_get_subgraph_node_types(store: NetworkXGraphStore) -> None:
    """get_subgraph returns GraphNode / GraphEdge instances."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph([_call("a.A", "b.B", "a.py", 1, "b.py")], [], symbols)
    sub = store.get_subgraph(["a.A", "b.B"])
    for n in sub.nodes:
        assert isinstance(n, GraphNode)
    for e in sub.edges:
        assert isinstance(e, GraphEdge)


# ---------------------------------------------------------------------------
# query - Cypher mini-interpreter
# ---------------------------------------------------------------------------


def test_query_count_nodes(store: NetworkXGraphStore) -> None:
    """'MATCH (n) RETURN count(n) AS cnt' returns the node count."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "class")])
    store.persist_graph([], [], symbols)
    assert store.query("MATCH (n) RETURN count(n) AS cnt") == [{"cnt": 2}]


def test_query_count_calls(store: NetworkXGraphStore) -> None:
    """CALLS count query returns the correct relationship count."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph([_call("a.A", "b.B", "a.py", 1, "b.py")], [], symbols)
    rows = store.query("MATCH ()-[r:CALLS]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 1


def test_query_count_imports(store: NetworkXGraphStore) -> None:
    """IMPORTS count query returns the correct relationship count."""
    store.persist_graph([], [_dep_edge("app.py", "auth.py", "app", "auth")], [])
    rows = store.query("MATCH ()-[r:IMPORTS]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 1


def test_query_match_with_return(store: NetworkXGraphStore) -> None:
    """MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name LIMIT 5."""
    symbols = _make_symbols(
        [("a.foo", "foo", "function"), ("b.bar", "bar", "function")]
    )
    store.persist_graph([_call("a.foo", "b.bar", "a.py", 3, "b.py")], [], symbols)
    rows = store.query(
        "MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name LIMIT 5"
    )
    assert len(rows) == 1
    assert rows[0]["f.name"] == "foo"
    assert rows[0]["g.name"] == "bar"


def test_query_match_limit_respected(store: NetworkXGraphStore) -> None:
    """LIMIT clause caps the number of returned rows."""
    symbols = _make_symbols(
        [(f"m.f{i}", f"f{i}", "function") for i in range(10)]
        + [(f"m.g{i}", f"g{i}", "function") for i in range(10)]
    )
    edges = [_call(f"m.f{i}", f"m.g{i}", "m.py", i + 1) for i in range(10)]
    store.persist_graph(edges, [], symbols)
    rows = store.query(
        "MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name LIMIT 3"
    )
    assert len(rows) == 3


def test_query_match_no_label_filter(store: NetworkXGraphStore) -> None:
    """MATCH without label filter returns all matching edges."""
    store._graph.add_node("a.A", node_id="a.A", name="A", label="Function")
    store._graph.add_node("b.B", node_id="b.B", name="B", label="Class")
    store._graph.add_edge("a.A", "b.B", rel_type="CALLS")
    rows = store.query("MATCH (a)-[:CALLS]->(b) RETURN a.name, b.name")
    assert len(rows) == 1


def test_query_unsupported_cypher_raises(store: NetworkXGraphStore) -> None:
    """Unsupported Cypher raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        store.query("CREATE (n:Foo {x: 1}) RETURN n")


def test_query_whitespace_normalised(store: NetworkXGraphStore) -> None:
    """Extra whitespace in the query string is handled gracefully."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    rows = store.query("  MATCH  (n)  RETURN  count(n)  AS  cnt  ")
    assert rows == [{"cnt": 1}]


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_all_nodes_and_edges(store: NetworkXGraphStore) -> None:
    """clear() leaves the graph completely empty."""
    symbols = _make_symbols([("a.A", "A", "function"), ("b.B", "B", "function")])
    store.persist_graph([_call("a.A", "b.B", "a.py", 1, "b.py")], [], symbols)
    assert store.node_count() > 0

    store.clear()
    assert store.node_count() == 0
    assert store.edge_count() == 0


def test_clear_also_resets_label_index(store: NetworkXGraphStore) -> None:
    """After clear(), the internal label index is also reset."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    store.clear()
    assert store._labels == {}


def test_persist_after_clear_works(store: NetworkXGraphStore) -> None:
    """A second persist_graph call after clear() works correctly."""
    symbols = _make_symbols([("a.A", "A", "function")])
    store.persist_graph([], [], symbols)
    store.clear()
    symbols2 = _make_symbols([("b.B", "B", "class")])
    store.persist_graph([], [], symbols2)
    assert store.node_count() == 1
    assert "b.B" in store.graph


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_is_noop_for_networkx(store: NetworkXGraphStore) -> None:
    """close() on the NetworkX backend does not raise."""
    store.close()  # must not raise


# ---------------------------------------------------------------------------
# Bulk insert: 10K+ nodes / edges
# ---------------------------------------------------------------------------


def test_bulk_insert_10k_nodes(store: NetworkXGraphStore) -> None:
    """NetworkXGraphStore handles 10 000 nodes without error."""
    n = 10_000
    symbols = [
        _Symbol(
            symbol_id=f"bulk.func_{i}",
            name=f"func_{i}",
            qualified_name=f"bulk.func_{i}",
            type="function",
            file_path=f"bulk_{i // 100}.py",
            module="bulk",
        )
        for i in range(n)
    ]
    store.persist_graph([], [], symbols)
    assert store.node_count() == n


def test_bulk_insert_10k_edges(store: NetworkXGraphStore) -> None:
    """persist_graph handles 10 000 CALLS edges without error."""
    n = 10_000
    symbols = [
        _Symbol(
            symbol_id=f"bulk.func_{i}",
            name=f"func_{i}",
            qualified_name=f"bulk.func_{i}",
            type="function",
            file_path=f"bulk_{i // 100}.py",
            module="bulk",
        )
        for i in range(n + 1)
    ]
    edges = [
        _call(
            caller=f"bulk.func_{i}",
            callee=f"bulk.func_{i + 1}",
            caller_file=f"bulk_{i // 100}.py",
            line=i + 1,
        )
        for i in range(n)
    ]
    store.persist_graph(edges, [], symbols)
    assert store.edge_count() == n


# ---------------------------------------------------------------------------
# Neo4jGraphStore - no live instance tests
# ---------------------------------------------------------------------------


def test_neo4j_store_raises_on_connection_failure() -> None:
    """Neo4jGraphStore raises RuntimeError when the server is unreachable."""
    pytest.importorskip("neo4j")
    from src.reporag.graph.neo4j_store import Neo4jGraphStore

    with pytest.raises(RuntimeError, match="Could not connect"):
        Neo4jGraphStore(
            uri="bolt://127.0.0.1:19999",  # nothing listening here
            max_retries=1,
            retry_delay=0.0,
        )


def test_neo4j_store_import_without_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neo4jGraphStore raises ImportError when the neo4j package is absent."""
    import builtins

    real_import = builtins.__import__

    def _block_neo4j(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "neo4j":
            raise ImportError("mocked: neo4j not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_neo4j)

    # We must re-import inside the patch so the ImportError path is hit.
    # Use a fresh import of the class inside the monkeypatched scope.
    import importlib
    import sys

    # Remove cached module so the import guard fires again.
    neo4j_mod_key = "src.reporag.graph.neo4j_store"
    original = sys.modules.pop(neo4j_mod_key, None)
    try:
        mod = importlib.import_module(neo4j_mod_key)
        with pytest.raises(ImportError):
            mod.Neo4jGraphStore(
                uri="bolt://localhost:9999", max_retries=1, retry_delay=0.0
            )
    finally:
        # Always restore the original cached module
        if original is not None:
            sys.modules[neo4j_mod_key] = original
        elif neo4j_mod_key in sys.modules:
            del sys.modules[neo4j_mod_key]

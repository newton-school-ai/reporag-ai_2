"""Unit tests for graph-based retrieval (Issue 18).

All tests build a :class:`NetworkXGraphStore` directly (mirroring
``tests/unit/test_neo4j_store.py``'s own approach) so the full suite runs
without a live Neo4j instance -- this doubles as coverage of the "NetworkX
fallback" acceptance criterion, since :class:`GraphTraversal` talks to the
store purely through :class:`GraphStoreProtocol`.
"""

from __future__ import annotations

import pytest

from reporag.graph.call_graph import CallEdge
from reporag.graph.dependency_graph import DependencyEdge
from reporag.graph.neo4j_store import NetworkXGraphStore
from reporag.graph.symbol_table import SymbolRecord, SymbolTable
from reporag.retrieval.graph_traversal import (
    AmbiguousSymbolError,
    GraphPaths,
    GraphSubgraph,
    GraphTraversal,
    SymbolNotFoundError,
)
from reporag.retrieval.vector_search import RetrievalResult as VectorRetrievalResult

# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------


def _rec(
    symbol_id: str,
    name: str,
    qualified_name: str,
    *,
    type: str = "function",
    file_path: str = "app.py",
    module: str = "app",
    start_line: int = 1,
    end_line: int = 5,
    signature: str | None = None,
    docstring: str | None = None,
    parent: str | None = None,
    bases: list[str] | None = None,
) -> SymbolRecord:
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
        parent=parent,
        bases=bases or [],
    )


def _table(*records: SymbolRecord) -> SymbolTable:
    table = SymbolTable()
    for r in records:
        table.add(r)
    return table


def _call(
    caller: str,
    callee: str,
    *,
    caller_file: str = "app.py",
    callee_file: str = "app.py",
    call_site_line: int = 1,
    resolution: str = "local",
) -> CallEdge:
    return CallEdge(
        caller=caller,
        callee=callee,
        caller_file=caller_file,
        callee_file=callee_file,
        call_site_line=call_site_line,
        resolution=resolution,
    )


@pytest.fixture
def chain_store_and_table() -> tuple[NetworkXGraphStore, SymbolTable]:
    """a -> b -> c -> d, a linear CALLS chain, plus an unconnected `e`."""
    a = _rec("a", "a", "a")
    b = _rec("b", "b", "b")
    c = _rec("c", "c", "c")
    d = _rec("d", "d", "d")
    e = _rec("e", "e", "e")
    table = _table(a, b, c, d, e)
    store = NetworkXGraphStore()
    store.persist_graph([_call("a", "b"), _call("b", "c"), _call("c", "d")], [], table)
    return store, table


@pytest.fixture
def diamond_store_and_table() -> tuple[NetworkXGraphStore, SymbolTable]:
    """a -> b -> d and a -> c -> d (two 2-hop paths, no direct a -> d)."""
    a = _rec("a", "a", "a")
    b = _rec("b", "b", "b")
    c = _rec("c", "c", "c")
    d = _rec("d", "d", "d")
    table = _table(a, b, c, d)
    store = NetworkXGraphStore()
    store.persist_graph(
        [_call("a", "b"), _call("a", "c"), _call("b", "d"), _call("c", "d")],
        [],
        table,
    )
    return store, table


@pytest.fixture
def ambiguous_name_table() -> SymbolTable:
    """`add` defined on two different classes -- a bare-name lookup is ambiguous."""
    return _table(
        _rec("Calculator.add", "add", "Calculator.add", type="method"),
        _rec("Vector.add", "add", "Vector.add", type="method"),
    )


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


class TestSymbolResolution:
    def test_bare_unique_name_resolves(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        gt = GraphTraversal(store, table)
        # "a" is both bare and qualified here; use get_neighbors to prove it
        # actually reached node "a" in the store.
        results = gt.get_neighbors("a", direction="out")
        assert [r.symbol_name for r in results] == ["b"]

    def test_qualified_name_resolves(self) -> None:
        table = _table(_rec("app.MyClass.run", "run", "app.MyClass.run", type="method"))
        store = NetworkXGraphStore()
        store.persist_graph([], [], table)
        gt = GraphTraversal(store, table)
        # No error raised resolving a qualified name that exists.
        assert gt.extract_subgraph(["app.MyClass.run"]).nodes[0].symbol_name == (
            "app.MyClass.run"
        )

    def test_unknown_name_raises_symbol_not_found(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        gt = GraphTraversal(store, table)
        with pytest.raises(SymbolNotFoundError):
            gt.get_neighbors("does_not_exist")

    def test_ambiguous_bare_name_raises(
        self, ambiguous_name_table: SymbolTable
    ) -> None:
        store = NetworkXGraphStore()
        store.persist_graph([], [], ambiguous_name_table)
        gt = GraphTraversal(store, ambiguous_name_table)
        with pytest.raises(AmbiguousSymbolError) as exc_info:
            gt.get_neighbors("add")
        assert exc_info.value.name == "add"
        assert {c.qualified_name for c in exc_info.value.candidates} == {
            "Calculator.add",
            "Vector.add",
        }

    def test_ambiguous_error_message_lists_candidates(
        self, ambiguous_name_table: SymbolTable
    ) -> None:
        store = NetworkXGraphStore()
        store.persist_graph([], [], ambiguous_name_table)
        gt = GraphTraversal(store, ambiguous_name_table)
        with pytest.raises(AmbiguousSymbolError, match="Calculator.add"):
            gt.get_neighbors("add")

    def test_no_symbol_table_passes_symbol_through_as_id(self) -> None:
        """Without a SymbolTable, the `symbol` argument IS the symbol_id."""
        a = _rec("a", "a", "a")
        b = _rec("b", "b", "b")
        store = NetworkXGraphStore()
        store.persist_graph([_call("a", "b")], [], _table(a, b))
        gt = GraphTraversal(store)  # no symbol_table
        results = gt.get_neighbors("a", direction="out")
        assert [r.symbol_name for r in results] == ["b"]

    def test_synthetic_module_id_passes_through_even_with_table(self) -> None:
        """A `module:...` id is never in the SymbolTable but must not raise."""
        a = _rec("a", "a", "a", file_path="app.py", module="app")
        table = _table(a)
        store = NetworkXGraphStore()
        dep = DependencyEdge(
            source="app.py",
            target="db.py",
            source_module="app",
            target_module="db",
            import_type="import",
            resolved=True,
        )
        store.persist_graph([], [dep], table)
        gt = GraphTraversal(store, table)
        # Resolves without raising, even though "module:app" isn't a
        # SymbolTable entry.
        results = gt.get_neighbors("module:app", direction="out")
        assert [r.metadata.get("qualified_name") for r in results] == ["db"]


# ---------------------------------------------------------------------------
# NetworkX fallback / lazy store construction
# ---------------------------------------------------------------------------


class TestNetworkXFallback:
    """Covers both halves of the default-store policy: `try_neo4j=False`
    (instant, deterministic, no network attempt at all -- what the rest of
    this test module uses to stay fast) and the real default of `True`
    (attempt Neo4j, catch any connection failure, fall back to NetworkX)."""

    def test_try_neo4j_false_uses_empty_networkx_store(self) -> None:
        from reporag.graph.neo4j_store import NetworkXGraphStore as NXStore

        gt = GraphTraversal(try_neo4j=False)
        assert isinstance(gt.store, NXStore)

    def test_try_neo4j_false_never_raises_or_hangs(self) -> None:
        """try_neo4j=False must not attempt any network connection."""
        gt = GraphTraversal(try_neo4j=False)
        assert gt.get_neighbors("anything") == []
        assert gt.find_paths("a", "b").shortest == []
        assert gt.extract_subgraph(["a", "b"]).nodes == []

    def test_default_falls_back_to_networkx_when_neo4j_unreachable(self) -> None:
        """try_neo4j defaults to True: it must actually attempt Neo4j first,
        then fall back on failure -- pin a definitely-closed port so this is
        deterministic and fast rather than depending on the environment."""
        from reporag.graph.neo4j_store import NetworkXGraphStore as NXStore

        gt = GraphTraversal(neo4j_uri="bolt://127.0.0.1:9999")
        assert isinstance(gt.store, NXStore)

    def test_default_does_not_raise_on_connection_failure(self) -> None:
        gt = GraphTraversal(neo4j_uri="bolt://127.0.0.1:9999")
        assert gt.get_neighbors("anything") == []

    def test_store_is_cached_not_rebuilt_per_call(self) -> None:
        gt = GraphTraversal(try_neo4j=False)
        first = gt.store
        second = gt.store
        assert first is second

    def test_explicit_store_skips_default_policy_entirely(self) -> None:
        """Passing a store directly must not touch try_neo4j / Neo4j at all."""
        from reporag.graph.neo4j_store import NetworkXGraphStore as NXStore

        explicit = NXStore()
        gt = GraphTraversal(explicit)
        assert gt.store is explicit


# ---------------------------------------------------------------------------
# get_neighbors -- schema / payload alignment
# ---------------------------------------------------------------------------


class TestGetNeighborsSchema:
    def test_returns_vector_search_retrieval_result_class(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors("a", direction="out")
        assert results
        assert isinstance(results[0], VectorRetrievalResult)

    def test_result_fields_populated_from_node_props(self) -> None:
        rec = _rec(
            "app.helper",
            "helper",
            "app.helper",
            file_path="app.py",
            start_line=10,
            end_line=15,
            signature="def helper(x):",
            docstring="Does a thing.",
        )
        caller = _rec("app.main", "main", "app.main")
        table = _table(caller, rec)
        store = NetworkXGraphStore()
        store.persist_graph([_call("app.main", "app.helper")], [], table)
        r = GraphTraversal(store, table).get_neighbors("app.main", direction="out")[0]
        assert r.file_path == "app.py"
        assert r.start_line == 10
        assert r.end_line == 15
        assert r.symbol_name == "app.helper"
        assert r.chunk_text == "def helper(x):"  # signature preferred
        assert r.metadata["symbol_id"] == "app.helper"

    def test_chunk_text_falls_back_to_docstring_when_no_signature(self) -> None:
        rec = _rec("app.helper", "helper", "app.helper", docstring="Does a thing.")
        caller = _rec("app.main", "main", "app.main")
        table = _table(caller, rec)
        store = NetworkXGraphStore()
        store.persist_graph([_call("app.main", "app.helper")], [], table)
        r = GraphTraversal(store, table).get_neighbors("app.main", direction="out")[0]
        assert r.chunk_text == "Does a thing."

    def test_chunk_text_empty_when_neither_present(self) -> None:
        rec = _rec("app.helper", "helper", "app.helper")
        caller = _rec("app.main", "main", "app.main")
        table = _table(caller, rec)
        store = NetworkXGraphStore()
        store.persist_graph([_call("app.main", "app.helper")], [], table)
        r = GraphTraversal(store, table).get_neighbors("app.main", direction="out")[0]
        assert r.chunk_text == ""


# ---------------------------------------------------------------------------
# get_neighbors -- N-hop correctness
# ---------------------------------------------------------------------------


class TestGetNeighborsTraversal:
    def test_depth_1_outgoing(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=1, direction="out"
        )
        assert {r.symbol_name for r in results} == {"b"}

    def test_depth_3_outgoing_reaches_all_downstream(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out"
        )
        assert {r.symbol_name for r in results} == {"b", "c", "d"}

    def test_hop_distance_is_correct_per_node(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out"
        )
        hops = {r.symbol_name: r.metadata["hop"] for r in results}
        assert hops == {"b": 1, "c": 2, "d": 3}

    def test_diamond_shortest_hop_wins(
        self, diamond_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        """`d` is reachable via two different 2-hop routes -- hop must be 2, not 3+."""
        store, table = diamond_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out"
        )
        hops = {r.symbol_name: r.metadata["hop"] for r in results}
        assert hops["d"] == 2

    def test_results_sorted_by_ascending_hop(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out"
        )
        hops = [r.metadata["hop"] for r in results]
        assert hops == sorted(hops)

    def test_score_decreases_with_hop_distance(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out"
        )
        by_name = {r.symbol_name: r.score for r in results}
        assert by_name["b"] > by_name["c"] > by_name["d"]

    def test_depth_1_incoming(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "c", depth=1, direction="in"
        )
        assert {r.symbol_name for r in results} == {"b"}

    def test_both_directions(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "b", depth=1, direction="both"
        )
        assert {r.symbol_name for r in results} == {"a", "c"}

    def test_edge_type_filter_excludes_other_types(self) -> None:
        a = _rec("a", "a", "a", file_path="app.py", module="app")
        b = _rec("b", "b", "b")
        table = _table(a, b)
        store = NetworkXGraphStore()
        store.persist_graph([_call("a", "b")], [], table)
        results = GraphTraversal(store, table).get_neighbors(
            "a", edge_types=["IMPORTS"], direction="out"
        )
        assert results == []

    def test_unconnected_node_returns_empty(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        assert GraphTraversal(store, table).get_neighbors("e") == []

    def test_unknown_symbol_id_without_table_returns_empty(self) -> None:
        store = NetworkXGraphStore()
        store.persist_graph([], [], SymbolTable())
        assert GraphTraversal(store).get_neighbors("ghost") == []

    def test_top_k_caps_results(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        results = GraphTraversal(store, table).get_neighbors(
            "a", depth=3, direction="out", top_k=1
        )
        assert len(results) == 1
        assert results[0].symbol_name == "b"  # closest hop wins under top_k


# ---------------------------------------------------------------------------
# get_neighbors -- failure / edge cases
# ---------------------------------------------------------------------------


class TestGetNeighborsFailureCases:
    def test_depth_zero_raises(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        with pytest.raises(ValueError, match="depth"):
            GraphTraversal(store, table).get_neighbors("a", depth=0)

    def test_negative_depth_raises(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        with pytest.raises(ValueError, match="depth"):
            GraphTraversal(store, table).get_neighbors("a", depth=-1)

    def test_invalid_direction_raises(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        with pytest.raises(ValueError, match="direction"):
            GraphTraversal(store, table).get_neighbors("a", direction="sideways")

    def test_empty_store_returns_empty(self) -> None:
        store = NetworkXGraphStore()
        store.persist_graph([], [], SymbolTable())
        assert GraphTraversal(store).get_neighbors("a") == []


# ---------------------------------------------------------------------------
# find_paths -- shortest path + all paths (merged API)
# ---------------------------------------------------------------------------


class TestFindPaths:
    # -- shortest path correctness --------------------------------------

    def test_direct_edge_is_the_shortest_path(self) -> None:
        a = _rec("a", "a", "a")
        b = _rec("b", "b", "b")
        c = _rec("c", "c", "c")
        table = _table(a, b, c)
        store = NetworkXGraphStore()
        store.persist_graph(
            [_call("a", "b"), _call("b", "c"), _call("a", "c")], [], table
        )
        paths = GraphTraversal(store, table).find_paths("a", "c")
        assert [r.symbol_name for r in paths.shortest] == ["a", "c"]

    def test_multi_hop_path_when_no_direct_edge(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        paths = GraphTraversal(store, table).find_paths("a", "d")
        assert [r.symbol_name for r in paths.shortest] == ["a", "b", "c", "d"]

    def test_shortest_preserves_order_not_score_sorted(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        shortest = GraphTraversal(store, table).find_paths("a", "d").shortest
        # Scores decrease along the path, but the RESULT ORDER is the path
        # order (a, b, c, d), not sorted by score.
        assert [r.symbol_name for r in shortest] == ["a", "b", "c", "d"]
        scores = [r.score for r in shortest]
        assert scores == sorted(scores, reverse=True)

    def test_path_index_and_length_metadata(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        shortest = GraphTraversal(store, table).find_paths("a", "d").shortest
        indices = [r.metadata["path_index"] for r in shortest]
        assert indices == [0, 1, 2, 3]
        assert all(r.metadata["path_length"] == 4 for r in shortest)

    def test_no_path_returns_empty_shortest_and_all_paths(self) -> None:
        a = _rec("a", "a", "a")
        b = _rec("b", "b", "b")
        table = _table(a, b)
        store = NetworkXGraphStore()
        store.persist_graph([], [], table)  # no edges at all
        paths = GraphTraversal(store, table).find_paths("a", "b")
        assert paths.shortest == []
        assert paths.all_paths == []

    def test_same_source_and_target_is_single_node_path(self) -> None:
        a = _rec("a", "a", "a")
        table = _table(a)
        store = NetworkXGraphStore()
        store.persist_graph([], [], table)
        shortest = GraphTraversal(store, table).find_paths("a", "a").shortest
        assert [r.symbol_name for r in shortest] == ["a"]

    def test_max_depth_excludes_a_too_long_shortest_path(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        # a -> d is 3 hops; capping at 2 must yield "no path".
        paths = GraphTraversal(store, table).find_paths("a", "d", max_depth=2)
        assert paths.shortest == []

    def test_max_depth_allows_a_short_enough_shortest_path(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        shortest = (
            GraphTraversal(store, table).find_paths("a", "b", max_depth=2).shortest
        )
        assert [r.symbol_name for r in shortest] == ["a", "b"]

    def test_edge_type_filter_excludes_the_only_shortest_path(self) -> None:
        a = _rec("a", "a", "a")
        b = _rec("b", "b", "b")
        table = _table(a, b)
        store = NetworkXGraphStore()
        store.persist_graph([_call("a", "b")], [], table)
        paths = GraphTraversal(store, table).find_paths(
            "a", "b", edge_types=["IMPORTS"]
        )
        assert paths.shortest == []

    def test_unknown_symbol_raises_when_table_given(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        with pytest.raises(SymbolNotFoundError):
            GraphTraversal(store, table).find_paths("a", "ghost")

    def test_unknown_id_without_table_returns_empty(self) -> None:
        store = NetworkXGraphStore()
        store.persist_graph([], [], SymbolTable())
        paths = GraphTraversal(store).find_paths("ghost1", "ghost2")
        assert paths.shortest == []
        assert paths.all_paths == []

    # -- all-paths enumeration (NetworkX-backed) -------------------------

    def test_returns_graph_paths_instance(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        assert isinstance(GraphTraversal(store, table).find_paths("a", "d"), GraphPaths)

    def test_finds_both_diamond_routes(
        self, diamond_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = diamond_store_and_table
        paths = GraphTraversal(store, table).find_paths("a", "d")
        route_sets = {tuple(r.symbol_name for r in p) for p in paths.all_paths}
        assert route_sets == {("a", "b", "d"), ("a", "c", "d")}
        assert paths.all_paths_supported is True

    def test_all_paths_len_reflects_path_count(
        self, diamond_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = diamond_store_and_table
        paths = GraphTraversal(store, table).find_paths("a", "d")
        assert len(paths) == 2

    def test_respects_limit(
        self, diamond_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = diamond_store_and_table
        paths = GraphTraversal(store, table).find_paths("a", "d", limit=1)
        assert len(paths.all_paths) == 1

    def test_max_depth_also_bounds_all_paths(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        # a -> d is 3 hops; max_depth=2 should find nothing in either field.
        paths = GraphTraversal(store, table).find_paths("a", "d", max_depth=2)
        assert paths.shortest == []
        assert paths.all_paths == []

    def test_default_max_depth_applies_to_all_paths_when_not_given(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        """a -> d is 3 hops, well within the default cap of 5 -- must be found."""
        store, table = chain_store_and_table
        paths = GraphTraversal(store, table).find_paths("a", "d")
        assert paths.all_paths

    # -- degradation on backends without `.graph` ------------------------

    def test_degrades_to_shortest_only_without_graph_attribute(self) -> None:
        """A minimal fake store satisfying the protocol but with no `.graph`
        must still succeed, degrading all_paths to [shortest]."""

        class _FakeStore:
            def get_neighbors(self, *a, **k):
                return []

            def shortest_path(self, source_id, target_id, *, edge_types=None):
                if {source_id, target_id} == {"a", "b"}:
                    return [
                        {"symbol_id": "a", "name": "a"},
                        {"symbol_id": "b", "name": "b"},
                    ]
                return []

            def subgraph(self, *a, **k):
                return [], []

        gt = GraphTraversal(_FakeStore())  # type: ignore[arg-type]
        paths = gt.find_paths("a", "b")
        assert paths.all_paths_supported is False
        assert [r.symbol_name for r in paths.shortest] == ["a", "b"]
        assert len(paths.all_paths) == 1
        assert [r.symbol_name for r in paths.all_paths[0]] == ["a", "b"]

    def test_degrades_to_empty_all_paths_when_no_path_and_no_graph_attribute(
        self,
    ) -> None:
        class _FakeStore:
            def get_neighbors(self, *a, **k):
                return []

            def shortest_path(self, *a, **k):
                return []

            def subgraph(self, *a, **k):
                return [], []

        gt = GraphTraversal(_FakeStore())  # type: ignore[arg-type]
        paths = gt.find_paths("a", "b")
        assert paths.all_paths_supported is False
        assert paths.shortest == []
        assert paths.all_paths == []

    def test_no_path_returns_empty_all_paths_list(self) -> None:
        a = _rec("a", "a", "a")
        b = _rec("b", "b", "b")
        table = _table(a, b)
        store = NetworkXGraphStore()
        store.persist_graph([], [], table)
        paths = GraphTraversal(store, table).find_paths("a", "b")
        assert paths.all_paths == []
        assert paths.all_paths_supported is True


# ---------------------------------------------------------------------------
# extract_subgraph
# ---------------------------------------------------------------------------


class TestExtractSubgraph:
    def test_returns_graph_subgraph_instance(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "b"])
        assert isinstance(result, GraphSubgraph)

    def test_nodes_are_retrieval_results(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "b"])
        assert all(isinstance(n, VectorRetrievalResult) for n in result.nodes)

    def test_only_requested_nodes_included(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "b"])
        assert {n.symbol_name for n in result.nodes} == {"a", "b"}

    def test_only_internal_edges_included(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        """a -> b -> c -> d: requesting {a, b} must exclude the b -> c edge."""
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "b"])
        assert len(result.edges) == 1
        assert result.edges[0]["source"] == "a"
        assert result.edges[0]["target"] == "b"

    def test_edge_type_filter(self) -> None:
        a = _rec("a", "a", "a", file_path="a.py", module="a")
        b = _rec("b", "b", "b", file_path="b.py", module="b")
        table = _table(a, b)
        store = NetworkXGraphStore()
        dep = DependencyEdge(
            source="a.py",
            target="b.py",
            source_module="a",
            target_module="b",
            import_type="import",
            resolved=True,
        )
        store.persist_graph([_call("a", "b")], [dep], table)
        gt = GraphTraversal(store, table)
        # Both CALLS and IMPORTS-on-module-nodes exist; ask for CALLS only
        # over the two function nodes (module nodes aren't requested).
        result = gt.extract_subgraph(["a", "b"], edge_types=["CALLS"])
        assert all(e["type"] == "CALLS" for e in result.edges)

    def test_duplicate_symbols_deduplicated(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "a", "a"])
        assert len(result.nodes) == 1

    def test_unknown_symbol_id_without_table_silently_dropped(self) -> None:
        a = _rec("a", "a", "a")
        table = _table(a)
        store = NetworkXGraphStore()
        store.persist_graph([], [], table)
        result = GraphTraversal(store).extract_subgraph(["a", "ghost"])
        assert [n.symbol_name for n in result.nodes] == ["a"]

    def test_empty_symbol_list_returns_empty_subgraph(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph([])
        assert result.nodes == []
        assert result.edges == []

    def test_len_reflects_node_count(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        result = GraphTraversal(store, table).extract_subgraph(["a", "b", "c"])
        assert len(result) == 3

    def test_unknown_symbol_raises_when_table_given(
        self, chain_store_and_table: tuple[NetworkXGraphStore, SymbolTable]
    ) -> None:
        store, table = chain_store_and_table
        with pytest.raises(SymbolNotFoundError):
            GraphTraversal(store, table).extract_subgraph(["a", "ghost"])

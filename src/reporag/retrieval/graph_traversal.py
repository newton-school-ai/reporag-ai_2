"""Graph-based retrieval.

Uses the code knowledge graph for structural queries: N-hop neighbors,
shortest + all paths between symbols, and subgraph extraction. Converts
graph results to the common RetrievalResult schema.

Why
---
Some questions aren't about *what code looks like semantically* (vector
search) or *what identifiers appear* (BM25) -- they're about *structure*:
"what calls ``authenticate_user``?", "how does data flow from the HTTP
handler down to the database?", "show me everything connected to the
``PaymentProcessor`` class." Answering these requires walking the code
knowledge graph built by :mod:`reporag.graph.call_graph` (who calls what),
:mod:`reporag.graph.dependency_graph` (who imports what), and
:mod:`reporag.graph.symbol_table` (name -> definition), all persisted
through the :class:`~reporag.graph.neo4j_store.GraphStoreProtocol`
abstraction (Issue 12) that already gives us both a real Neo4j backend and
a pure-NetworkX fallback with an identical interface.

Design
------
:class:`GraphTraversal` is a thin query-side layer on top of an existing
:class:`~reporag.graph.neo4j_store.GraphStoreProtocol` store -- it does not
build the graph itself (that's Issues 9/10/12's job) and it does not
reimplement traversal (that already lives in
:class:`~reporag.graph.neo4j_store.NetworkXGraphStore` /
:class:`~reporag.graph.neo4j_store.Neo4jGraphStore`, so both backends stay
in lockstep automatically). This module adds exactly three things on top:

1. **Symbol-name resolution.** The store's own methods key everything by
   opaque ``symbol_id`` strings. Callers think in names
   (``"get_user_by_id"``, ``"Calculator.add"``), not ids, so
   :class:`GraphTraversal` accepts an optional
   :class:`~reporag.graph.symbol_table.SymbolTable` and resolves a bare or
   qualified name to its ``symbol_id`` before ever touching the store --
   raising a clear, actionable error (:class:`SymbolNotFoundError`,
   :class:`AmbiguousSymbolError`) instead of a silent empty result when a
   name doesn't exist or matches more than one definition.
2. **Hop-distance recovery.** ``GraphStoreProtocol.get_neighbors`` returns a
   flat *set* of reachable nodes for a given ``depth`` -- it doesn't say
   *which* hop each node was first reached at, because BFS layer
   information isn't part of the protocol's return type. :meth:`get_neighbors`
   recovers it by calling the store once per hop (1, 2, ..., depth) and
   taking the set difference between consecutive rings, so a caller asking
   for ``depth=3`` still gets to know "this one's a direct call, that one's
   two hops away."
3. **Common result schema.** Every method returns
   :class:`~reporag.retrieval.vector_search.RetrievalResult` (or a small
   container of them for paths/subgraphs), the exact same dataclass
   :class:`~reporag.retrieval.vector_search.VectorSearch` and
   :class:`~reporag.retrieval.bm25_search.BM25Search` return, so the future
   RRF fusion step (:mod:`reporag.retrieval.fusion`, Issue 19) can treat
   graph, vector, and lexical hits interchangeably.

Default store -- "try Neo4j, fall back to NetworkX"
-----------------------------------------------------
When no *store* is passed in, :class:`GraphTraversal` -- by default --
attempts to connect to Neo4j at ``settings.neo4j_uri`` and, on any
connection failure, transparently falls back to an empty
:class:`~reporag.graph.neo4j_store.NetworkXGraphStore` (logging a warning).
This mirrors :func:`~reporag.graph.neo4j_store.GraphStore`'s own
``fallback=True`` semantics, but with a deliberately tighter retry budget
(1 attempt, no inter-attempt sleep by default, both overridable via
``neo4j_max_retries`` / ``neo4j_retry_backoff``) -- ``Neo4jGraphStore``'s
own default of 3 retries with doubling backoff is tuned for a long-lived
ingestion pipeline, not a query-time helper that should fail fast. Pass
``try_neo4j=False`` to skip the connection attempt entirely and go
straight to NetworkX (useful for tests and any offline workflow), or pass
an already-constructed store (of either backend) to skip this logic
altogether.

Known limitation -- ``chunk_text``
-----------------------------------
The graph only stores *symbol metadata* (name, signature, docstring,
location) -- never the source-code body of a chunk; that lives in the
Qdrant / BM25 payloads built by :mod:`reporag.embedding.index_builder`.
:attr:`RetrievalResult.chunk_text` for a graph-derived result is therefore
best-effort: it falls back to the symbol's ``signature``, then its
``docstring``, then ``""``. Callers that need the actual code body should
cross-reference ``file_path`` / ``start_line`` / ``end_line`` against a
vector or BM25 result for the same symbol, or look the chunk up directly.

Known limitation -- traversal cost
-----------------------------------
:meth:`get_neighbors`'s per-hop-ring approach calls
``store.get_neighbors`` once per hop, and each of those calls re-runs a
BFS from scratch up to that hop count (see (2) above) -- so the total cost
is ``O(depth**2)`` rather than ``O(depth)``. This is a deliberate
trade-off: recovering accurate hop distances from a protocol that doesn't
expose them is worth a small constant-factor cost for the small depths
(1-3 hops) structural queries actually use; it is not intended for
very large *depth* values.

Known limitation -- "all paths" enumeration
---------------------------------------------
:meth:`find_paths` always returns the shortest path (every backend
supports that via ``GraphStoreProtocol.shortest_path``). *Enumerating
every simple path*, however, has no Neo4j-protocol equivalent as clean as
``shortestPath`` -- so it is only a real enumeration
(``all_paths_supported=True``) when the backend exposes its graph directly
for NetworkX-native traversal (currently
:class:`~reporag.graph.neo4j_store.NetworkXGraphStore`, via its public
``.graph`` property). On a backend without that (``Neo4jGraphStore``),
``find_paths`` still succeeds -- it just degrades ``all_paths`` to
``[shortest]`` (or ``[]`` if no path exists at all) rather than raising,
so calling code never has to special-case the backend just to get *a*
usable answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reporag.graph.neo4j_store import GraphStoreProtocol
    from reporag.graph.symbol_table import SymbolRecord, SymbolTable

from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Valid values for the `direction` argument shared by get_neighbors().
_VALID_DIRECTIONS = frozenset({"in", "out", "both"})

# Prefix neo4j_store._module_node_props() uses for synthetic per-file
# Module node ids -- these are never registered in a SymbolTable (which only
# indexes function/class/method/variable definitions), so a resolvable
# symbol string starting with this prefix is passed through as-is rather
# than treated as an unresolvable name. See _resolve()'s docstring.
_MODULE_ID_PREFIX = "module:"

# Cutoff used for find_paths()'s all-simple-paths enumeration when the
# caller doesn't supply max_depth -- unbounded simple-path search over a
# real call graph can be combinatorially explosive, so a default cap keeps
# it tractable.
_DEFAULT_ALL_PATHS_MAX_DEPTH = 5

# Default retry budget for GraphTraversal's own lazy "try Neo4j, fall back
# to NetworkX" default-store construction -- deliberately tighter than
# Neo4jGraphStore's own ingestion-pipeline-tuned defaults (3 retries,
# doubling backoff). See the module docstring's "Default store" section.
_DEFAULT_NEO4J_MAX_RETRIES = 1
_DEFAULT_NEO4J_RETRY_BACKOFF = 0.5


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SymbolNotFoundError(KeyError):
    """Raised when a symbol name has no matching definition in the SymbolTable."""


class AmbiguousSymbolError(ValueError):
    """Raised when a bare symbol name matches more than one definition.

    Attributes:
        name: The ambiguous bare name as given by the caller.
        candidates: Every :class:`~reporag.graph.symbol_table.SymbolRecord`
            that matched, so callers can present the choices (or pick a
            fully qualified name and retry).
    """

    def __init__(self, name: str, candidates: list[SymbolRecord]) -> None:
        self.name = name
        self.candidates = candidates
        qualified = ", ".join(c.qualified_name for c in candidates)
        super().__init__(
            f"Symbol name {name!r} is ambiguous; matches: {qualified}. "
            "Pass a fully qualified name instead."
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class GraphSubgraph:
    """The induced subgraph over a set of symbols.

    Unlike :meth:`GraphTraversal.get_neighbors` / :meth:`GraphTraversal.find_paths`,
    a subgraph is fundamentally a (nodes, edges) pair -- an edge has no
    natural representation as a single :class:`RetrievalResult`, so it is
    kept as a plain metadata dict (``source``, ``target``, ``type``, plus
    any backend-specific extras) rather than forced into that schema.

    Attributes:
        nodes: One :class:`RetrievalResult` per node in the subgraph.
        edges: One dict per edge connecting two nodes both present in
            *nodes* (edges leaving the requested symbol set are excluded --
            see :meth:`GraphTraversal.extract_subgraph`).
    """

    nodes: list[RetrievalResult] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of nodes in the subgraph."""
        return len(self.nodes)


@dataclass
class GraphPaths:
    """Result of :meth:`GraphTraversal.find_paths`: shortest path + all paths.

    Attributes:
        shortest: The shortest path between the two symbols, in traversal
            order (source first, target last). Empty when no path exists.
        all_paths: Every simple path within ``max_depth`` hops, each in
            traversal order, sorted shortest-first. When the backend can't
            truly enumerate all paths (see ``all_paths_supported``), this
            degrades to ``[shortest]`` (or ``[]`` if there's no path at
            all) so :meth:`~GraphTraversal.find_paths` never fails outright
            on any backend -- see the module docstring's "Known limitation
            -- 'all paths' enumeration" section.
        all_paths_supported: ``True`` when ``all_paths`` reflects a real
            enumeration; ``False`` when it's the degraded single-path
            fallback.
    """

    shortest: list[RetrievalResult] = field(default_factory=list)
    all_paths: list[list[RetrievalResult]] = field(default_factory=list)
    all_paths_supported: bool = False

    def __len__(self) -> int:
        """Return the number of paths in `all_paths`."""
        return len(self.all_paths)


# ---------------------------------------------------------------------------
# GraphTraversal
# ---------------------------------------------------------------------------


class GraphTraversal:
    """Performs structural graph queries over the code knowledge graph.

    Args:
        store: A pre-built :class:`~reporag.graph.neo4j_store.GraphStoreProtocol`
            (either :class:`~reporag.graph.neo4j_store.Neo4jGraphStore` or
            :class:`~reporag.graph.neo4j_store.NetworkXGraphStore`). When
            omitted, a store is constructed lazily on first use following
            the "try Neo4j, fall back to NetworkX" policy described in the
            module docstring's "Default store" section.
        symbol_table: An optional :class:`~reporag.graph.symbol_table.SymbolTable`
            used to resolve bare/qualified symbol *names* (``"get_user"``,
            ``"Calculator.add"``) to the ``symbol_id`` values the store
            actually keys on. When omitted, every ``symbol`` argument below
            is assumed to already be a literal ``symbol_id`` (or synthetic
            ``"module:..."`` id) and is passed straight through to the
            store -- useful when the caller already has ids from a prior
            query.
        try_neo4j: Whether the lazy default-store construction (only used
            when *store* is omitted) should attempt a Neo4j connection at
            all. ``True`` (default) tries Neo4j and falls back to NetworkX
            on failure; ``False`` skips the attempt and goes straight to
            an empty NetworkX store.
        neo4j_uri: Override ``settings.neo4j_uri`` for the default-store
            connection attempt. Ignored if *store* is given or
            ``try_neo4j=False``.
        neo4j_username: Override ``settings.neo4j_user``.
        neo4j_password: Override ``settings.neo4j_password``.
        neo4j_max_retries: Connection attempts before falling back. See the
            module docstring -- deliberately tighter than
            ``Neo4jGraphStore``'s own ingestion-tuned default.
        neo4j_retry_backoff: Initial retry backoff (seconds); doubles each
            attempt. Only matters when ``neo4j_max_retries > 1``.

    Raises:
        SymbolNotFoundError: From any method below, if *symbol_table* is
            given and a *symbol* argument matches no definition.
        AmbiguousSymbolError: From any method below, if *symbol_table* is
            given and a bare-name *symbol* argument matches more than one
            definition (e.g. ``"add"`` defined in two different classes).
    """

    def __init__(
        self,
        store: GraphStoreProtocol | None = None,
        symbol_table: SymbolTable | None = None,
        *,
        try_neo4j: bool = True,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        neo4j_max_retries: int = _DEFAULT_NEO4J_MAX_RETRIES,
        neo4j_retry_backoff: float = _DEFAULT_NEO4J_RETRY_BACKOFF,
    ) -> None:
        """Initialize GraphTraversal with an optional store and symbol table."""
        self._store = store
        self.symbol_table = symbol_table
        self._try_neo4j = try_neo4j
        self._neo4j_uri = neo4j_uri
        self._neo4j_username = neo4j_username
        self._neo4j_password = neo4j_password
        self._neo4j_max_retries = neo4j_max_retries
        self._neo4j_retry_backoff = neo4j_retry_backoff

    @property
    def store(self) -> GraphStoreProtocol:
        """Get the underlying graph store, constructing the default lazily."""
        if self._store is None:
            self._store = self._connect_default_store()
        return self._store

    def _connect_default_store(self) -> GraphStoreProtocol:
        """Implement the "try Neo4j, fall back to NetworkX" default-store policy."""
        from reporag.graph.neo4j_store import NetworkXGraphStore

        if not self._try_neo4j:
            logger.info(
                "GraphTraversal: try_neo4j=False, using an empty NetworkXGraphStore."
            )
            return NetworkXGraphStore()

        from reporag.config import settings
        from reporag.graph.neo4j_store import Neo4jGraphStore

        uri = self._neo4j_uri or settings.neo4j_uri
        username = self._neo4j_username or settings.neo4j_user
        password = self._neo4j_password or settings.neo4j_password.get_secret_value()
        try:
            store: GraphStoreProtocol = Neo4jGraphStore(
                uri,
                username=username,
                password=password,
                max_retries=self._neo4j_max_retries,
                retry_backoff=self._neo4j_retry_backoff,
            )
            logger.info("GraphTraversal: connected to Neo4j at %s", uri)
            return store
        except Exception as exc:  # noqa: BLE001 - any connection failure falls back
            logger.warning(
                "GraphTraversal: Neo4j unavailable at %s (%s); "
                "falling back to NetworkXGraphStore.",
                uri,
                exc,
            )
            return NetworkXGraphStore()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve(self, symbol: str) -> str:
        """Resolve a symbol name to a store ``symbol_id`` via the SymbolTable.

        Resolution order: exact fully-qualified name, then unique bare name.
        A synthetic ``"module:..."`` id (see the module docstring's "Known
        limitation" note) is passed through unresolved, since the
        SymbolTable never indexes module nodes.
        """
        if self.symbol_table is None:
            return symbol

        record = self.symbol_table.lookup_qualified(symbol)
        if record is not None:
            return record.symbol_id

        matches = self.symbol_table.lookup(symbol)
        if len(matches) == 1:
            return matches[0].symbol_id
        if len(matches) > 1:
            raise AmbiguousSymbolError(symbol, matches)

        if symbol.startswith(_MODULE_ID_PREFIX):
            return symbol

        raise SymbolNotFoundError(
            f"No symbol named {symbol!r} found in the symbol table."
        )

    @staticmethod
    def _node_to_result(
        node: dict[str, Any],
        *,
        score: float,
        extra: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Convert one graph node-property dict to a RetrievalResult.

        See the module docstring's "Known limitation -- chunk_text" note:
        graph nodes carry no source body, so ``chunk_text`` is best-effort.
        """
        metadata = dict(node)
        if extra:
            metadata.update(extra)
        return RetrievalResult(
            score=score,
            file_path=node.get("file_path") or "",
            start_line=node.get("start_line"),
            end_line=node.get("end_line"),
            symbol_name=node.get("qualified_name") or node.get("name"),
            chunk_text=node.get("signature") or node.get("docstring") or "",
            metadata=metadata,
        )

    def _shortest_path_results(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: list[str] | None,
        max_depth: int | None,
    ) -> list[RetrievalResult]:
        """Resolve the shortest path (already-resolved ids) into RetrievalResults."""
        path_nodes = self.store.shortest_path(
            source_id, target_id, edge_types=edge_types
        )
        if not path_nodes:
            return []

        hops = len(path_nodes) - 1
        if max_depth is not None and hops > max_depth:
            return []

        length = len(path_nodes)
        return [
            self._node_to_result(
                node,
                score=1.0 / (i + 1),
                extra={"path_index": i, "path_length": length},
            )
            for i, node in enumerate(path_nodes)
        ]

    def _all_paths_results(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: list[str] | None,
        max_depth: int,
        limit: int,
    ) -> list[list[RetrievalResult]]:
        """Enumerate simple paths (already-resolved ids) via the store's `.graph`.

        Only called when ``hasattr(self.store, "graph")`` -- see
        :meth:`find_paths`.
        """
        import networkx as nx

        graph = self.store.graph  # type: ignore[attr-defined]

        if source_id not in graph or target_id not in graph:
            return []

        if edge_types is not None:
            allowed = {
                (u, v)
                for u, v, d in graph.edges(data=True)
                if d.get("type") in edge_types
            }
            working = (
                graph.edge_subgraph(allowed).copy() if allowed else graph.__class__()
            )
            working.add_nodes_from(graph.nodes(data=True))
        else:
            working = graph

        undirected = working.to_undirected(as_view=True)
        try:
            raw_paths = nx.all_simple_paths(
                undirected, source=source_id, target=target_id, cutoff=max_depth
            )
        except nx.NodeNotFound:
            return []

        results: list[list[RetrievalResult]] = []
        for raw_path in raw_paths:
            length = len(raw_path)
            results.append(
                [
                    self._node_to_result(
                        dict(graph.nodes[nid]),
                        score=1.0 / (i + 1),
                        extra={"path_index": i, "path_length": length},
                    )
                    for i, nid in enumerate(raw_path)
                ]
            )
            if len(results) >= limit:
                break

        results.sort(key=len)
        return results

    # ------------------------------------------------------------------
    # N-hop neighbor query
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        symbol: str,
        *,
        depth: int = 1,
        edge_types: list[str] | None = None,
        direction: str = "both",
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return nodes reachable from *symbol* within *depth* hops.

        Args:
            symbol: Symbol name (resolved via ``symbol_table``) or a literal
                ``symbol_id`` when no ``symbol_table`` was given.
            depth: Maximum number of hops. Must be ``>= 1``.
            edge_types: Restrict traversal to these relationship labels
                (e.g. ``["CALLS"]``). ``None`` follows every edge type.
            direction: ``"out"`` (callees/dependents), ``"in"`` (callers/
                dependencies), or ``"both"``.
            top_k: Cap the number of returned results. ``None`` returns
                every reachable node.

        Returns:
            Results sorted by ascending hop distance (closer neighbors
            first; ties broken by symbol name for determinism), each
            carrying ``score = 1.0 / hop`` and ``metadata["hop"]`` with the
            exact hop distance. An unknown *symbol* (no node in the store)
            returns ``[]`` rather than raising -- the store itself makes no
            distinction between "node absent" and "node has no neighbors."

        Raises:
            ValueError: If ``depth < 1`` or *direction* is not one of
                ``"in"``, ``"out"``, ``"both"``.
            SymbolNotFoundError: See class docstring.
            AmbiguousSymbolError: See class docstring.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
            )

        node_id = self._resolve(symbol)

        hop_by_id: dict[str, int] = {}
        node_by_id: dict[str, dict[str, Any]] = {}
        for hop in range(1, depth + 1):
            ring = self.store.get_neighbors(
                node_id, edge_types=edge_types, depth=hop, direction=direction
            )
            for node in ring:
                nid = node.get("symbol_id")
                if nid is None or nid in hop_by_id:
                    continue
                hop_by_id[nid] = hop
                node_by_id[nid] = node

        results = [
            self._node_to_result(
                node_by_id[nid],
                score=1.0 / hop,
                extra={"hop": hop, "relation": "neighbor"},
            )
            for nid, hop in hop_by_id.items()
        ]
        results.sort(key=lambda r: (r.metadata["hop"], r.symbol_name or ""))

        if top_k is not None:
            results = results[:top_k]
        return results

    # ------------------------------------------------------------------
    # Path query between two symbols (shortest + all)
    # ------------------------------------------------------------------

    def find_paths(
        self,
        source: str,
        target: str,
        *,
        edge_types: list[str] | None = None,
        max_depth: int | None = None,
        limit: int = 10,
    ) -> GraphPaths:
        """Return the shortest path and (when supported) every simple path.

        Args:
            source: Start symbol name (or literal ``symbol_id``).
            target: End symbol name (or literal ``symbol_id``).
            edge_types: Restrict traversal to these relationship types.
                ``None`` allows any edge type.
            max_depth: Bounds both parts of the result: the shortest path
                is discarded (treated as "no path") if it's longer than
                this many hops, and all-paths enumeration is cut off at
                this many hops. Defaults to
                :data:`_DEFAULT_ALL_PATHS_MAX_DEPTH` (5) when ``None`` --
                unbounded simple-path search can be combinatorially
                explosive, so *some* cap always applies to ``all_paths``
                even if the shortest-path check itself stays unbounded.
            limit: Maximum number of paths in ``all_paths`` (paths are
                generated shortest-first by length).

        Returns:
            A :class:`GraphPaths`. Each path (both ``shortest`` and every
            entry in ``all_paths``) is in **traversal order** (source
            first, target last) -- never re-sorted by score, since
            reordering would destroy the path's meaning. Each node still
            carries a ``score`` (``1.0 / (position + 1)``) purely so a
            downstream fusion step can rank path nodes against other
            retrieval methods; ``metadata["path_index"]`` /
            ``metadata["path_length"]`` carry the same information
            unambiguously. See :class:`GraphPaths` for the "all paths
            unsupported on this backend" degradation behavior.

        Raises:
            SymbolNotFoundError: See class docstring.
            AmbiguousSymbolError: See class docstring.
        """
        source_id = self._resolve(source)
        target_id = self._resolve(target)
        all_paths_cap = (
            max_depth if max_depth is not None else _DEFAULT_ALL_PATHS_MAX_DEPTH
        )

        shortest = self._shortest_path_results(
            source_id, target_id, edge_types=edge_types, max_depth=max_depth
        )

        if hasattr(self.store, "graph"):
            all_paths = self._all_paths_results(
                source_id,
                target_id,
                edge_types=edge_types,
                max_depth=all_paths_cap,
                limit=limit,
            )
            supported = True
        else:
            logger.debug(
                "%s has no `.graph` attribute; find_paths() degrades "
                "all_paths to [shortest] instead of enumerating.",
                type(self.store).__name__,
            )
            all_paths = [shortest] if shortest else []
            supported = False

        return GraphPaths(
            shortest=shortest, all_paths=all_paths, all_paths_supported=supported
        )

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def extract_subgraph(
        self,
        symbols: list[str],
        *,
        edge_types: list[str] | None = None,
    ) -> GraphSubgraph:
        """Return the induced subgraph over *symbols*.

        Args:
            symbols: Symbol names (or literal ``symbol_id`` values) to
                include. Duplicates are removed, preserving first
                occurrence order.
            edge_types: Keep only edges whose ``type`` is in this list.
                ``None`` keeps every edge. This is a post-hoc filter (the
                underlying protocol's ``subgraph`` has no edge-type
                parameter), applied after extraction.

        Returns:
            A :class:`GraphSubgraph` with one :class:`RetrievalResult` per
            resolved node found in the store (unresolvable/unknown ids are
            silently dropped, matching the underlying store's own
            behavior) and the edges connecting them.

        Raises:
            SymbolNotFoundError: See class docstring.
            AmbiguousSymbolError: See class docstring.
        """
        node_ids: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            nid = self._resolve(symbol)
            if nid not in seen:
                seen.add(nid)
                node_ids.append(nid)

        nodes, edges = self.store.subgraph(node_ids)

        if edge_types is not None:
            edges = [e for e in edges if e.get("type") in edge_types]

        results = [
            self._node_to_result(node, score=1.0, extra={"relation": "subgraph_member"})
            for node in nodes
        ]
        return GraphSubgraph(nodes=results, edges=edges)

"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j and exposes Cypher query helpers for neighbors, shortest
path, and subgraph extraction.  A pure-NetworkX fallback implements the
same :class:`GraphStoreProtocol` so the full test suite runs without a
running Neo4j instance.

Architecture
------------
Three concrete classes share one common protocol:

* :class:`GraphStoreProtocol` -- the structural ``Protocol`` every backend
  must satisfy.  Callers should type-hint against this.
* :class:`Neo4jGraphStore` -- wraps the official ``neo4j`` Python driver.
  Uses ``MERGE`` for idempotent ingestion, batched transactions for 10 K+
  node scale, and simple exponential retry on transient connection errors.
* :class:`NetworkXGraphStore` -- pure-NetworkX ``DiGraph`` store.  Suitable
  for unit tests, CI pipelines, and local debugging where starting a Neo4j
  container is inconvenient.

Factory
-------
:func:`GraphStore` auto-selects the right backend::

    from src.reporag.graph.neo4j_store import GraphStore

    # Auto-connects to Neo4j; falls back to NetworkX on failure:
    store = GraphStore(uri="bolt://localhost:7687", fallback=True)

    # Always use NetworkX (good for tests):
    store = GraphStore(fallback=True)

Node labels
-----------
=====================  ======================================
SymbolRecord.type      Neo4j label (also NetworkX node attr)
=====================  ======================================
``"function"``         ``Function``
``"method"``           ``Function``  (``is_method=True``)
``"class"``            ``Class``
``"variable"``         ``Symbol``
*(any other)*          ``Symbol``
*(synthetic)*          ``Module``   (one per distinct file)
=====================  ======================================

Edge types
----------
================  =============  ==============================
Relationship      Label          Source data
================  =============  ==============================
call_graph        ``CALLS``      ``CallEdge``
dependency_graph  ``IMPORTS``    ``DependencyEdge``
class bases       ``INHERITS``   derived from ``SymbolRecord``
parent chain      ``CONTAINS``   derived from ``SymbolRecord``
================  =============  ==============================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import networkx as nx

if TYPE_CHECKING:
    # Heavy import only needed for type-checking, not at module load time.
    from neo4j import Driver

from reporag.graph.call_graph import CallEdge
from reporag.graph.dependency_graph import DependencyEdge
from reporag.graph.symbol_table import SymbolRecord, SymbolTable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default batch size for bulk MERGE transactions.
_BATCH_SIZE = 500

#: Number of times to retry a failed Neo4j connection before giving up.
_CONNECT_RETRIES = 3

#: Seconds to wait between connection retry attempts (doubles each attempt).
_RETRY_BACKOFF = 2.0

#: Map from SymbolRecord.type to Neo4j node label.
_LABEL_MAP: dict[str, str] = {
    "function": "Function",
    "method": "Function",
    "class": "Class",
}

#: Fallback label for any type not in _LABEL_MAP.
_DEFAULT_LABEL = "Symbol"

#: Label for synthetic per-file module nodes.
_MODULE_LABEL = "Module"

#: Common base label applied to every code node for easy global queries.
_BASE_LABEL = "_Node"


def _label_for(record: SymbolRecord) -> str:
    """Return the primary Neo4j label for a symbol record."""
    return _LABEL_MAP.get(record.type, _DEFAULT_LABEL)


# ---------------------------------------------------------------------------
# GraphQueryResult -- thin wrapper around raw query results
# ---------------------------------------------------------------------------


@dataclass
class GraphQueryResult:
    """Result of a :meth:`GraphStoreProtocol.query` call.

    Attributes:
        records: A list of ``dict[str, Any]`` rows, one per result row.
            Each value is whatever the driver (or NetworkX emulation)
            returned -- string, int, nested dict, etc.
        summary: Optional human-readable summary line (e.g. counters from
            Neo4j, or a placeholder for NetworkX).
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):  # type: ignore[override]
        return iter(self.records)

    def __repr__(self) -> str:
        return f"GraphQueryResult({len(self.records)} rows)"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Common interface shared by Neo4j and NetworkX backends.

    All methods are synchronous.  Implementations are expected to be
    constructed once per pipeline run and then treated as a long-lived
    resource (connect -> persist -> query -> close).
    """

    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ) -> None:
        """Ingest the full code knowledge graph.

        Creates (or updates) all nodes and edges in a single bulk pass.
        Idempotent: re-running with the same data produces the same graph.

        Args:
            call_edges: Directed function-call edges from
                :class:`~src.reporag.graph.call_graph.CallGraphBuilder`.
            dep_edges: Directed module-import edges from
                :class:`~src.reporag.graph.dependency_graph.DependencyGraphBuilder`.
            symbol_table: The repository-global symbol registry from
                :class:`~src.reporag.graph.symbol_table.SymbolTable`.
        """
        ...  # pragma: no cover

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> GraphQueryResult:
        """Execute a raw Cypher query and return all result rows.

        Args:
            cypher: A Cypher query string (Neo4j) or a plain key that the
                NetworkX backend may interpret as a named query.
            params: Optional query parameters keyed by name.

        Returns:
            A :class:`GraphQueryResult` wrapping the list of row dicts.

        Raises:
            NotImplementedError: When the backend does not support arbitrary
                Cypher (e.g. :class:`NetworkXGraphStore`).
        """
        ...  # pragma: no cover

    def get_neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        depth: int = 1,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return nodes reachable from *node_id* within *depth* hops.

        Args:
            node_id: The ``symbol_id`` (or module dotted name) of the start
                node.
            edge_types: Restrict traversal to these relationship labels
                (e.g. ``["CALLS"]``).  ``None`` means all types.
            depth: Maximum number of hops (default 1).
            direction: One of ``"out"`` (follow outgoing edges only),
                ``"in"`` (follow incoming edges only), or ``"both"``.

        Returns:
            A list of node-property dicts, one per reachable neighbor.
        """
        ...  # pragma: no cover

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the shortest path between two nodes as a sequence of nodes.

        Args:
            source_id: ``symbol_id`` of the start node.
            target_id: ``symbol_id`` of the end node.
            edge_types: Restrict traversal to these relationship types.
                ``None`` means all types.

        Returns:
            Ordered list of node-property dicts from *source_id* to
            *target_id*, inclusive.  Empty list when no path exists.
        """
        ...  # pragma: no cover

    def subgraph(
        self,
        node_ids: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract the induced subgraph over *node_ids*.

        Args:
            node_ids: ``symbol_id`` values of the nodes to include.

        Returns:
            A ``(nodes, edges)`` tuple where *nodes* is a list of
            node-property dicts and *edges* is a list of edge-property dicts
            (each containing ``source``, ``target``, ``type``, plus any
            extra attributes).
        """
        ...  # pragma: no cover

    def clear(self) -> None:
        """Delete all nodes and relationships from the store."""
        ...  # pragma: no cover

    def close(self) -> None:
        """Release driver / connection resources."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers shared by both backends
# ---------------------------------------------------------------------------


def _symbol_to_node_props(record: SymbolRecord) -> dict[str, Any]:
    """Convert a SymbolRecord to a flat dict of Neo4j / NetworkX properties."""
    return {
        "symbol_id": record.symbol_id,
        "name": record.name,
        "qualified_name": record.qualified_name,
        "type": record.type,
        "label": _label_for(record),
        "file_path": record.file_path,
        "module": record.module,
        "start_line": record.start_line,
        "end_line": record.end_line,
        "signature": record.signature or "",
        "docstring": record.docstring or "",
        "parent": record.parent or "",
        "is_async": record.is_async,
        "is_method": record.type == "method",
        "language": record.language,
        "bases": "|".join(record.bases),  # lists -> pipe-separated for Neo4j props
        "decorators": "|".join(record.decorators),
    }


def _module_node_props(file_path: str, module: str) -> dict[str, Any]:
    """Build a synthetic Module node property dict for a given file."""
    symbol_id = f"module:{module}" if module else f"module:{file_path}"
    return {
        "symbol_id": symbol_id,
        "name": module.split(".")[-1] if module else file_path,
        "qualified_name": module,
        "type": "module",
        "label": _MODULE_LABEL,
        "file_path": file_path,
        "module": module,
        "start_line": 1,
        "end_line": 0,
    }


def _build_synthetic_edges(
    symbol_table: SymbolTable,
) -> tuple[
    list[tuple[str, str, dict[str, Any]]], list[tuple[str, str, dict[str, Any]]]
]:
    """Derive INHERITS and CONTAINS edges directly from the symbol table.

    Returns:
        (inherits_edges, contains_edges) where each element is a list of
        ``(source_id, target_id, props)`` triples.
    """
    inherits: list[tuple[str, str, dict[str, Any]]] = []
    contains: list[tuple[str, str, dict[str, Any]]] = []

    for record in symbol_table:
        # CONTAINS: parent -> child
        if record.parent:
            parent_rec = symbol_table.lookup_qualified(record.parent)
            if parent_rec is not None:
                contains.append(
                    (
                        parent_rec.symbol_id,
                        record.symbol_id,
                        {"type": "CONTAINS"},
                    )
                )

        # INHERITS: class -> base class
        if record.type == "class" and record.bases:
            for base_name in record.bases:
                # bases may be simple names or dotted; try both.
                target = symbol_table.lookup_qualified(base_name)
                if target is None:
                    matches = symbol_table.lookup(base_name)
                    target = matches[0] if matches else None
                if target is not None and target.type == "class":
                    inherits.append(
                        (
                            record.symbol_id,
                            target.symbol_id,
                            {"type": "INHERITS", "base_name": base_name},
                        )
                    )

    return inherits, contains


def _chunk(items: list[Any], size: int):  # type: ignore[type-arg]
    """Yield successive *size*-length chunks from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# NetworkXGraphStore
# ---------------------------------------------------------------------------


class NetworkXGraphStore:
    """Pure-NetworkX graph store implementing :class:`GraphStoreProtocol`.

    - No external services required -- ideal for unit tests and CI.
    - Stores all node/edge properties as NetworkX attribute dicts.
    - :meth:`query` raises :class:`NotImplementedError` because Cypher is
      Neo4j-specific; all other helpers work fully.

    The internal ``DiGraph`` uses ``symbol_id`` as the node key.  This
    matches the key used by :class:`Neo4jGraphStore`, so the same helper
    code works for both.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, **attrs: Any) -> None:
        """Merge a node -- updates attributes if already present."""
        if self._g.has_node(node_id):
            self._g.nodes[node_id].update(attrs)
        else:
            self._g.add_node(node_id, **attrs)

    def _add_edge(self, src: str, tgt: str, **attrs: Any) -> None:
        """Merge an edge -- updates attributes if already present."""
        if self._g.has_edge(src, tgt):
            self._g.edges[src, tgt].update(attrs)
        else:
            self._g.add_edge(src, tgt, **attrs)

    # ------------------------------------------------------------------
    # GraphStoreProtocol implementation
    # ------------------------------------------------------------------

    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ) -> None:
        """Ingest all nodes and edges into the NetworkX DiGraph."""
        # --- Symbol nodes ---
        seen_modules: dict[str, dict[str, Any]] = {}
        for record in symbol_table:
            props = _symbol_to_node_props(record)
            self._add_node(record.symbol_id, **props)
            # Collect synthetic module nodes while iterating.
            if record.file_path and record.module not in seen_modules:
                seen_modules[record.module] = _module_node_props(
                    record.file_path, record.module
                )

        # --- Synthetic Module nodes ---
        for mod_props in seen_modules.values():
            self._add_node(mod_props["symbol_id"], **mod_props)

        # --- CALLS edges ---
        for edge in call_edges:
            if not edge.resolved:
                continue
            caller_rec = symbol_table.lookup_qualified(edge.caller)
            callee_rec = symbol_table.lookup_qualified(edge.callee)
            if caller_rec is None or callee_rec is None:
                # Try bare-name lookup as a fallback.
                if caller_rec is None:
                    matches = symbol_table.lookup(edge.caller)
                    caller_rec = matches[0] if matches else None
                if callee_rec is None:
                    matches = symbol_table.lookup(edge.callee)
                    callee_rec = matches[0] if matches else None
            if caller_rec is None or callee_rec is None:
                continue
            self._add_edge(
                caller_rec.symbol_id,
                callee_rec.symbol_id,
                type="CALLS",
                call_type=edge.call_type,
                resolution=edge.resolution,
                call_site_line=edge.call_site_line,
                is_recursive=edge.is_recursive,
            )

        # --- IMPORTS edges (Module -> Module) ---
        for dep in dep_edges:
            if not dep.resolved:
                continue
            src_mod = dep.source_module
            tgt_mod = dep.target_module.lstrip(".")  # strip relative dots
            src_id = f"module:{src_mod}"
            tgt_id = f"module:{tgt_mod}"
            # Ensure nodes exist even if the file had no symbols.
            if not self._g.has_node(src_id):
                self._add_node(src_id, **_module_node_props(dep.source, src_mod))
            if not self._g.has_node(tgt_id):
                self._add_node(tgt_id, **_module_node_props(dep.target, tgt_mod))
            self._add_edge(
                src_id,
                tgt_id,
                type="IMPORTS",
                import_type=dep.import_type,
                line=dep.line,
                is_relative=dep.is_relative,
                is_wildcard=dep.is_wildcard,
            )

        # --- INHERITS and CONTAINS edges ---
        inherits, contains = _build_synthetic_edges(symbol_table)
        for src_id, tgt_id, props in inherits:
            self._add_edge(src_id, tgt_id, **props)
        for src_id, tgt_id, props in contains:
            self._add_edge(src_id, tgt_id, **props)

        logger.debug(
            "NetworkXGraphStore: %d nodes, %d edges",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> GraphQueryResult:
        """Cypher is not supported by the NetworkX backend.

        Raises:
            NotImplementedError: Always.  Use :meth:`get_neighbors`,
                :meth:`shortest_path`, or :meth:`subgraph` instead.
        """
        raise NotImplementedError(
            "NetworkXGraphStore does not support Cypher queries. "
            "Use get_neighbors(), shortest_path(), or subgraph() instead, "
            "or switch to Neo4jGraphStore for Cypher support."
        )

    def get_neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        depth: int = 1,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return neighbor node-property dicts within *depth* hops.

        Args:
            node_id: The start node's ``symbol_id``.
            edge_types: Relationship labels to follow (``None`` = all).
            depth: Maximum hops.
            direction: ``"out"``, ``"in"``, or ``"both"``.

        Returns:
            List of node-property dicts for each reachable neighbor
            (excluding the start node itself).
        """
        if node_id not in self._g:
            return []

        if direction == "out":
            graph = self._g
        elif direction == "in":
            graph = self._g.reverse(copy=False)
        else:
            graph = self._g.to_undirected(as_view=True)

        # BFS up to *depth* hops, optionally filtered by edge type.
        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for nbr in graph.neighbors(nid):
                    if nbr in visited:
                        continue
                    # Filter by edge type when requested.
                    if edge_types is not None:
                        if direction == "in":
                            # reversed graph: original edge was nbr -> nid
                            edge_data = self._g.edges.get((nbr, nid), {})
                        elif direction == "out":
                            edge_data = self._g.edges.get((nid, nbr), {})
                        else:
                            edge_data = self._g.edges.get(
                                (nid, nbr), self._g.edges.get((nbr, nid), {})
                            )
                        if edge_data.get("type") not in edge_types:
                            continue
                    next_frontier.add(nbr)
                    visited.add(nbr)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(node_id)
        return [dict(self._g.nodes[n]) for n in visited if self._g.has_node(n)]

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the shortest path from *source_id* to *target_id*.

        Uses an undirected view so edges are traversable in both directions,
        matching Neo4j's ``shortestPath`` default behaviour.

        Args:
            source_id: Start node ``symbol_id``.
            target_id: End node ``symbol_id``.
            edge_types: When provided, only edges whose ``type`` attr is in
                this list may be traversed.

        Returns:
            Ordered list of node-property dicts from source to target.
            Empty list when no path exists or either node is unknown.
        """
        if source_id not in self._g or target_id not in self._g:
            return []

        if edge_types is not None:
            # Build a filtered view containing only matching edge types.
            allowed_edges = [
                (u, v)
                for u, v, d in self._g.edges(data=True)
                if d.get("type") in edge_types
            ]
            filtered = nx.DiGraph()
            filtered.add_nodes_from(self._g.nodes(data=True))
            filtered.add_edges_from(
                (u, v, self._g.edges[u, v]) for u, v in allowed_edges
            )
            graph: nx.Graph = filtered.to_undirected(as_view=False)
        else:
            graph = self._g.to_undirected(as_view=True)

        try:
            path_nodes = nx.shortest_path(graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return [dict(self._g.nodes[n]) for n in path_nodes if self._g.has_node(n)]

    def subgraph(
        self,
        node_ids: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the induced subgraph over *node_ids*.

        Args:
            node_ids: ``symbol_id`` values of the nodes to include.

        Returns:
            ``(nodes, edges)`` where *nodes* is a list of node-property dicts
            and *edges* is a list of edge-property dicts (each with ``source``
            and ``target`` keys prepended).
        """
        id_set = set(node_ids)
        nodes = [dict(self._g.nodes[n]) for n in node_ids if self._g.has_node(n)]
        edges = []
        for u, v, data in self._g.edges(data=True):
            if u in id_set and v in id_set:
                edge_props = {"source": u, "target": v}
                edge_props.update(data)
                edges.append(edge_props)
        return nodes, edges

    def clear(self) -> None:
        """Wipe all nodes and edges from the in-memory graph."""
        self._g.clear()

    def close(self) -> None:
        """No-op for the NetworkX backend."""
        pass

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying :class:`networkx.DiGraph` for direct use."""
        return self._g

    def __repr__(self) -> str:
        return (
            f"NetworkXGraphStore("
            f"{self._g.number_of_nodes()} nodes, "
            f"{self._g.number_of_edges()} edges)"
        )


# ---------------------------------------------------------------------------
# Neo4jGraphStore
# ---------------------------------------------------------------------------


# Import lazily so that projects without neo4j installed can still use the
# NetworkX fallback.
def _import_neo4j():  # type: ignore[return]
    """Import the neo4j driver, raising ImportError with a friendly message."""
    try:
        import neo4j  # noqa: F401

        return neo4j
    except ImportError as exc:
        raise ImportError(
            "The 'neo4j' package is not installed. "
            "Install it with: pip install neo4j>=5.0"
        ) from exc


class Neo4jGraphStore:
    """Neo4j-backed graph store implementing :class:`GraphStoreProtocol`.

    Connects to Neo4j via the official Bolt/HTTP driver, creates uniqueness
    constraints, and ingests the knowledge graph in batched transactions so
    10 K+ nodes can be handled efficiently.

    Args:
        uri: Bolt URI of the Neo4j instance, e.g. ``"bolt://localhost:7687"``.
        username: Neo4j username (default ``"neo4j"``).
        password: Neo4j password (default ``"reporag123"``).
        database: Target database name (default ``"neo4j"``).
        batch_size: Number of nodes/edges to write per transaction batch.
        max_retries: How many times to retry a failed connection.
        retry_backoff: Initial wait (seconds) between retries; doubles each
            attempt.
    """

    def __init__(
        self,
        uri: str,
        *,
        username: str = "neo4j",
        password: str = "reporag123",
        database: str = "neo4j",
        batch_size: int = _BATCH_SIZE,
        max_retries: int = _CONNECT_RETRIES,
        retry_backoff: float = _RETRY_BACKOFF,
    ) -> None:
        self._uri = uri
        self._database = database
        self._batch_size = batch_size
        self._driver: Driver | None = None

        neo4j = _import_neo4j()
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
                # Verify connectivity with a cheap ping.
                driver.verify_connectivity()
                self._driver = driver
                logger.info("Connected to Neo4j at %s (attempt %d)", uri, attempt)
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Neo4j connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_backoff * (2 ** (attempt - 1)))

        if self._driver is None:
            raise ConnectionError(
                f"Could not connect to Neo4j at {uri} after {max_retries} attempts. "
                f"Last error: {last_exc}"
            ) from last_exc

        self._ensure_constraints()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints (idempotent via IF NOT EXISTS)."""
        assert self._driver is not None
        labels = ["Function", "Class", "Module", "Symbol", "_Node"]
        with self._driver.session(database=self._database) as session:
            for label in labels:
                try:
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.symbol_id IS UNIQUE"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not create constraint for %s: %s", label, exc)

    # ------------------------------------------------------------------
    # persist_graph
    # ------------------------------------------------------------------

    def persist_graph(
        self,
        call_edges: list[CallEdge],
        dep_edges: list[DependencyEdge],
        symbol_table: SymbolTable,
    ) -> None:
        """Bulk-ingest all nodes and edges using batched MERGE transactions."""
        assert self._driver is not None

        # 1. Symbol nodes
        symbol_props = [_symbol_to_node_props(r) for r in symbol_table]
        self._batch_merge_nodes(symbol_props)

        # 2. Synthetic Module nodes
        seen: dict[str, dict[str, Any]] = {}
        for record in symbol_table:
            if record.module and record.module not in seen:
                seen[record.module] = _module_node_props(
                    record.file_path, record.module
                )
        self._batch_merge_nodes(list(seen.values()))

        # 3. CALLS edges
        calls_data: list[dict[str, Any]] = []
        for edge in call_edges:
            if not edge.resolved:
                continue
            caller_rec = symbol_table.lookup_qualified(edge.caller)
            callee_rec = symbol_table.lookup_qualified(edge.callee)
            if caller_rec is None:
                matches = symbol_table.lookup(edge.caller)
                caller_rec = matches[0] if matches else None
            if callee_rec is None:
                matches = symbol_table.lookup(edge.callee)
                callee_rec = matches[0] if matches else None
            if caller_rec is None or callee_rec is None:
                continue
            calls_data.append(
                {
                    "src": caller_rec.symbol_id,
                    "tgt": callee_rec.symbol_id,
                    "call_type": edge.call_type,
                    "resolution": edge.resolution,
                    "call_site_line": edge.call_site_line,
                    "is_recursive": edge.is_recursive,
                }
            )
        self._batch_merge_edges(calls_data, "CALLS")

        # 4. IMPORTS edges (Module -> Module)
        imports_data: list[dict[str, Any]] = []
        for dep in dep_edges:
            if not dep.resolved:
                continue
            tgt_mod = dep.target_module.lstrip(".")
            imports_data.append(
                {
                    "src": f"module:{dep.source_module}",
                    "tgt": f"module:{tgt_mod}",
                    "import_type": dep.import_type,
                    "line": dep.line,
                    "is_relative": dep.is_relative,
                    "is_wildcard": dep.is_wildcard,
                }
            )
        self._batch_merge_edges(imports_data, "IMPORTS")

        # 5. INHERITS and CONTAINS
        inherits, contains = _build_synthetic_edges(symbol_table)
        inherits_data = [{"src": s, "tgt": t, **p} for s, t, p in inherits]
        contains_data = [{"src": s, "tgt": t, **p} for s, t, p in contains]
        self._batch_merge_edges(inherits_data, "INHERITS")
        self._batch_merge_edges(contains_data, "CONTAINS")

        logger.info(
            "Neo4jGraphStore: persisted %d symbols, %d call edges, "
            "%d import edges, %d module nodes",
            len(symbol_props),
            len(calls_data),
            len(imports_data),
            len(seen),
        )

    def _batch_merge_nodes(self, props_list: list[dict[str, Any]]) -> None:
        """MERGE nodes in batches, applying the correct label per node."""
        assert self._driver is not None
        for batch in _chunk(props_list, self._batch_size):
            with self._driver.session(database=self._database) as session:
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (n:_Node {symbol_id: row.symbol_id})
                    SET n += row
                    WITH n, row
                    CALL apoc.create.addLabels(n, [row.label]) YIELD node
                    RETURN count(node)
                    """,
                    batch=batch,
                )

    def _batch_merge_edges(self, edges: list[dict[str, Any]], rel_type: str) -> None:
        """MERGE edges of a single relationship type in batches.

        Each element of *edges* must have ``"src"`` and ``"tgt"`` keys with
        ``symbol_id`` values; remaining keys become relationship properties.
        """
        if not edges:
            return
        assert self._driver is not None
        cypher = f"""
            UNWIND $batch AS row
            MATCH (a:_Node {{symbol_id: row.src}})
            MATCH (b:_Node {{symbol_id: row.tgt}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += row
        """
        for batch in _chunk(edges, self._batch_size):
            with self._driver.session(database=self._database) as session:
                session.run(cypher, batch=batch)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> GraphQueryResult:
        """Run an arbitrary Cypher query and return all result rows.

        Args:
            cypher: A valid Cypher statement.
            params: Optional named parameters referenced in *cypher*.

        Returns:
            :class:`GraphQueryResult` wrapping the list of row dicts.
        """
        assert self._driver is not None
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, parameters=params or {})
            records = [dict(r) for r in result]
            summary = result.consume()
            return GraphQueryResult(
                records=records,
                summary=(
                    f"counters={summary.counters}"
                    if hasattr(summary, "counters")
                    else ""
                ),
            )

    def get_neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        depth: int = 1,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return neighbors within *depth* hops via Cypher traversal."""
        assert self._driver is not None
        if direction == "out":
            rel_pattern = "-[r*1..{d}]->"
        elif direction == "in":
            rel_pattern = "<-[r*1..{d}]-"
        else:
            rel_pattern = "-[r*1..{d}]-"

        rel_clause = rel_pattern.format(d=depth)

        if edge_types:
            types_str = "|".join(edge_types)
            cypher = f"""
                MATCH (n {{symbol_id: $id}}){rel_clause.replace("[r*", f"[r:{types_str}*")}(m)
                WHERE n <> m
                RETURN DISTINCT properties(m) AS node
            """
        else:
            cypher = f"""
                MATCH (n {{symbol_id: $id}}){rel_clause}(m)
                WHERE n <> m
                RETURN DISTINCT properties(m) AS node
            """

        result = self.query(cypher, {"id": node_id})
        return [row.get("node", row) for row in result.records]

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return shortest path nodes via Cypher ``shortestPath``."""
        assert self._driver is not None
        if edge_types:
            types_str = "|".join(edge_types)
            rel_spec = f"[:{types_str}*]"
        else:
            rel_spec = "[*]"

        cypher = f"""
            MATCH (a {{symbol_id: $src}}), (b {{symbol_id: $tgt}})
            MATCH p = shortestPath((a)-{rel_spec}-(b))
            RETURN [n IN nodes(p) | properties(n)] AS path_nodes
        """
        result = self.query(cypher, {"src": source_id, "tgt": target_id})
        if not result.records:
            return []
        return result.records[0].get("path_nodes", [])

    def subgraph(
        self,
        node_ids: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract induced subgraph over *node_ids* via Cypher."""
        assert self._driver is not None
        node_cypher = """
            MATCH (n) WHERE n.symbol_id IN $ids
            RETURN properties(n) AS node
        """
        edge_cypher = """
            MATCH (a)-[r]->(b)
            WHERE a.symbol_id IN $ids AND b.symbol_id IN $ids
            RETURN a.symbol_id AS source, b.symbol_id AS target,
                   type(r) AS type, properties(r) AS props
        """
        nodes_result = self.query(node_cypher, {"ids": node_ids})
        edges_result = self.query(edge_cypher, {"ids": node_ids})
        nodes = [row["node"] for row in nodes_result.records]
        edges = []
        for row in edges_result.records:
            ep = {"source": row["source"], "target": row["target"], "type": row["type"]}
            ep.update(row.get("props", {}))
            edges.append(ep)
        return nodes, edges

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all nodes and relationships (DETACH DELETE)."""
        self.query("MATCH (n) DETACH DELETE n")
        logger.info("Neo4jGraphStore: all nodes and edges deleted")

    def close(self) -> None:
        """Close the underlying Neo4j driver connection pool."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4jGraphStore: driver closed")

    def __enter__(self) -> Neo4jGraphStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Neo4jGraphStore(uri={self._uri!r}, database={self._database!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def GraphStore(  # noqa: N802  (factory named like a class by convention)
    uri: str | None = None,
    *,
    username: str = "neo4j",
    password: str = "reporag123",
    database: str = "neo4j",
    batch_size: int = _BATCH_SIZE,
    fallback: bool = False,
) -> GraphStoreProtocol:
    """Create and return the best available graph store backend.

    Resolution order:

    1. If *uri* is provided and *fallback* is ``False``, connect to Neo4j
       and raise on failure.
    2. If *uri* is provided and *fallback* is ``True``, try Neo4j; silently
       return :class:`NetworkXGraphStore` on any connection error.
    3. If *uri* is ``None``, always return :class:`NetworkXGraphStore`.

    Args:
        uri: Bolt URI of the Neo4j instance.  ``None`` skips Neo4j.
        username: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.
        batch_size: Batch size forwarded to :class:`Neo4jGraphStore`.
        fallback: When ``True``, fall back to NetworkX if Neo4j is
            unreachable instead of raising an exception.

    Returns:
        A concrete store implementing :class:`GraphStoreProtocol`.

    Examples::

        # Test / CI usage (no Docker required):
        store = GraphStore(fallback=True)

        # Production usage (raises if Neo4j is down):
        store = GraphStore(uri="bolt://localhost:7687")

        # Best-effort: tries Neo4j, silently falls back:
        store = GraphStore(uri="bolt://localhost:7687", fallback=True)
    """
    if uri is None:
        logger.debug("GraphStore: no URI provided, using NetworkXGraphStore")
        return NetworkXGraphStore()

    try:
        return Neo4jGraphStore(
            uri,
            username=username,
            password=password,
            database=database,
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        if fallback:
            logger.warning(
                "GraphStore: Neo4j unavailable (%s); falling back to NetworkXGraphStore",
                exc,
            )
            return NetworkXGraphStore()
        raise

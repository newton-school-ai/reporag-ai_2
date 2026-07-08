"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j.  Provides Cypher query helpers for neighbors, shortest
path, and subgraph extraction.  Includes a NetworkX fallback that
implements the **same interface** so tests can run without a live Neo4j
instance.

Architecture
------------
:class:`GraphStore` is an **abstract base class** that declares the full
public interface.  Two concrete implementations are provided:

* :class:`Neo4jGraphStore` -- backed by a live Neo4j instance via the
  official ``neo4j`` driver.  Bulk inserts use batched transactions so
  10 K+ nodes are handled without exhausting driver memory.  Connection
  errors are caught and surfaced with a configurable retry policy
  (``max_retries`` + ``retry_delay``).  Node upserts try the APOC plugin
  first and fall back automatically to a per-label ``MERGE`` strategy so
  the store works on both Community and Enterprise editions.

* :class:`NetworkXGraphStore` -- backed by :mod:`networkx`.  No external
  service required; ideal for unit tests and CI environments.  Implements
  every method in the abstract interface so the same test suite exercises
  both backends.

Node labels
-----------
Node labels in Neo4j are derived from the ``type`` field of a
:class:`~src.reporag.graph.symbol_table.SymbolRecord`:

+-----------+------------------+
| type      | Neo4j label      |
+===========+==================+
| function  | Function         |
+-----------+------------------+
| method    | Function         |
+-----------+------------------+
| class     | Class            |
+-----------+------------------+
| module    | Module           |
+-----------+------------------+
| *other*   | Symbol           |
+-----------+------------------+

Edge types
----------
+-------------------+---------------------+
| source            | Neo4j relationship  |
+===================+=====================+
| CallEdge          | CALLS               |
+-------------------+---------------------+
| DependencyEdge    | IMPORTS             |
+-------------------+---------------------+
| SymbolRecord with | INHERITS            |
| non-empty bases   |                     |
+-------------------+---------------------+
| SymbolRecord with | CONTAINS            |
| parent != None    |                     |
+-------------------+---------------------+

Usage::

    # Neo4j backend (requires a running Neo4j instance)
    from src.reporag.graph.neo4j_store import Neo4jGraphStore

    store = Neo4jGraphStore(uri="bolt://localhost:7687",
                             username="neo4j", password="password")
    store.persist_graph(call_edges, dep_edges, symbols)
    neighbors = store.get_neighbors("examples.auth.authenticate_user", depth=2)
    path = store.shortest_path("examples.app.main", "examples.db.get_user")
    store.close()

    # NetworkX backend (no external service)
    from src.reporag.graph.neo4j_store import NetworkXGraphStore

    store = NetworkXGraphStore()
    store.persist_graph(call_edges, dep_edges, symbols)
    result = store.get_neighbors("examples.auth.authenticate_user", depth=1)
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label / type helpers
# ---------------------------------------------------------------------------

_TYPE_TO_LABEL: dict[str, str] = {
    "function": "Function",
    "method": "Function",
    "class": "Class",
    "module": "Module",
}


def _symbol_label(symbol_type: str) -> str:
    """Map a :class:`~src.reporag.graph.symbol_table.SymbolRecord` type to a Neo4j label."""
    return _TYPE_TO_LABEL.get(symbol_type, "Symbol")


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A single node returned from a graph query.

    Attributes:
        node_id:    Stable node identifier (typically ``symbol_id`` or
                    a module path).
        label:      Neo4j label / node kind (``"Function"``, ``"Class"``,
                    ``"Module"``, ``"Symbol"``).
        properties: Arbitrary key/value payload stored on the node.
    """

    node_id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed relationship returned from a graph query.

    Attributes:
        source:     ``node_id`` of the start node.
        target:     ``node_id`` of the end node.
        rel_type:   Neo4j relationship type (e.g. ``"CALLS"``).
        properties: Arbitrary key/value payload stored on the relationship.
    """

    source: str
    target: str
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphResult:
    """Nodes and edges constituting a subgraph extraction result."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """Abstract interface for the code knowledge graph store.

    Both :class:`Neo4jGraphStore` and :class:`NetworkXGraphStore` implement
    this interface so callers can swap backends transparently.
    """

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @abstractmethod
    def persist_graph(
        self,
        call_edges: Iterable[Any],
        dep_edges: Iterable[Any],
        symbols: Iterable[Any],
    ) -> None:
        """Persist the full code knowledge graph.

        Args:
            call_edges: Iterable of :class:`~src.reporag.graph.call_graph.CallEdge`
                objects (produces ``CALLS`` relationships).
            dep_edges:  Iterable of
                :class:`~src.reporag.graph.dependency_graph.DependencyEdge`
                objects (produces ``IMPORTS`` relationships).
            symbols:    Iterable of
                :class:`~src.reporag.graph.symbol_table.SymbolRecord` objects
                (produces nodes, ``CONTAINS`` and ``INHERITS`` edges).
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete **all** nodes and relationships from the store."""

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> SubgraphResult:
        """Return all nodes reachable from *node_id* within *depth* hops.

        Args:
            node_id:   The ``symbol_id`` / module path to start from.
            depth:     Maximum number of relationship hops to follow.
            rel_types: Restrict traversal to these relationship types.
                       ``None`` means follow all types.

        Returns:
            A :class:`SubgraphResult` containing the reachable nodes and
            the edges connecting them (start node excluded from nodes list).
        """

    @abstractmethod
    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        rel_types: list[str] | None = None,
    ) -> list[str]:
        """Return the ``node_id`` sequence of the shortest path.

        Args:
            source_id: Start node identifier.
            target_id: End node identifier.
            rel_types: Edge types to traverse (``None`` = all).

        Returns:
            Ordered list of ``node_id`` values from *source_id* to
            *target_id* (inclusive), or an empty list when no path exists.
        """

    @abstractmethod
    def get_subgraph(self, node_ids: list[str]) -> SubgraphResult:
        """Extract the induced subgraph for the given set of node ids.

        Returns all edges whose *both* endpoints are in *node_ids*.
        """

    @abstractmethod
    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher query (Neo4j backend) or a safe subset (NetworkX).

        The NetworkX backend implements a small safe subset used by unit tests
        (see :meth:`NetworkXGraphStore.query`).  Any unsupported Cypher raises
        :class:`NotImplementedError`.

        Args:
            cypher:     Cypher query string.
            parameters: Optional parameter map substituted into the query.

        Returns:
            List of row dicts -- each key corresponds to an ``AS`` alias in
            the ``RETURN`` clause.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend (driver connections etc.)."""


# ---------------------------------------------------------------------------
# Neo4j implementation
# ---------------------------------------------------------------------------

_NEO4J_BATCH_SIZE = 500  # rows per transaction batch


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store.

    Wraps the official ``neo4j`` Python driver.  All write operations use
    explicit transactions so failures are atomic.  Bulk inserts are chunked
    into batches of :data:`_NEO4J_BATCH_SIZE` rows to prevent memory
    exhaustion for very large repositories.

    Node upserts first attempt to use the **APOC** plugin
    (``apoc.merge.node``) for dynamic-label ``MERGE``.  If APOC is not
    available the store falls back to per-label ``MERGE`` statements, which
    are universally compatible with both Community and Enterprise editions.

    Connection errors during construction are retried up to *max_retries*
    times with *retry_delay* seconds between attempts.  After exhausting
    retries the original exception is re-raised wrapped in a
    :class:`RuntimeError`.

    Args:
        uri:         Bolt / bolt+s URI (``"bolt://localhost:7687"``).
        username:    Neo4j username (default ``"neo4j"``).
        password:    Neo4j password (default ``"password"``).
        database:    Target database (``None`` uses the driver default).
        max_retries: Number of retry attempts on connection failure (>= 1).
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "neo4j package is required for Neo4jGraphStore. "
                "Install it with: pip install neo4j"
            ) from exc

        self._uri = uri
        self._database = database
        self._max_retries = max(1, max_retries)
        self._retry_delay = retry_delay
        # Set during first upsert attempt; avoids repeated APOC probing.
        self._apoc_available: bool | None = None

        self._driver = self._connect_with_retry(GraphDatabase, username, password)
        logger.info("Neo4jGraphStore connected to %s", uri)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect_with_retry(
        self, GraphDatabase: Any, username: str, password: str  # noqa: N803
    ) -> Any:
        """Create the Neo4j driver, retrying on transient connection failures."""
        last_exc: Exception = RuntimeError("No connection attempt made")
        for attempt in range(1, self._max_retries + 1):
            try:
                driver = GraphDatabase.driver(self._uri, auth=(username, password))
                driver.verify_connectivity()
                return driver
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Neo4j connection attempt %d/%d failed (%s); "
                        "retrying in %.1fs ...",
                        attempt,
                        self._max_retries,
                        exc,
                        self._retry_delay,
                    )
                    time.sleep(self._retry_delay)
                else:
                    logger.error(
                        "Neo4j connection failed after %d attempt(s): %s",
                        self._max_retries,
                        exc,
                    )
        raise RuntimeError(
            f"Could not connect to Neo4j at {self._uri} after "
            f"{self._max_retries} attempt(s)"
        ) from last_exc

    def _session(self) -> Any:
        kwargs: dict[str, Any] = {}
        if self._database:
            kwargs["database"] = self._database
        return self._driver.session(**kwargs)

    # ------------------------------------------------------------------
    # Index / constraint setup
    # ------------------------------------------------------------------

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints for ``node_id`` (idempotent).

        Tries the Neo4j 4.4+ syntax first; silently ignores errors on older
        editions that use a different DDL form.
        """
        labels = ["Function", "Class", "Module", "Symbol"]
        with self._session() as session:
            for label in labels:
                with contextlib.suppress(Exception):
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.node_id IS UNIQUE"
                    )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def persist_graph(
        self,
        call_edges: Iterable[Any],
        dep_edges: Iterable[Any],
        symbols: Iterable[Any],
    ) -> None:
        """Persist the full code knowledge graph to Neo4j.

        Execution order:

        1. Ensure uniqueness constraints.
        2. Upsert symbol nodes (APOC if available, else per-label MERGE).
        3. Create ``CONTAINS`` / ``INHERITS`` edges from symbol metadata.
        4. Create ``CALLS`` edges from *call_edges*.
        5. Create ``IMPORTS`` edges from *dep_edges*.
        """
        self._ensure_constraints()

        symbol_list = list(symbols)
        call_list = list(call_edges)
        dep_list = list(dep_edges)

        logger.info(
            "Persisting graph: %d symbols, %d call edges, %d dep edges",
            len(symbol_list),
            len(call_list),
            len(dep_list),
        )

        self._upsert_symbol_nodes(symbol_list)
        self._create_structural_edges(symbol_list)
        self._create_call_edges(call_list)
        self._create_import_edges(dep_list)

    def _upsert_symbol_nodes(self, symbols: list[Any]) -> None:
        """Bulk-upsert symbol records as typed Neo4j nodes.

        Probes for APOC on the first call and stores the result so
        subsequent calls skip the probe.  Falls back to per-label MERGE
        when APOC is absent.
        """
        if not symbols:
            return

        # Probe APOC availability once
        if self._apoc_available is None:
            try:
                with self._session() as session:
                    session.run("RETURN apoc.version() AS v").consume()
                self._apoc_available = True
                logger.debug("APOC plugin detected; using apoc.merge.node")
            except Exception:  # noqa: BLE001
                self._apoc_available = False
                logger.debug("APOC not available; using per-label MERGE fallback")

        if self._apoc_available:
            self._upsert_nodes_apoc(symbols)
        else:
            self._upsert_nodes_no_apoc(symbols)

    def _upsert_nodes_apoc(self, symbols: list[Any]) -> None:
        """Upsert nodes using ``apoc.merge.node`` (dynamic label support)."""
        for batch in _chunks(symbols, _NEO4J_BATCH_SIZE):
            rows = [
                {
                    "node_id": s.symbol_id,
                    "name": s.name,
                    "qualified_name": s.qualified_name,
                    "symbol_type": s.type,
                    "file_path": s.file_path,
                    "module": s.module,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "signature": s.signature or "",
                    "docstring": s.docstring or "",
                    "language": s.language,
                    "is_async": s.is_async,
                    "label": _symbol_label(s.type),
                }
                for s in batch
            ]
            with self._session() as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    CALL apoc.merge.node(
                        [row.label],
                        {node_id: row.node_id},
                        row
                    ) YIELD node
                    RETURN count(node)
                    """,
                    rows=rows,
                )

    def _upsert_nodes_no_apoc(self, symbols: list[Any]) -> None:
        """Upsert nodes with per-label ``MERGE`` (no APOC required)."""
        # Group by label so each MERGE targets a single concrete label.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for s in symbols:
            label = _symbol_label(s.type)
            by_label.setdefault(label, []).append(
                {
                    "node_id": s.symbol_id,
                    "name": s.name,
                    "qualified_name": s.qualified_name,
                    "symbol_type": s.type,
                    "file_path": s.file_path,
                    "module": s.module,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "signature": s.signature or "",
                    "docstring": s.docstring or "",
                    "language": s.language,
                    "is_async": s.is_async,
                }
            )
        for label, rows in by_label.items():
            for batch in _chunks(rows, _NEO4J_BATCH_SIZE):
                # Each batch gets its own session so a partial failure in
                # one label does not silently abandon remaining batches.
                with self._session() as session:
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{node_id: row.node_id}})
                        SET n += row
                        """,
                        rows=batch,
                    )

    def _create_structural_edges(self, symbols: list[Any]) -> None:
        """Create CONTAINS and INHERITS edges derived from symbol metadata."""
        contains_rows: list[dict[str, str]] = []
        inherits_rows: list[dict[str, str]] = []

        for s in symbols:
            if s.parent:
                contains_rows.append({"parent_id": s.parent, "child_id": s.symbol_id})
            for base in s.bases:
                inherits_rows.append({"class_id": s.symbol_id, "base_name": base})

        with self._session() as session:
            for batch in _chunks(contains_rows, _NEO4J_BATCH_SIZE):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (parent {node_id: row.parent_id})
                    MATCH (child  {node_id: row.child_id})
                    MERGE (parent)-[:CONTAINS]->(child)
                    """,
                    rows=batch,
                )
            for batch in _chunks(inherits_rows, _NEO4J_BATCH_SIZE):
                # Match base class by bare name; take first hit per class
                # when multiple nodes share a name (best-effort resolution).
                # The CALL subquery scopes LIMIT to each individual row so
                # that a single LIMIT 1 does not abort the whole batch.
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (cls {node_id: row.class_id})
                    CALL {
                        WITH row
                        MATCH (base) WHERE base.name = row.base_name
                        RETURN base LIMIT 1
                    }
                    MERGE (cls)-[:INHERITS]->(base)
                    """,
                    rows=batch,
                )

    def _create_call_edges(self, call_edges: list[Any]) -> None:
        """Create CALLS relationships from :class:`CallEdge` objects."""
        rows = [
            {
                "caller_id": e.caller,
                "callee_id": e.callee,
                "call_site_line": e.call_site_line,
                "call_type": e.call_type,
                "resolution": e.resolution,
                "is_recursive": e.is_recursive,
                "caller_file": e.caller_file,
                "callee_file": e.callee_file or "",
            }
            for e in call_edges
            if e.resolved
        ]
        with self._session() as session:
            for batch in _chunks(rows, _NEO4J_BATCH_SIZE):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (caller {node_id: row.caller_id})
                    MATCH (callee {node_id: row.callee_id})
                    MERGE (caller)-[r:CALLS {
                        call_site_line: row.call_site_line,
                        caller_file:    row.caller_file
                    }]->(callee)
                    SET r.call_type    = row.call_type,
                        r.resolution   = row.resolution,
                        r.is_recursive = row.is_recursive,
                        r.callee_file  = row.callee_file
                    """,
                    rows=batch,
                )

    def _create_import_edges(self, dep_edges: list[Any]) -> None:
        """Create IMPORTS relationships from :class:`DependencyEdge` objects.

        Matches on ``node_id`` (the file path) rather than the ``module``
        property to avoid false matches when two modules share a dotted
        name segment.  Only resolved edges (where ``target`` is a project
        file path) produce a relationship.
        """
        rows = [
            {
                "source_id": e.source,  # file path used as node_id
                "target_id": e.target,  # file path used as node_id
                "source_module": e.source_module,
                "target_module": e.target_module,
                "import_type": e.import_type,
                "line": e.line,
                "is_relative": e.is_relative,
                "is_wildcard": e.is_wildcard,
            }
            for e in dep_edges
            if e.resolved
        ]
        with self._session() as session:
            for batch in _chunks(rows, _NEO4J_BATCH_SIZE):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (src {node_id: row.source_id})
                    MATCH (tgt {node_id: row.target_id})
                    MERGE (src)-[r:IMPORTS {
                        line:        row.line,
                        source_id:   row.source_id
                    }]->(tgt)
                    SET r.import_type    = row.import_type,
                        r.source_module  = row.source_module,
                        r.target_module  = row.target_module,
                        r.is_relative    = row.is_relative,
                        r.is_wildcard    = row.is_wildcard
                    """,
                    rows=batch,
                )

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all nodes and relationships from Neo4j (batched to avoid OOM)."""
        with self._session() as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 "
                    "DETACH DELETE n "
                    "RETURN count(n) AS cnt"
                )
                record = result.single()
                if record is None or record["cnt"] == 0:
                    break
        logger.info("Neo4j graph cleared")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> SubgraphResult:
        """Return nodes reachable from *node_id* within *depth* hops."""
        rel_filter = "" if not rel_types else ":" + "|".join(rel_types)
        cypher = (
            f"MATCH (start {{node_id: $node_id}})-[r{rel_filter}*1..{depth}]-(neighbor) "
            "RETURN DISTINCT neighbor, r"
        )
        rows = self.query(cypher, {"node_id": node_id})
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for row in rows:
            n = row.get("neighbor")
            if n and hasattr(n, "items"):
                nid = n.get("node_id", "")
                label = list(n.labels)[0] if hasattr(n, "labels") else "Symbol"
                nodes[nid] = GraphNode(node_id=nid, label=label, properties=dict(n))
            r = row.get("r")
            if r:
                rels = r if isinstance(r, list) else [r]
                for rel in rels:
                    if hasattr(rel, "start_node") and hasattr(rel, "end_node"):
                        edges.append(
                            GraphEdge(
                                source=rel.start_node.get("node_id", ""),
                                target=rel.end_node.get("node_id", ""),
                                rel_type=rel.type,
                            )
                        )
        return SubgraphResult(nodes=list(nodes.values()), edges=edges)

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        rel_types: list[str] | None = None,
    ) -> list[str]:
        """Return the shortest path between two nodes as a list of node ids."""
        rel_filter = "" if not rel_types else ":" + "|".join(rel_types)
        cypher = (
            "MATCH (src {node_id: $src}), (tgt {node_id: $tgt}), "
            f"path = shortestPath((src)-[{rel_filter}*]-(tgt)) "
            "RETURN [n IN nodes(path) | n.node_id] AS ids"
        )
        rows = self.query(cypher, {"src": source_id, "tgt": target_id})
        if not rows:
            return []
        return list(rows[0].get("ids", []))

    def get_subgraph(self, node_ids: list[str]) -> SubgraphResult:
        """Extract the induced subgraph for the given node ids."""
        cypher = (
            "MATCH (a)-[r]->(b) "
            "WHERE a.node_id IN $ids AND b.node_id IN $ids "
            "RETURN a, r, b"
        )
        rows = self.query(cypher, {"ids": node_ids})
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for row in rows:
            for key in ("a", "b"):
                n = row.get(key)
                if n and hasattr(n, "items"):
                    nid = n.get("node_id", "")
                    label = list(n.labels)[0] if hasattr(n, "labels") else "Symbol"
                    nodes[nid] = GraphNode(node_id=nid, label=label, properties=dict(n))
            r = row.get("r")
            if r and hasattr(r, "start_node"):
                edges.append(
                    GraphEdge(
                        source=r.start_node.get("node_id", ""),
                        target=r.end_node.get("node_id", ""),
                        rel_type=r.type,
                    )
                )
        return SubgraphResult(nodes=list(nodes.values()), edges=edges)

    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher query and return list-of-dicts results."""
        with self._session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def close(self) -> None:
        """Close the Neo4j driver and release all connections."""
        self._driver.close()
        logger.info("Neo4jGraphStore driver closed")


# ---------------------------------------------------------------------------
# NetworkX fallback implementation
# ---------------------------------------------------------------------------


class NetworkXGraphStore(GraphStore):
    """In-memory graph store backed by :mod:`networkx`.

    Implements the same :class:`GraphStore` interface as
    :class:`Neo4jGraphStore` so the same test suite can run in CI without a
    Neo4j service.

    ``query()`` does **not** execute arbitrary Cypher; it parses a small
    safe subset used by unit tests::

        MATCH (n) RETURN count(n) AS cnt
        MATCH ()-[r:TYPE]->() RETURN count(r) AS cnt
        MATCH (a:Label)-[:TYPE]->(b:Label) RETURN a.prop, b.prop [LIMIT n]

    Any other Cypher raises :class:`NotImplementedError`.

    Args:
        directed: When ``True`` (default) uses a :class:`networkx.DiGraph`;
            ``False`` uses :class:`networkx.Graph` (rarely needed).
    """

    def __init__(self, *, directed: bool = True) -> None:
        import networkx as nx  # type: ignore[import]

        self._nx = nx
        self._graph: Any = nx.DiGraph() if directed else nx.Graph()
        # node_id -> Neo4j label
        self._labels: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def persist_graph(
        self,
        call_edges: Iterable[Any],
        dep_edges: Iterable[Any],
        symbols: Iterable[Any],
    ) -> None:
        """Populate the NetworkX graph from symbols and edges."""
        symbol_list = list(symbols)
        call_list = list(call_edges)
        dep_list = list(dep_edges)

        # 1. Symbol nodes
        for s in symbol_list:
            label = _symbol_label(s.type)
            self._graph.add_node(
                s.symbol_id,
                node_id=s.symbol_id,
                name=s.name,
                qualified_name=s.qualified_name,
                type=s.type,
                file_path=s.file_path,
                module=s.module,
                start_line=s.start_line,
                end_line=s.end_line,
                signature=s.signature,
                docstring=s.docstring,
                language=s.language,
                is_async=s.is_async,
                label=label,
            )
            self._labels[s.symbol_id] = label

        # 2. CONTAINS / INHERITS from symbol metadata
        #    Build a name -> node_id index once for O(1) base-class lookup.
        name_index: dict[str, list[str]] = {}
        for nid, data in self._graph.nodes(data=True):
            name_index.setdefault(data.get("name", ""), []).append(nid)

        for s in symbol_list:
            if s.parent and s.parent in self._graph:
                self._graph.add_edge(s.parent, s.symbol_id, rel_type="CONTAINS")
            for base in s.bases:
                for match_id in name_index.get(base, []):
                    if match_id != s.symbol_id:
                        self._graph.add_edge(s.symbol_id, match_id, rel_type="INHERITS")

        # 3. CALLS edges (resolved only)
        for e in call_list:
            if not e.resolved:
                continue
            # Auto-stub missing caller / callee nodes (partial symbol table)
            for nid in (e.caller, e.callee):
                if nid not in self._graph:
                    self._graph.add_node(nid, node_id=nid, name=nid, label="Symbol")
                    self._labels[nid] = "Symbol"
            self._graph.add_edge(
                e.caller,
                e.callee,
                rel_type="CALLS",
                call_site_line=e.call_site_line,
                call_type=e.call_type,
                resolution=e.resolution,
                is_recursive=e.is_recursive,
                caller_file=e.caller_file,
                callee_file=e.callee_file or "",
            )

        # 4. IMPORTS edges
        #    Node key for module files is the file path (e.source / e.target),
        #    consistent with how Neo4jGraphStore matches on node_id.
        for e in dep_list:
            if not e.resolved:
                continue
            # Auto-stub module nodes keyed by file path if absent
            for nid, mod_name in (
                (e.source, e.source_module),
                (e.target, e.target_module),
            ):
                if nid not in self._graph:
                    self._graph.add_node(
                        nid,
                        node_id=nid,
                        name=mod_name,
                        module=mod_name,
                        label="Module",
                    )
                    self._labels[nid] = "Module"
            self._graph.add_edge(
                e.source,
                e.target,
                rel_type="IMPORTS",
                import_type=e.import_type,
                line=e.line,
                is_relative=e.is_relative,
                is_wildcard=e.is_wildcard,
            )

        logger.info(
            "NetworkXGraphStore: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def clear(self) -> None:
        """Clear the in-memory graph."""
        self._graph.clear()
        self._labels.clear()
        logger.info("NetworkXGraphStore cleared")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> SubgraphResult:
        """BFS up to *depth* hops from *node_id* (undirected traversal).

        Edges are deduplicated: each (source, target, rel_type) triple
        appears at most once in the result, regardless of how many BFS
        paths traverse it.
        """
        if node_id not in self._graph:
            return SubgraphResult()

        # Snapshot the undirected view once before BFS so any concurrent
        # mutations don't affect traversal and self-loop semantics are clear.
        undirected = self._graph.to_undirected(as_view=False)

        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}
        seen_edges: set[tuple[str, str, str]] = set()
        collected_edges: list[GraphEdge] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for neighbor in undirected.neighbors(n):
                    # Respect the rel_type filter using directed edge data.
                    rt = self._directed_rel_type(n, neighbor)
                    if rel_types is not None and rt not in rel_types:
                        continue
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    # Canonical (directed) edge key prevents duplicates.
                    if self._graph.has_edge(n, neighbor):
                        src, tgt = n, neighbor
                    else:
                        src, tgt = neighbor, n
                    key = (src, tgt, rt)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        collected_edges.append(
                            GraphEdge(source=src, target=tgt, rel_type=rt)
                        )
            visited |= next_frontier
            frontier = next_frontier

        nodes = [
            GraphNode(
                node_id=nid,
                label=self._labels.get(nid, "Symbol"),
                properties=dict(self._graph.nodes[nid]),
            )
            for nid in visited
            if nid != node_id
        ]
        return SubgraphResult(nodes=nodes, edges=collected_edges)

    def _directed_rel_type(self, a: str, b: str) -> str:
        """Return the ``rel_type`` of the directed edge between *a* and *b*."""
        data = self._graph.get_edge_data(a, b) or self._graph.get_edge_data(b, a) or {}
        return data.get("rel_type", "")

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        rel_types: list[str] | None = None,
    ) -> list[str]:
        """Return shortest undirected path between two node ids.

        When *rel_types* is provided only edges of those types are eligible.
        Returns ``[]`` if either node is absent or no path exists.
        """
        if source_id not in self._graph or target_id not in self._graph:
            return []
        try:
            if rel_types:
                filtered = self._nx.DiGraph()
                for u, v, data in self._graph.edges(data=True):
                    if data.get("rel_type") in rel_types:
                        filtered.add_edge(u, v)
                # Add isolated nodes so NodeNotFound is raised correctly
                for nid in (source_id, target_id):
                    if nid not in filtered:
                        filtered.add_node(nid)
                return list(
                    self._nx.shortest_path(
                        filtered.to_undirected(), source_id, target_id
                    )
                )
            return list(
                self._nx.shortest_path(
                    self._graph.to_undirected(), source_id, target_id
                )
            )
        except (self._nx.NetworkXNoPath, self._nx.NodeNotFound):
            return []

    def get_subgraph(self, node_ids: list[str]) -> SubgraphResult:
        """Induced subgraph for the given node ids."""
        id_set = set(node_ids)
        nodes = [
            GraphNode(
                node_id=nid,
                label=self._labels.get(nid, "Symbol"),
                properties=dict(self._graph.nodes[nid]),
            )
            for nid in node_ids
            if nid in self._graph
        ]
        edges = [
            GraphEdge(
                source=u,
                target=v,
                rel_type=data.get("rel_type", ""),
                properties={k: val for k, val in data.items() if k != "rel_type"},
            )
            for u, v, data in self._graph.edges(data=True)
            if u in id_set and v in id_set
        ]
        return SubgraphResult(nodes=nodes, edges=edges)

    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a limited subset of Cypher against the NetworkX graph.

        Supported patterns (case-insensitive, whitespace-normalised):

        * ``MATCH (n) RETURN count(n) AS cnt``
        * ``MATCH ()-[r:TYPE]->() RETURN count(r) AS cnt``
        * ``MATCH (a:Label)-[:TYPE]->(b:Label) RETURN a.prop, b.prop [LIMIT n]``

        Raises :class:`NotImplementedError` for anything else.
        """
        import re

        # ``parameters`` is intentionally unused here: the mini-interpreter
        # works on graph state, not on parameterised literals.
        q = " ".join(cypher.split())  # normalise whitespace

        # --- count(n) ---
        if re.search(r"MATCH \(n\) RETURN count\(n\) AS cnt", q, re.I):
            return [{"cnt": self._graph.number_of_nodes()}]

        # --- count(r) by relationship type ---
        m = re.search(r"MATCH \(\)-\[r:(\w+)\]->\(\) RETURN count\(r\) AS cnt", q, re.I)
        if m:
            rel_type = m.group(1).upper()  # normalise so CALLS == calls
            cnt = sum(
                1
                for _, _, d in self._graph.edges(data=True)
                if (d.get("rel_type") or "").upper() == rel_type
            )
            return [{"cnt": cnt}]

        # --- MATCH (a:Label)-[:TYPE]->(b:Label) RETURN a.prop, b.prop [LIMIT n] ---
        m = re.search(
            r"MATCH \((\w+)(?::(\w+))?\)-\[:(\w+)\]->\((\w+)(?::(\w+))?\)"
            r"\s*RETURN\s*(\S+),\s*(\S+)(?:\s+LIMIT\s+(\d+))?",
            q,
            re.I,
        )
        if m:
            a_label = m.group(2)  # may be None (no label filter)
            rel_type = m.group(3).upper()  # normalise casing
            b_label = m.group(5)  # may be None
            a_ret = m.group(6)  # e.g. "f.name"
            b_ret = m.group(7)  # e.g. "g.name"
            a_prop = a_ret.split(".")[-1]
            b_prop = b_ret.split(".")[-1]
            limit = int(m.group(8)) if m.group(8) else None

            results: list[dict[str, Any]] = []
            for u, v, data in self._graph.edges(data=True):
                # Compare normalised so query casing does not matter.
                if (data.get("rel_type") or "").upper() != rel_type:
                    continue
                u_data = self._graph.nodes.get(u, {})
                v_data = self._graph.nodes.get(v, {})
                if a_label and u_data.get("label") != a_label:
                    continue
                if b_label and v_data.get("label") != b_label:
                    continue
                results.append({a_ret: u_data.get(a_prop), b_ret: v_data.get(b_prop)})
                if limit is not None and len(results) >= limit:
                    break
            return results

        raise NotImplementedError(
            f"NetworkXGraphStore.query does not support this Cypher pattern:\n{cypher}"
        )

    def close(self) -> None:
        """No-op for the NetworkX backend (no external resources to release)."""

    # ------------------------------------------------------------------
    # Extra NetworkX-specific helpers
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """Expose the underlying :class:`networkx.DiGraph` for advanced use."""
        return self._graph

    def node_count(self) -> int:
        """Return the number of nodes currently in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return the number of edges currently in the graph."""
        return self._graph.number_of_edges()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunks(lst: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive *size*-sized slices of *lst*."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

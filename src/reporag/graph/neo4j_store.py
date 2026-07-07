"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j and provides an identical API via a NetworkX fallback so
callers never need to know which backend is active.

Architecture
------------
``GraphStore``
    Abstract base class that defines the backend-agnostic public interface:
    connection lifecycle, node/relationship CRUD, bulk ingestion helpers that
    consume the existing :class:`~src.reporag.graph.symbol_table.SymbolRecord`,
    :class:`~src.reporag.graph.call_graph.CallEdge`, and
    :class:`~src.reporag.graph.dependency_graph.DependencyEdge` data models, and
    graph-query helpers (find by id/name, call traversal, path queries, etc.).

``Neo4jStore``
    Primary backend. Wraps the official ``neo4j`` driver; bulk operations use
    ``UNWIND $rows ...`` batched into 1 000-row transactions so 10 000+ nodes are
    handled efficiently. Exposes an additional ``query(cypher, params)`` method
    for callers that intentionally depend on Neo4j-specific functionality.

``NetworkXStore``
    In-process fallback backed by a :class:`networkx.MultiDiGraph`. Implements
    every method of ``GraphStore`` using NetworkX traversal APIs so the same
    application code runs without a live Neo4j server (useful for tests,
    offline analysis, and CI pipelines).

``create_graph_store``
    Factory that returns the requested backend, defaulting to Neo4j.

Node labels
-----------
Derived from :attr:`SymbolRecord.type`:

- ``"class"``                    ->  ``Class``
- ``"function"`` / ``"method"`` ->  ``Function``
- ``"module"``                   ->  ``Module``
- anything else                 ->  ``Symbol``   (generic fallback)

Relationship types
------------------
- ``CALLS``    - populated from :class:`~src.reporag.graph.call_graph.CallEdge`
- ``IMPORTS``  - populated from
  :class:`~src.reporag.graph.dependency_graph.DependencyEdge`
- ``INHERITS`` - inferred from :attr:`SymbolRecord.bases`
- ``CONTAINS`` - inferred from :attr:`SymbolRecord.parent`

Usage::

    from src.reporag.graph.neo4j_store import create_graph_store

    store = create_graph_store("neo4j", uri="bolt://localhost:7687",
                               username="neo4j", password="secret")
    store.connect()
    store.bulk_ingest_symbol_records(symbol_table.records)
    store.bulk_ingest_call_edges(call_edges)
    store.bulk_ingest_dependency_edges(dep_edges)

    hits = store.outgoing_calls("examples.sample_repo.auth.authenticate_user")
    path = store.shortest_path("examples.sample_repo.app.login",
                               "examples.sample_repo.db.get_user")
    store.close()

    # Drop-in swap to the NetworkX fallback (no server needed):
    store = create_graph_store("networkx")
    store.connect()  # no-op for NetworkX
    ...
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # Imported at type-check time only so the module is importable even when
    # the ``neo4j`` package is not installed (e.g. NetworkX-only environment).
    import neo4j as _neo4j_pkg

import networkx as nx

from src.reporag.graph.call_graph import CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge
from src.reporag.graph.symbol_table import SymbolRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Neo4j bulk-insert transaction size - keeps memory pressure low while
# minimising the number of round-trips for 10 000+ node ingestion.
_BATCH_SIZE = 1_000

# Maps SymbolRecord.type -> Neo4j node label / NetworkX label attribute.
_LABEL_MAP: dict[str, str] = {
    "class": "Class",
    "function": "Function",
    "method": "Function",
    "module": "Module",
}
_DEFAULT_LABEL = "Symbol"


def _node_label(symbol_type: str) -> str:
    """Return the graph node label for a given *symbol_type* string."""
    return _LABEL_MAP.get(symbol_type, _DEFAULT_LABEL)


# ---------------------------------------------------------------------------
# Public data helpers
# ---------------------------------------------------------------------------


def _record_to_node_props(record: SymbolRecord) -> dict[str, Any]:
    """Convert a :class:`SymbolRecord` to a flat property dict for a graph node.

    Uses ``SymbolRecord.to_dict()`` as the canonical serialisation, but
    coerces list fields (``decorators``, ``bases``) to JSON strings so they
    survive Neo4j property constraints.  The ``id`` property is set to
    ``symbol_id`` (the stable, unique registry key).
    """
    props = record.to_dict()
    # Neo4j does not allow list-of-lists; serialise list fields to JSON.
    for key in ("decorators", "bases"):
        if isinstance(props.get(key), list):
            props[key] = json.dumps(props[key])
    props["id"] = record.symbol_id
    return props


def _call_edge_to_rel_props(edge: CallEdge) -> dict[str, Any]:
    """Convert a :class:`CallEdge` to a flat property dict for a ``CALLS`` relationship."""
    props = edge.to_dict()
    props.pop("caller", None)
    props.pop("callee", None)
    return props


def _dep_edge_to_rel_props(edge: DependencyEdge) -> dict[str, Any]:
    """Convert a :class:`DependencyEdge` to a flat property dict for an ``IMPORTS`` relationship."""
    return {
        "import_type": edge.import_type,
        "target_module": edge.target_module,
        "source_module": edge.source_module,
        "line": edge.line,
        "is_relative": edge.is_relative,
        "relative_level": edge.relative_level,
        "is_wildcard": edge.is_wildcard,
        "resolved": edge.resolved,
        # imported_names is a list-of-tuples; serialise for transport.
        "imported_names": json.dumps(edge.imported_names),
    }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    """Backend-agnostic interface for the code knowledge graph store.

    Every method in this class is implemented by both :class:`Neo4jStore`
    and :class:`NetworkXStore`, so application code can switch backends
    without modification.  Raw Cypher access (``query()``) is intentionally
    *not* part of this interface because it is Neo4j-specific; see
    :meth:`Neo4jStore.query` for that capability.

    Lifecycle::

        store.connect()
        try:
            ...
        finally:
            store.close()
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the backend (or initialise in-memory state)."""

    @abstractmethod
    def close(self) -> None:
        """Release all backend resources."""

    @abstractmethod
    def clear(self) -> None:
        """Delete every node and relationship in the graph."""

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def create_node(self, label: str, props: dict[str, Any]) -> str:
        """Create (or merge) a single node with *label* and *props*.

        Args:
            label: The node label (e.g. ``"Function"``, ``"Class"``).
            props: Property mapping.  **Must include an ``"id"`` key** that
                uniquely identifies the node; ``MERGE`` semantics are used so
                a second call with the same ``id`` is idempotent.

        Returns:
            The ``id`` value from *props*.

        Raises:
            ValueError: If ``props`` does not contain an ``"id"`` key.
        """

    @abstractmethod
    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        """Create (or merge) a directed relationship between two nodes.

        Args:
            from_id: The ``id`` of the source node.
            to_id:   The ``id`` of the target node.
            rel_type: Relationship type string (e.g. ``"CALLS"``).
            props:   Optional property mapping for the relationship.

        Raises:
            KeyError: If either *from_id* or *to_id* does not exist in the graph.
        """

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    @abstractmethod
    def bulk_create_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> list[str]:
        """Efficiently create (or merge) many nodes of the same *label*.

        Callers should prefer this over repeated :meth:`create_node` calls
        when inserting hundreds or thousands of nodes at once.  The Neo4j
        backend issues ``UNWIND`` transactions batched into
        :data:`_BATCH_SIZE` rows each.

        Args:
            label: Node label shared by all rows.
            rows:  List of property dicts, each containing an ``"id"`` key.

        Returns:
            Ordered list of ``id`` strings matching the input *rows*.
        """

    @abstractmethod
    def bulk_create_relationships(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """Efficiently create (or merge) many relationships in one call.

        Args:
            rows: List of dicts, each with keys:
                - ``"from_id"``  - source node id
                - ``"to_id"``    - target node id
                - ``"rel_type"`` - relationship type string
                - ``"props"``    - (optional) relationship property mapping
        """

    # ------------------------------------------------------------------
    # Domain-specific ingestion helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def ingest_symbol_record(self, record: SymbolRecord) -> None:
        """Ingest one :class:`~src.reporag.graph.symbol_table.SymbolRecord`.

        Creates the node plus any ``CONTAINS`` / ``INHERITS`` edges derivable
        from the record's ``parent`` and ``bases`` fields.
        """

    @abstractmethod
    def ingest_call_edge(self, edge: CallEdge) -> None:
        """Ingest one :class:`~src.reporag.graph.call_graph.CallEdge`.

        Creates stub nodes for the caller and callee if they are not already
        present, then creates a ``CALLS`` relationship between them.
        """

    @abstractmethod
    def ingest_dependency_edge(self, edge: DependencyEdge) -> None:
        """Ingest one :class:`~src.reporag.graph.dependency_graph.DependencyEdge`.

        Creates stub Module nodes if needed and a ``IMPORTS`` relationship.
        """

    @abstractmethod
    def bulk_ingest_symbol_records(self, records: list[SymbolRecord]) -> None:
        """Efficiently ingest many :class:`SymbolRecord` objects.

        Groups records by label and issues a single :meth:`bulk_create_nodes`
        call per label, then adds ``CONTAINS`` / ``INHERITS`` relationships in
        a second pass.
        """

    @abstractmethod
    def bulk_ingest_call_edges(self, edges: list[CallEdge]) -> None:
        """Efficiently ingest many :class:`CallEdge` objects."""

    @abstractmethod
    def bulk_ingest_dependency_edges(self, edges: list[DependencyEdge]) -> None:
        """Efficiently ingest many :class:`DependencyEdge` objects."""

    # ------------------------------------------------------------------
    # Graph query helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def find_by_id(self, node_id: str) -> dict[str, Any] | None:
        """Return the property dict for the node with ``id == node_id``, or ``None``."""

    @abstractmethod
    def find_by_qualified_name(self, qualified_name: str) -> dict[str, Any] | None:
        """Return the node whose ``qualified_name`` matches, or ``None``."""

    @abstractmethod
    def outgoing_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return callee nodes reachable via ``CALLS`` from *node_id*.

        Each dict has ``"node"`` (target node props) and ``"rel"``
        (relationship props) keys.
        """

    @abstractmethod
    def incoming_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return caller nodes that have a ``CALLS`` edge into *node_id*.

        Each dict has ``"node"`` (source node props) and ``"rel"``
        (relationship props) keys.
        """

    @abstractmethod
    def imported_modules(self, node_id: str) -> list[dict[str, Any]]:
        """Return nodes reachable via ``IMPORTS`` from *node_id*."""

    @abstractmethod
    def inheritance_relationships(self, node_id: str) -> list[dict[str, Any]]:
        """Return nodes reachable via ``INHERITS`` from *node_id*."""

    @abstractmethod
    def containment_hierarchy(self, node_id: str) -> list[dict[str, Any]]:
        """Return all nodes transitively reachable via ``CONTAINS`` from *node_id*."""

    @abstractmethod
    def shortest_path(self, from_id: str, to_id: str) -> list[dict[str, Any]]:
        """Return the shortest undirected path between two nodes.

        Each element in the returned list is the property dict of one node on
        the path (including the start and end nodes).  Returns an empty list
        when no path exists or when either node is not in the graph.
        """

    @abstractmethod
    def neighborhood(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """Return the subgraph within *depth* hops of *node_id*.

        Returns a dict::

            {
                "nodes": [<node props>, ...],
                "edges": [{"from": str, "to": str, "type": str, "props": dict}, ...],
            }
        """


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


class Neo4jStore(GraphStore):
    """Neo4j-backed graph store using the official ``neo4j`` Python driver.

    Bulk operations use ``UNWIND $rows ...`` batched into :data:`_BATCH_SIZE`
    rows per transaction, giving efficient throughput for 10 000+ node/edge
    ingestion without saturating the server's transaction memory.

    Args:
        uri:      Bolt/Neo4j URI, e.g. ``"bolt://localhost:7687"``.
        username: Neo4j database user.
        password: Neo4j database password.
        database: Target database (default ``"neo4j"``).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
    ) -> None:
        """Initialise connection parameters (driver created in :meth:`connect`)."""
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver: _neo4j_pkg.Driver | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create the Neo4j driver and verify connectivity.

        Raises:
            neo4j.exceptions.ServiceUnavailable: If the server is unreachable.
        """
        import neo4j  # noqa: PLC0415 - deferred to avoid import cost at module load

        self._driver = neo4j.GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        self._driver.verify_connectivity()
        self._ensure_constraints()
        logger.info("Connected to Neo4j at %s (database=%s)", self._uri, self._database)

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraint on node id if it does not exist."""

        self.query(
            """
            CREATE CONSTRAINT node_id_unique IF NOT EXISTS
            FOR (n)
            REQUIRE n.id IS UNIQUE
            """
        )

    def close(self) -> None:
        """Close the driver, releasing all underlying connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def clear(self) -> None:
        """Delete every node and relationship: ``MATCH (n) DETACH DELETE n``."""
        self.query("MATCH (n) DETACH DELETE n")
        logger.debug("Neo4j database cleared")

    # ------------------------------------------------------------------
    # Backend-specific: raw Cypher access
    # ------------------------------------------------------------------

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute an arbitrary Cypher query and return results as Python dicts.

        This method is intentionally **not** part of the :class:`GraphStore`
        interface because it is Neo4j-specific.  Use it only when the caller
        explicitly depends on Neo4j.

        Args:
            cypher: A Cypher query string.
            params: Optional parameter mapping (``$name`` placeholders).

        Returns:
            List of dicts where each key is a result column name and each
            value is the corresponding Python object (node dicts, primitives,
            lists, etc.).

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if self._driver is None:
            raise RuntimeError("Neo4jStore.connect() must be called before query()")
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, parameters=params or {})
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_in_batches(
        self,
        cypher: str,
        rows: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> None:
        """Execute *cypher* (which must use ``$rows``) in batches of *batch_size*.

        Reads :data:`_BATCH_SIZE` from the module at *call time* (not at class
        definition time) so that test code can temporarily lower it by patching
        the module-level constant.
        """
        if self._driver is None:
            raise RuntimeError(
                "Neo4jStore.connect() must be called before bulk operations"
            )
        # Defer reading _BATCH_SIZE to call time so patching works in tests.
        effective_size = batch_size if batch_size is not None else _BATCH_SIZE
        for start in range(0, len(rows), effective_size):
            batch = rows[start : start + effective_size]
            with self._driver.session(database=self._database) as session:
                session.run(cypher, parameters={"rows": batch})

    @staticmethod
    def _node_result_to_dict(record: Any, key: str = "n") -> dict[str, Any]:
        """Extract a node's property dict from a Neo4j result record."""
        node = record[key]
        return dict(node) if node is not None else {}

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, label: str, props: dict[str, Any]) -> str:
        """Merge a single node, setting all *props* on match or create.

        Raises:
            ValueError: If ``props`` has no ``"id"`` key.
        """
        if "id" not in props:
            raise ValueError(
                "props must contain an 'id' key to uniquely identify the node"
            )
        cypher = (
            f"MERGE (n:{label} {{id: $id}}) " "SET n += $props " "RETURN n.id AS id"
        )
        # Separate the MERGE key from the full SET payload so the MERGE
        # clause stays a single-property lookup (efficient with an index).
        self.query(cypher, {"id": props["id"], "props": props})
        return str(props["id"])

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        """Merge a directed relationship between two existing nodes.

        Raises:
            RuntimeError: If either node does not exist.
        """
        cypher = (
            "MATCH (a {id: $from_id}), (b {id: $to_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += $props"
        )
        if not self.find_by_id(from_id):
            raise KeyError(f"{from_id} not found")

        if not self.find_by_id(to_id):
            raise KeyError(f"{to_id} not found")

        self.query(cypher, {"from_id": from_id, "to_id": to_id, "props": props or {}})

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def bulk_create_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> list[str]:
        """UNWIND-based bulk node creation, batched for large datasets."""
        if not rows:
            return []
        cypher = (
            "UNWIND $rows AS row " f"MERGE (n:{label} {{id: row.id}}) " "SET n += row"
        )
        self._run_in_batches(cypher, rows)
        return [str(r["id"]) for r in rows]

    def bulk_create_relationships(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """UNWIND-based bulk relationship creation, batched for large datasets.

        Each element of *rows* must have ``"from_id"``, ``"to_id"``,
        ``"rel_type"``, and optionally ``"props"``.  All rows in a single
        batch must share the same ``rel_type`` because Cypher relationship
        types cannot be parameterised.  This method groups rows by type first.
        """
        if not rows:
            return
        # Group by rel_type so each UNWIND batch is type-homogeneous.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(row["rel_type"], []).append(row)

        for rel_type, typed_rows in by_type.items():
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a {id: row.from_id}), (b {id: row.to_id}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += row.props"
            )
            normalised = [
                {
                    "from_id": r["from_id"],
                    "to_id": r["to_id"],
                    "props": r.get("props") or {},
                }
                for r in typed_rows
            ]
            self._run_in_batches(cypher, normalised)

    # ------------------------------------------------------------------
    # Domain-specific ingestion helpers
    # ------------------------------------------------------------------

    def ingest_symbol_record(self, record: SymbolRecord) -> None:
        """Ingest one ``SymbolRecord``, creating the node plus structural edges."""
        label = _node_label(record.type)
        props = _record_to_node_props(record)
        self.create_node(label, props)

        # CONTAINS: parent -> child
        if record.parent:
            parent_stub_id = record.parent
            if not self.find_by_id(parent_stub_id):
                self.create_node(
                    "Symbol", {"id": parent_stub_id, "qualified_name": parent_stub_id}
                )
            self.create_relationship(parent_stub_id, record.symbol_id, "CONTAINS")

        # INHERITS: this class -> each base (only for class nodes)
        if record.type == "class":
            for base in record.bases:
                if not self.find_by_id(base):
                    self.create_node("Class", {"id": base, "qualified_name": base})
                self.create_relationship(record.symbol_id, base, "INHERITS")

    def ingest_call_edge(self, edge: CallEdge) -> None:
        """Ingest one ``CallEdge``, creating stub nodes and a ``CALLS`` relationship."""
        caller_id = edge.caller
        callee_id = edge.callee
        if not self.find_by_id(caller_id):
            self.create_node("Function", {"id": caller_id, "name": caller_id})
        if not self.find_by_id(callee_id):
            self.create_node("Function", {"id": callee_id, "name": callee_id})
        self.create_relationship(
            caller_id, callee_id, "CALLS", _call_edge_to_rel_props(edge)
        )

    def ingest_dependency_edge(self, edge: DependencyEdge) -> None:
        """Ingest one ``DependencyEdge``, creating Module stubs and an ``IMPORTS`` relationship."""
        source_id = edge.source
        target_id = edge.target
        if not self.find_by_id(source_id):
            self.create_node("Module", {"id": source_id, "file_path": source_id})
        if not self.find_by_id(target_id):
            self.create_node("Module", {"id": target_id, "file_path": target_id})
        self.create_relationship(
            source_id, target_id, "IMPORTS", _dep_edge_to_rel_props(edge)
        )

    def bulk_ingest_symbol_records(self, records: list[SymbolRecord]) -> None:
        """Group by label and bulk-create nodes, then add structural edges."""
        if not records:
            return

        # Pass 1: bulk create nodes grouped by label.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            label = _node_label(record.type)
            by_label.setdefault(label, []).append(_record_to_node_props(record))
        for label, rows in by_label.items():
            self.bulk_create_nodes(label, rows)

        # Pass 2: CONTAINS and INHERITS relationships.
        rel_rows: list[dict[str, Any]] = []
        known_ids = {r.symbol_id for r in records}

        for record in records:
            if record.parent:
                if record.parent not in known_ids:
                    # Create a lightweight stub for the parent if unknown.
                    self.create_node(
                        "Symbol",
                        {"id": record.parent, "qualified_name": record.parent},
                    )
                    known_ids.add(record.parent)
                rel_rows.append(
                    {
                        "from_id": record.parent,
                        "to_id": record.symbol_id,
                        "rel_type": "CONTAINS",
                        "props": {},
                    }
                )

            if record.type == "class":
                for base in record.bases:
                    if base not in known_ids:
                        self.create_node("Class", {"id": base, "qualified_name": base})
                        known_ids.add(base)
                    rel_rows.append(
                        {
                            "from_id": record.symbol_id,
                            "to_id": base,
                            "rel_type": "INHERITS",
                            "props": {},
                        }
                    )

        self.bulk_create_relationships(rel_rows)

    def bulk_ingest_call_edges(self, edges: list[CallEdge]) -> None:
        """Ensure caller/callee nodes exist, then bulk-create ``CALLS`` relationships."""
        if not edges:
            return
        # Collect all node ids that need to exist.
        existing_ids: set[str] = set()
        all_node_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for edge in edges:
            for nid, name in ((edge.caller, edge.caller), (edge.callee, edge.callee)):
                if nid not in seen:
                    seen.add(nid)
                    all_node_rows.append({"id": nid, "name": name})
        _ = existing_ids  # nodes are MERGE-d so we don't need to pre-check
        self.bulk_create_nodes("Function", all_node_rows)

        rel_rows = [
            {
                "from_id": edge.caller,
                "to_id": edge.callee,
                "rel_type": "CALLS",
                "props": _call_edge_to_rel_props(edge),
            }
            for edge in edges
        ]
        self.bulk_create_relationships(rel_rows)

    def bulk_ingest_dependency_edges(self, edges: list[DependencyEdge]) -> None:
        """Ensure source/target Module nodes exist, then bulk-create ``IMPORTS`` relationships."""
        if not edges:
            return
        seen: set[str] = set()
        node_rows: list[dict[str, Any]] = []
        for edge in edges:
            for fid in (edge.source, edge.target):
                if fid not in seen:
                    seen.add(fid)
                    node_rows.append({"id": fid, "file_path": fid})
        self.bulk_create_nodes("Module", node_rows)

        rel_rows = [
            {
                "from_id": edge.source,
                "to_id": edge.target,
                "rel_type": "IMPORTS",
                "props": _dep_edge_to_rel_props(edge),
            }
            for edge in edges
        ]
        self.bulk_create_relationships(rel_rows)

    # ------------------------------------------------------------------
    # Graph query helpers
    # ------------------------------------------------------------------

    def find_by_id(self, node_id: str) -> dict[str, Any] | None:
        """Return the property dict for the node with ``id == node_id``, or ``None``."""
        results = self.query("MATCH (n {id: $id}) RETURN n", {"id": node_id})
        if not results:
            return None
        node = results[0].get("n")
        return dict(node) if node is not None else None

    def find_by_qualified_name(self, qualified_name: str) -> dict[str, Any] | None:
        """Return the node whose ``qualified_name`` matches, or ``None``."""
        results = self.query(
            "MATCH (n {qualified_name: $qname}) RETURN n",
            {"qname": qualified_name},
        )
        if not results:
            return None
        node = results[0].get("n")
        return dict(node) if node is not None else None

    def outgoing_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return callee nodes and relationship props for all ``CALLS`` from *node_id*."""
        results = self.query(
            "MATCH (n {id: $id})-[r:CALLS]->(m) RETURN m, r",
            {"id": node_id},
        )
        return [{"node": dict(row["m"]), "rel": dict(row["r"])} for row in results]

    def incoming_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return caller nodes and relationship props for all ``CALLS`` into *node_id*."""
        results = self.query(
            "MATCH (m)-[r:CALLS]->(n {id: $id}) RETURN m, r",
            {"id": node_id},
        )
        return [{"node": dict(row["m"]), "rel": dict(row["r"])} for row in results]

    def imported_modules(self, node_id: str) -> list[dict[str, Any]]:
        """Return nodes reachable via a single ``IMPORTS`` hop from *node_id*."""
        results = self.query(
            "MATCH (n {id: $id})-[r:IMPORTS]->(m) RETURN m, r",
            {"id": node_id},
        )
        return [{"node": dict(row["m"]), "rel": dict(row["r"])} for row in results]

    def inheritance_relationships(self, node_id: str) -> list[dict[str, Any]]:
        """Return base-class nodes reachable via ``INHERITS`` from *node_id*."""
        results = self.query(
            "MATCH (n {id: $id})-[r:INHERITS]->(m) RETURN m, r",
            {"id": node_id},
        )
        return [{"node": dict(row["m"]), "rel": dict(row["r"])} for row in results]

    def containment_hierarchy(self, node_id: str) -> list[dict[str, Any]]:
        """Return all nodes reachable via transitive ``CONTAINS`` from *node_id*."""
        results = self.query(
            "MATCH (n {id: $id})-[:CONTAINS*]->(m) RETURN m",
            {"id": node_id},
        )
        return [{"node": dict(row["m"])} for row in results]

    def shortest_path(self, from_id: str, to_id: str) -> list[dict[str, Any]]:
        """Return nodes on the shortest undirected path between *from_id* and *to_id*.

        Returns an empty list when no path exists.
        """
        results = self.query(
            "MATCH p = shortestPath((a {id: $from_id})-[*]-(b {id: $to_id})) "
            "RETURN nodes(p) AS path_nodes",
            {"from_id": from_id, "to_id": to_id},
        )
        if not results:
            return []
        path_nodes = results[0].get("path_nodes", [])
        return [dict(n) for n in path_nodes]

    def neighborhood(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """Return the subgraph within *depth* hops of *node_id*.

        Returns::

            {"nodes": [<prop dict>, ...], "edges": [{"from": str, "to": str,
             "type": str, "props": dict}, ...]}
        """
        results = self.query(
            "MATCH (n {id: $id})-[r*1..$depth]-(m) "
            "RETURN collect(DISTINCT m) AS nbrs, "
            "       collect(DISTINCT r) AS rels, "
            "       n AS center",
            {"id": node_id, "depth": depth},
        )
        if not results:
            center = self.find_by_id(node_id)
            return {"nodes": [center] if center else [], "edges": []}

        row = results[0]
        center = dict(row["center"]) if row.get("center") else {}
        nodes: list[dict[str, Any]] = [center]
        nodes_seen: set[str] = {str(center.get("id", ""))}

        for nbr in row.get("nbrs", []):
            n = dict(nbr)
            nid = str(n.get("id", ""))
            if nid not in nodes_seen:
                nodes.append(n)
                nodes_seen.add(nid)

        # ``r`` in a variable-length pattern is a list of relationship lists.
        edges: list[dict[str, Any]] = []
        for rel_group in row.get("rels", []):
            # Each element may itself be a list when the path length > 1.
            rel_list = rel_group if isinstance(rel_group, list) else [rel_group]
            for rel in rel_list:
                if rel is None:
                    continue
                edges.append(
                    {
                        "from": str(rel.start_node.get("id", "")),  # type: ignore[union-attr]
                        "to": str(rel.end_node.get("id", "")),  # type: ignore[union-attr]
                        "type": rel.type,  # type: ignore[union-attr]
                        "props": dict(rel),  # type: ignore[arg-type]
                    }
                )

        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# NetworkX fallback backend
# ---------------------------------------------------------------------------


class NetworkXStore(GraphStore):
    """In-process fallback graph store backed by :class:`networkx.MultiDiGraph`.

    Exposes the identical :class:`GraphStore` interface so callers can switch
    from Neo4j to NetworkX without any code changes.  All query helpers are
    implemented using NetworkX traversal APIs -- no Cypher involved.

    Because the graph lives entirely in memory, ``connect()`` and ``close()``
    are no-ops, and there is no notion of transaction batching (``bulk_*``
    methods simply iterate).

    This backend is well-suited for:

    - Unit / integration tests that should not require a live database.
    - Offline analysis of a repository.
    - CI pipelines where standing up a Neo4j container is impractical.
    """

    def __init__(self) -> None:
        """Initialise the empty in-memory graph."""
        # MultiDiGraph supports multiple edge types (CALLS, IMPORTS, ...)
        # between the same pair of nodes.
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """No-op for the in-memory backend."""

    def close(self) -> None:
        """No-op for the in-memory backend."""

    def clear(self) -> None:
        """Remove all nodes and edges from the in-memory graph."""
        self._graph.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_props(self, node_id: str) -> dict[str, Any] | None:
        """Return the property dict for *node_id*, or ``None`` if absent."""
        if node_id not in self._graph:
            return None
        return dict(self._graph.nodes[node_id])

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, label: str, props: dict[str, Any]) -> str:
        """Add or update a node in the in-memory graph.

        Raises:
            ValueError: If ``props`` has no ``"id"`` key.
        """
        if "id" not in props:
            raise ValueError(
                "props must contain an 'id' key to uniquely identify the node"
            )
        node_id = str(props["id"])
        if node_id in self._graph:
            # MERGE semantics: update props on an existing node.
            self._graph.nodes[node_id].update(props)
            self._graph.nodes[node_id]["label"] = label
        else:
            self._graph.add_node(node_id, label=label, **props)
        return node_id

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        """Add a directed edge (with MERGE semantics on key ``rel_type``).

        Raises:
            KeyError: If either node is not in the graph.
        """
        if from_id not in self._graph:
            raise KeyError(f"Source node '{from_id}' not found in the graph")
        if to_id not in self._graph:
            raise KeyError(f"Target node '{to_id}' not found in the graph")
        # Check for an existing edge of this type to provide MERGE semantics.
        existing_keys = self._graph.get_edge_data(from_id, to_id) or {}
        for key, edge_data in existing_keys.items():
            if edge_data.get("rel_type") == rel_type:
                # Update props on the existing edge.
                if props:
                    self._graph[from_id][to_id][key].update(props)
                return
        self._graph.add_edge(
            from_id, to_id, key=rel_type, rel_type=rel_type, **(props or {})
        )

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def bulk_create_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> list[str]:
        """Iterate and call :meth:`create_node` for each row."""
        return [self.create_node(label, row) for row in rows]

    def bulk_create_relationships(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """Iterate and call :meth:`create_relationship` for each row."""
        for row in rows:
            self.create_relationship(
                row["from_id"],
                row["to_id"],
                row["rel_type"],
                row.get("props"),
            )

    # ------------------------------------------------------------------
    # Domain-specific ingestion helpers
    # ------------------------------------------------------------------

    def ingest_symbol_record(self, record: SymbolRecord) -> None:
        """Ingest one ``SymbolRecord`` into the in-memory graph."""
        label = _node_label(record.type)
        props = _record_to_node_props(record)
        self.create_node(label, props)

        if record.parent:
            if record.parent not in self._graph:
                self.create_node(
                    "Symbol", {"id": record.parent, "qualified_name": record.parent}
                )
            self.create_relationship(record.parent, record.symbol_id, "CONTAINS")

        if record.type == "class":
            for base in record.bases:
                if base not in self._graph:
                    self.create_node("Class", {"id": base, "qualified_name": base})
                self.create_relationship(record.symbol_id, base, "INHERITS")

    def ingest_call_edge(self, edge: CallEdge) -> None:
        """Ingest one ``CallEdge`` into the in-memory graph."""
        caller_id = edge.caller
        callee_id = edge.callee
        if caller_id not in self._graph:
            self.create_node("Function", {"id": caller_id, "name": caller_id})
        if callee_id not in self._graph:
            self.create_node("Function", {"id": callee_id, "name": callee_id})
        self.create_relationship(
            caller_id, callee_id, "CALLS", _call_edge_to_rel_props(edge)
        )

    def ingest_dependency_edge(self, edge: DependencyEdge) -> None:
        """Ingest one ``DependencyEdge`` into the in-memory graph."""
        source_id = edge.source
        target_id = edge.target
        if source_id not in self._graph:
            self.create_node("Module", {"id": source_id, "file_path": source_id})
        if target_id not in self._graph:
            self.create_node("Module", {"id": target_id, "file_path": target_id})
        self.create_relationship(
            source_id, target_id, "IMPORTS", _dep_edge_to_rel_props(edge)
        )

    def bulk_ingest_symbol_records(self, records: list[SymbolRecord]) -> None:
        """Bulk-ingest many ``SymbolRecord`` objects, collecting structural edges."""
        known_ids: set[str] = set()
        for record in records:
            label = _node_label(record.type)
            self.create_node(label, _record_to_node_props(record))
            known_ids.add(record.symbol_id)

        # Structural edges in a second pass so parent nodes are guaranteed to exist.
        for record in records:
            if record.parent:
                if record.parent not in known_ids and record.parent not in self._graph:
                    self.create_node(
                        "Symbol",
                        {"id": record.parent, "qualified_name": record.parent},
                    )
                    known_ids.add(record.parent)
                self.create_relationship(record.parent, record.symbol_id, "CONTAINS")

            if record.type == "class":
                for base in record.bases:
                    if base not in known_ids and base not in self._graph:
                        self.create_node("Class", {"id": base, "qualified_name": base})
                        known_ids.add(base)
                    self.create_relationship(record.symbol_id, base, "INHERITS")

    def bulk_ingest_call_edges(self, edges: list[CallEdge]) -> None:
        """Bulk-ingest ``CallEdge`` objects into the in-memory graph."""
        for edge in edges:
            self.ingest_call_edge(edge)

    def bulk_ingest_dependency_edges(self, edges: list[DependencyEdge]) -> None:
        """Bulk-ingest ``DependencyEdge`` objects into the in-memory graph."""
        for edge in edges:
            self.ingest_dependency_edge(edge)

    # ------------------------------------------------------------------
    # Graph query helpers
    # ------------------------------------------------------------------

    def find_by_id(self, node_id: str) -> dict[str, Any] | None:
        """Return the property dict for the node with ``id == node_id``, or ``None``."""
        return self._node_props(node_id)

    def find_by_qualified_name(self, qualified_name: str) -> dict[str, Any] | None:
        """Return the first node whose ``qualified_name`` property matches, or ``None``."""
        for _node_id, data in self._graph.nodes(data=True):
            if data.get("qualified_name") == qualified_name:
                return dict(data)
        return None

    def outgoing_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return callee nodes and ``CALLS`` relationship props from *node_id*."""
        results: list[dict[str, Any]] = []
        for _, to_id, edge_data in self._graph.out_edges(node_id, data=True):
            if edge_data.get("rel_type") == "CALLS":
                target_props = self._node_props(to_id) or {}
                rel_props = {k: v for k, v in edge_data.items() if k != "rel_type"}
                results.append({"node": target_props, "rel": rel_props})
        return results

    def incoming_calls(self, node_id: str) -> list[dict[str, Any]]:
        """Return caller nodes and ``CALLS`` relationship props into *node_id*."""
        results: list[dict[str, Any]] = []
        for from_id, _, edge_data in self._graph.in_edges(node_id, data=True):
            if edge_data.get("rel_type") == "CALLS":
                source_props = self._node_props(from_id) or {}
                rel_props = {k: v for k, v in edge_data.items() if k != "rel_type"}
                results.append({"node": source_props, "rel": rel_props})
        return results

    def imported_modules(self, node_id: str) -> list[dict[str, Any]]:
        """Return nodes reachable via a single ``IMPORTS`` hop from *node_id*."""
        results: list[dict[str, Any]] = []
        for _, to_id, edge_data in self._graph.out_edges(node_id, data=True):
            if edge_data.get("rel_type") == "IMPORTS":
                target_props = self._node_props(to_id) or {}
                rel_props = {k: v for k, v in edge_data.items() if k != "rel_type"}
                results.append({"node": target_props, "rel": rel_props})
        return results

    def inheritance_relationships(self, node_id: str) -> list[dict[str, Any]]:
        """Return base-class nodes reachable via ``INHERITS`` from *node_id*."""
        results: list[dict[str, Any]] = []
        for _, to_id, edge_data in self._graph.out_edges(node_id, data=True):
            if edge_data.get("rel_type") == "INHERITS":
                target_props = self._node_props(to_id) or {}
                rel_props = {k: v for k, v in edge_data.items() if k != "rel_type"}
                results.append({"node": target_props, "rel": rel_props})
        return results

    def containment_hierarchy(self, node_id: str) -> list[dict[str, Any]]:
        """Return all nodes reachable via transitive ``CONTAINS`` from *node_id*."""
        results: list[dict[str, Any]] = []
        visited: set[str] = {node_id}
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for _, to_id, edge_data in self._graph.out_edges(current, data=True):
                if edge_data.get("rel_type") == "CONTAINS" and to_id not in visited:
                    visited.add(to_id)
                    props = self._node_props(to_id) or {}
                    results.append({"node": props})
                    queue.append(to_id)
        return results

    def shortest_path(self, from_id: str, to_id: str) -> list[dict[str, Any]]:
        """Return nodes on the shortest undirected path between *from_id* and *to_id*.

        Uses NetworkX's bidirectional shortest-path algorithm on the
        underlying undirected view of the graph.  Returns an empty list when
        either node is absent or no path exists.
        """
        if from_id not in self._graph or to_id not in self._graph:
            return []
        undirected = self._graph.to_undirected()
        try:
            path: list[str] = nx.shortest_path(undirected, from_id, to_id)
        except nx.NetworkXNoPath:
            return []
        return [self._node_props(n) or {} for n in path]

    def neighborhood(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """Return the subgraph within *depth* hops of *node_id*.

        Uses BFS over the undirected view so edges are traversed in both
        directions.  Returns ``{"nodes": [], "edges": []}`` when *node_id* is
        not in the graph.
        """
        if node_id not in self._graph:
            return {"nodes": [], "edges": []}

        # BFS to collect all nodes within `depth` hops.
        undirected = self._graph.to_undirected(as_view=True)
        visited: dict[str, int] = {node_id: 0}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        while queue:
            current, dist = queue.popleft()
            if dist >= depth:
                continue
            for nbr in undirected.neighbors(current):
                if nbr not in visited:
                    visited[nbr] = dist + 1
                    queue.append((nbr, dist + 1))

        nodes = [self._node_props(n) or {} for n in visited]

        # Collect directed edges whose both endpoints are within the subgraph.
        edges: list[dict[str, Any]] = []
        for from_id, to_id, edge_data in self._graph.edges(visited.keys(), data=True):
            if to_id in visited:
                edges.append(
                    {
                        "from": from_id,
                        "to": to_id,
                        "type": edge_data.get("rel_type", ""),
                        "props": {
                            k: v for k, v in edge_data.items() if k != "rel_type"
                        },
                    }
                )

        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

Backend = Literal["neo4j", "networkx"]


def create_graph_store(
    backend: Backend = "neo4j",
    **kwargs: Any,
) -> GraphStore:
    """Instantiate and return a :class:`GraphStore` for the requested *backend*.

    Args:
        backend: ``"neo4j"`` (default) or ``"networkx"``.
        **kwargs: Passed verbatim to the backend constructor.
            For ``"neo4j"``: ``uri``, ``username``, ``password``, ``database``.
            For ``"networkx"``: no arguments accepted.

    Returns:
        A :class:`GraphStore` instance.  Call :meth:`~GraphStore.connect` before
        performing any operations.

    Raises:
        ValueError: If *backend* is not a recognised value.

    Example::

        store = create_graph_store("neo4j", uri="bolt://localhost:7687",
                                   username="neo4j", password="s3cr3t")
        store.connect()
        ...
        store.close()

        # Swap to the in-memory fallback:
        store = create_graph_store("networkx")
        store.connect()  # no-op
        ...
    """
    if backend == "neo4j":
        return Neo4jStore(**kwargs)
    if backend == "networkx":
        return NetworkXStore()
    raise ValueError(f"Unknown backend {backend!r}. Choose 'neo4j' or 'networkx'.")

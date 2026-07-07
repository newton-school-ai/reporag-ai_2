"""Neo4j graph store with a Cypher query layer (and a NetworkX fallback).

This is the persistence layer of the code knowledge graph. The three upstream
builders each produce one facet of the graph:

* Issue 9 -- the **call graph** (:class:`~src.reporag.graph.call_graph.CallEdge`):
  ``caller -> callee`` edges.
* Issue 10 -- the **dependency graph**
  (:class:`~src.reporag.graph.dependency_graph.DependencyEdge`): module import
  edges.
* Issue 11 -- the **symbol table**
  (:class:`~src.reporag.graph.symbol_table.SymbolRecord`): the node metadata
  (file, line range, type, signature, docstring).

:func:`GraphStore` stitches those three into a single labelled property graph and
persists it, then answers structural questions -- neighbours, shortest paths,
subgraph extraction -- that pure vector search cannot.

Schema
------
Every node carries a shared ``:CodeNode`` label (with a uniqueness constraint on
``id`` so lookups and ``MERGE`` are indexed) plus one *specific* label:

* ``Function`` -- a function or method (``is_method`` distinguishes them).
* ``Class``    -- a class definition.
* ``Module``   -- a file / dotted module (project-local or external).

Relationships:

* ``CALLS``    -- ``Function -> Function`` / ``Function -> Class`` (constructor).
* ``IMPORTS``  -- ``Module -> Module``.
* ``INHERITS`` -- ``Class -> Class`` (resolved base classes only).
* ``CONTAINS`` -- lexical containment: ``Module -> top-level symbol`` and
  ``Class -> method`` / nested definition.

Two interchangeable backends
----------------------------
Both implement the same :class:`BaseGraphStore` interface, so the exact same test
suite and application code run against either:

* :class:`Neo4jGraphStore` -- the real thing: batched ``UNWIND`` transactions for
  fast bulk insert (10k+ nodes), a uniqueness constraint, connection retry with
  exponential backoff, and arbitrary Cypher via :meth:`~BaseGraphStore.query`.
* :class:`NetworkXGraphStore` -- an in-memory ``MultiDiGraph`` fallback requiring
  no server. It powers unit tests and degraded-mode operation; the structured
  helpers (:meth:`~BaseGraphStore.get_neighbors`,
  :meth:`~BaseGraphStore.shortest_path`, :meth:`~BaseGraphStore.subgraph`) work
  identically, while raw :meth:`~BaseGraphStore.query` (Cypher) is Neo4j-only.

Use the :func:`GraphStore` factory to pick a backend, with automatic fallback to
NetworkX when Neo4j cannot be reached::

    from src.reporag.graph.neo4j_store import GraphStore

    store = GraphStore(uri="bolt://localhost:7687",
                       user="neo4j", password="reporag123")
    store.persist_graph(call_edges, dep_edges, symbols)
    rows = store.query(
        "MATCH (f:Function)-[:CALLS]->(g:Function) "
        "RETURN f.name AS caller, g.name AS callee LIMIT 5"
    )
    for r in rows:
        print(r["caller"], "->", r["callee"])
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

from src.reporag.graph.call_graph import MODULE_SCOPE, CallEdge
from src.reporag.graph.dependency_graph import DependencyEdge, DependencyGraphResult
from src.reporag.graph.symbol_table import SymbolRecord, SymbolTable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Label every node shares; the uniqueness constraint / index lives here so that
#: node lookups and relationship ``MATCH`` are O(1) regardless of specific label.
SHARED_LABEL = "CodeNode"

LABEL_FUNCTION = "Function"
LABEL_CLASS = "Class"
LABEL_MODULE = "Module"
LABEL_VARIABLE = "Variable"

EDGE_CALLS = "CALLS"
EDGE_IMPORTS = "IMPORTS"
EDGE_INHERITS = "INHERITS"
EDGE_CONTAINS = "CONTAINS"

VALID_EDGE_TYPES = frozenset({EDGE_CALLS, EDGE_IMPORTS, EDGE_INHERITS, EDGE_CONTAINS})

# Symbol-table ``type`` -> specific node label.
_TYPE_TO_LABEL = {
    "function": LABEL_FUNCTION,
    "method": LABEL_FUNCTION,
    "class": LABEL_CLASS,
    "variable": LABEL_VARIABLE,
}

# Module node ids are namespaced so they can never collide with a symbol id
# (which is a dotted qualified name such as ``pkg.mod.func``).
_MODULE_ID_PREFIX = "module::"

# A Cypher identifier (label / relationship type) we are willing to interpolate
# into a query string. Everything the store injects is validated against this to
# make injection structurally impossible.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

Direction = Literal["out", "in", "both"]


# ---------------------------------------------------------------------------
# Graph value types
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the code graph: a stable ``id``, one label, and properties.

    Attributes:
        id: Globally unique node key. For code symbols this is the symbol
            table's ``symbol_id`` (dotted qualified name); for modules it is
            ``module::<dotted-name>``.
        label: The specific label -- ``Function``, ``Class``, ``Module`` or
            ``Variable``. (In Neo4j the node also carries the shared
            ``CodeNode`` label.)
        properties: JSON-primitive properties (name, file_path, line range,
            signature, docstring, ...). Always includes ``id``.
    """

    id: str
    label: str
    properties: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        name = self.properties.get("name", self.id)
        return f"GraphNode({self.label} {name!r} id={self.id!r})"


@dataclass
class GraphEdge:
    """A directed relationship ``source -> target`` of a given ``type``.

    Attributes:
        type: One of :data:`VALID_EDGE_TYPES`.
        source: ``id`` of the source :class:`GraphNode`.
        target: ``id`` of the target :class:`GraphNode`.
        properties: JSON-primitive edge metadata (e.g. ``call_site_lines`` for
            a ``CALLS`` edge, ``imported_names`` for an ``IMPORTS`` edge).
    """

    type: str
    source: str
    target: str
    properties: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GraphEdge({self.source} -[{self.type}]-> {self.target})"


@dataclass
class Subgraph:
    """An extracted slice of the graph: a set of nodes and the edges among them."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class PersistSummary:
    """Counts returned by :meth:`BaseGraphStore.persist_graph`."""

    nodes: int
    edges: int

    def __repr__(self) -> str:
        return f"PersistSummary(nodes={self.nodes}, edges={self.edges})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_name(file_path: str) -> str:
    """Return the dotted module name for *file_path*.

    Mirrors :func:`src.reporag.graph.symbol_table._module_name` /
    :class:`src.reporag.graph.call_graph._ModuleIndex` so a file resolves to the
    same module name everywhere in the knowledge graph (POSIX semantics on every
    host; ``__init__`` folds into its package directory).
    """
    pure = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(pure.parts[:-1]) + [pure.stem]
    if pure.stem == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p not in ("", ".", "/"))


def _module_node_id(module: str) -> str:
    """Return the namespaced node id for a dotted *module* name."""
    return f"{_MODULE_ID_PREFIX}{module}"


def _label_for_type(symbol_type: str) -> str:
    """Map a symbol-table ``type`` to a specific node label."""
    return _TYPE_TO_LABEL.get(symbol_type, symbol_type.capitalize() or "Symbol")


def _safe_ident(value: str) -> str:
    """Return *value* if it is a valid Cypher identifier, else raise.

    Guards every label / relationship type the store interpolates into a query
    string, so a malformed symbol ``type`` can never become Cypher injection.
    """
    if not _IDENT_RE.fullmatch(value):
        raise ValueError(f"Unsafe Cypher identifier: {value!r}")
    return value


def _chunks(seq: Sequence[dict], size: int) -> Iterator[list[dict]]:
    """Yield *seq* in lists of at most *size* items."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _serialise_imported_names(pairs: Iterable[tuple[str, str | None]]) -> list[str]:
    """Flatten ``(name, alias)`` pairs to ``["name", "name as alias"]`` strings.

    Neo4j properties cannot hold a list of tuples, so the import bindings are
    stored as plain strings.
    """
    out: list[str] = []
    for name, alias in pairs:
        out.append(name if alias is None else f"{name} as {alias}")
    return out


# ---------------------------------------------------------------------------
# Graph builder: (symbols, call edges, dep edges) -> (nodes, edges)
# ---------------------------------------------------------------------------


class _GraphBuilder:
    """Turns the three upstream graph facets into deduplicated nodes and edges.

    - **Why it exists**: Keeps the (non-trivial) mapping from symbols/edges to a
      labelled property graph in one backend-agnostic place, so
      :class:`Neo4jGraphStore` and :class:`NetworkXGraphStore` persist an
      identical graph.
    - **Algorithm**: Builds symbol and module nodes first, indexes them by
      qualified name and by ``(file, local-qualified-name)``, then resolves each
      call/import/inheritance/containment relationship against those indexes.
    - **Edge cases**: Unresolved calls (no project target), dotted/generic base
      classes, and ambiguous base names are dropped rather than guessed -- the
      graph never contains a fabricated edge. Duplicate ``CALLS`` between the
      same pair collapse into one edge that accumulates every call-site line.
    - **Correctness choice**: Node ids come straight from the symbol table's
      ``symbol_id`` (and namespaced module ids), so relationships line up with
      the exact nodes the symbols produced.
    """

    def __init__(
        self,
        symbols: SymbolTable | Iterable[SymbolRecord] | None,
        call_edges: Iterable[CallEdge] | None,
        dep_edges: Iterable[DependencyEdge] | DependencyGraphResult | None,
    ) -> None:
        """Normalise the three (possibly ``None``) inputs into concrete lists."""
        self._records: list[SymbolRecord] = [] if symbols is None else list(symbols)
        self._calls: list[CallEdge] = [] if call_edges is None else list(call_edges)
        if dep_edges is None:
            self._deps: list[DependencyEdge] = []
        elif isinstance(dep_edges, DependencyGraphResult):
            self._deps = list(dep_edges.edges)
        else:
            self._deps = list(dep_edges)

        self._nodes: dict[str, GraphNode] = {}
        # Accumulate edges keyed by (type, source, target) so duplicates merge.
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}

        # Indexes populated while building symbol nodes.
        self._by_qualified: dict[str, str] = {}  # qualified_name -> symbol_id
        self._by_file_local: dict[tuple[str, str], str] = {}  # (file, local) -> id
        self._classes_by_name: dict[str, list[str]] = {}  # name -> [class ids]
        self._classes_by_module: dict[tuple[str, str], str] = {}  # (mod,name)->id
        self._file_to_module: dict[str, str] = {}

    # ------------------------------------------------------------------

    def build(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return the full ``(nodes, edges)`` for the assembled graph."""
        self._build_symbol_nodes()
        self._build_contains_edges()
        self._build_call_edges()
        self._build_import_edges()
        self._build_inherits_edges()
        return list(self._nodes.values()), list(self._edges.values())

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _build_symbol_nodes(self) -> None:
        """Create one node per symbol record and populate the lookup indexes."""
        for rec in self._records:
            node_id = rec.symbol_id
            self._nodes[node_id] = GraphNode(
                id=node_id,
                label=_label_for_type(rec.type),
                properties={
                    "id": node_id,
                    "name": rec.name,
                    "qualified_name": rec.qualified_name,
                    "type": rec.type,
                    "file_path": rec.file_path,
                    "module": rec.module,
                    "start_line": rec.start_line,
                    "end_line": rec.end_line,
                    "signature": rec.signature,
                    "docstring": rec.docstring,
                    "parent": rec.parent,
                    "decorators": list(rec.decorators),
                    "bases": list(rec.bases),
                    "is_async": rec.is_async,
                    "is_method": rec.type == "method",
                    "language": rec.language,
                },
            )

            self._by_qualified[rec.qualified_name] = node_id
            self._by_file_local[(rec.file_path, self._local_qname(rec))] = node_id
            if rec.module:
                self._file_to_module.setdefault(rec.file_path, rec.module)
            if rec.type == "class":
                self._classes_by_name.setdefault(rec.name, []).append(node_id)
                self._classes_by_module[(rec.module, rec.name)] = node_id
            # Ensure the module that owns this symbol exists as a node.
            self._ensure_module_node(rec.module, file_path=rec.file_path)

    def _ensure_module_node(
        self,
        module: str,
        *,
        file_path: str | None = None,
        external: bool | None = None,
    ) -> str | None:
        """Create or enrich a ``Module`` node, returning its id (``None`` if blank).

        Later calls with better information (a concrete ``file_path`` or a known
        ``external`` flag) fill in fields a bare first mention left unset.
        """
        if not module:
            return None
        node_id = _module_node_id(module)
        node = self._nodes.get(node_id)
        if node is None:
            node = GraphNode(
                id=node_id,
                label=LABEL_MODULE,
                properties={
                    "id": node_id,
                    "name": module,
                    "module": module,
                    "file_path": file_path,
                    "external": bool(external) if external is not None else False,
                },
            )
            self._nodes[node_id] = node
        else:
            if file_path is not None and node.properties.get("file_path") is None:
                node.properties["file_path"] = file_path
            if external is not None and not external:
                node.properties["external"] = False
        return node_id

    @staticmethod
    def _local_qname(rec: SymbolRecord) -> str:
        """Return the file-local qualified name (module prefix stripped).

        The call graph names callers/callees with file-local qualified names
        (``User.__init__``), whereas the symbol table stores module-prefixed
        ones (``pkg.db.User.__init__``); stripping the prefix lets the two line
        up when resolving ``CALLS`` endpoints.
        """
        prefix = f"{rec.module}."
        if rec.module and rec.qualified_name.startswith(prefix):
            return rec.qualified_name[len(prefix) :]
        return rec.qualified_name

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _add_edge(
        self, edge_type: str, source: str, target: str, properties: dict
    ) -> None:
        """Insert an edge, merging into an existing ``(type, src, tgt)`` if any."""
        key = (edge_type, source, target)
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = GraphEdge(edge_type, source, target, properties)
            return
        # Merge: accumulate call-site lines / counts so repeated calls collapse.
        if "call_site_lines" in properties:
            existing.properties.setdefault("call_site_lines", [])
            existing.properties["call_site_lines"].extend(properties["call_site_lines"])
            existing.properties["count"] = existing.properties.get("count", 1) + 1
            existing.properties["is_recursive"] = existing.properties.get(
                "is_recursive", False
            ) or properties.get("is_recursive", False)

    def _build_contains_edges(self) -> None:
        """Link each symbol to its container (module or enclosing class/function)."""
        for rec in self._records:
            child_id = rec.symbol_id
            if rec.parent is None:
                parent_id = _module_node_id(rec.module) if rec.module else None
            else:
                parent_id = self._by_qualified.get(rec.parent)
            if parent_id is not None and parent_id != child_id:
                self._add_edge(EDGE_CONTAINS, parent_id, child_id, {})

    def _build_call_edges(self) -> None:
        """Turn resolved :class:`CallEdge` objects into ``CALLS`` relationships."""
        for ce in self._calls:
            source_id = self._call_source_id(ce)
            if source_id is None:
                continue
            if not ce.resolved or ce.callee_file is None:
                continue  # unresolved: no project target to point at
            target_id = self._by_file_local.get((ce.callee_file, ce.callee))
            if target_id is None:
                continue
            self._add_edge(
                EDGE_CALLS,
                source_id,
                target_id,
                {
                    "call_type": ce.call_type,
                    "resolution": ce.resolution,
                    "call_site_lines": [ce.call_site_line],
                    "count": 1,
                    "is_recursive": ce.is_recursive,
                },
            )

    def _call_source_id(self, ce: CallEdge) -> str | None:
        """Resolve a call's caller to a node id (symbol, or module for scope)."""
        if ce.caller == MODULE_SCOPE:
            module = self._file_to_module.get(ce.caller_file) or _module_name(
                ce.caller_file
            )
            return self._ensure_module_node(module, file_path=ce.caller_file)
        return self._by_file_local.get((ce.caller_file, ce.caller))

    def _build_import_edges(self) -> None:
        """Turn :class:`DependencyEdge` objects into ``IMPORTS`` relationships.

        A resolved import's target module is keyed by the *resolved file's* full
        dotted name (``_module_name(de.target)``), not the short name written at
        the import site (``de.target_module`` is ``"auth"`` for
        ``from auth import x``). This makes the ``IMPORTS`` target coincide with
        the very ``Module`` node that ``CONTAINS`` that file's symbols, keeping
        the call, dependency, and symbol facets one connected graph.
        """
        for de in self._deps:
            source_module = _module_name(de.source)
            source_id = self._ensure_module_node(
                source_module, file_path=de.source, external=False
            )
            target_module = _module_name(de.target) if de.resolved else de.target_module
            target_id = self._ensure_module_node(
                target_module,
                file_path=de.target if de.resolved else None,
                external=not de.resolved,
            )
            if source_id is None or target_id is None:
                continue
            self._add_edge(
                EDGE_IMPORTS,
                source_id,
                target_id,
                {
                    "import_type": de.import_type,
                    "line": de.line,
                    "is_relative": de.is_relative,
                    "relative_level": de.relative_level,
                    "is_wildcard": de.is_wildcard,
                    "resolved": de.resolved,
                    "imported_names": _serialise_imported_names(de.imported_names),
                },
            )

    def _build_inherits_edges(self) -> None:
        """Link classes to resolvable base classes via ``INHERITS`` edges."""
        for rec in self._records:
            if rec.type != "class":
                continue
            for base in rec.bases:
                if "." in base or "[" in base:
                    continue  # dotted / generic bases cannot be resolved soundly
                target_id = self._resolve_base_class(rec.module, base)
                if target_id is not None and target_id != rec.symbol_id:
                    self._add_edge(EDGE_INHERITS, rec.symbol_id, target_id, {})

    def _resolve_base_class(self, module: str, base_name: str) -> str | None:
        """Resolve a bare base-class name to a class node id, soundly.

        Prefers a class of that name in the *same module*; otherwise accepts a
        globally unique class by that name. An ambiguous or unknown base yields
        ``None`` (no edge) rather than a guess.
        """
        same_module = self._classes_by_module.get((module, base_name))
        if same_module is not None:
            return same_module
        candidates = self._classes_by_name.get(base_name, [])
        if len(candidates) == 1:
            return candidates[0]
        return None


# ---------------------------------------------------------------------------
# Base store interface
# ---------------------------------------------------------------------------


class BaseGraphStore(ABC):
    """Backend-agnostic interface for persisting and querying the code graph.

    Concrete backends (:class:`Neo4jGraphStore`, :class:`NetworkXGraphStore`)
    implement the abstract primitives; the shared logic here
    (:meth:`persist_graph`, :meth:`subgraph`, context-manager support) is written
    once against that interface.
    """

    #: ``"neo4j"`` or ``"networkx"`` -- which backend is active.
    backend: str

    # -- Bulk persistence -------------------------------------------------

    def persist_graph(
        self,
        call_edges: Iterable[CallEdge] | None = None,
        dep_edges: Iterable[DependencyEdge] | DependencyGraphResult | None = None,
        symbols: SymbolTable | Iterable[SymbolRecord] | None = None,
        *,
        clear: bool = False,
    ) -> PersistSummary:
        """Assemble and persist the graph from the three upstream facets.

        Args:
            call_edges: Call graph edges (Issue 9).
            dep_edges: Dependency edges or a
                :class:`~src.reporag.graph.dependency_graph.DependencyGraphResult`
                (Issue 10).
            symbols: A :class:`~src.reporag.graph.symbol_table.SymbolTable` or an
                iterable of :class:`~src.reporag.graph.symbol_table.SymbolRecord`
                (Issue 11) -- the source of node metadata.
            clear: When ``True``, wipe the store before inserting (idempotent
                re-ingest). ``MERGE`` semantics already make re-persisting safe.

        Returns:
            A :class:`PersistSummary` with the node and edge counts written.
        """
        nodes, edges = _GraphBuilder(symbols, call_edges, dep_edges).build()
        if clear:
            self.clear()
        self.add_nodes(nodes)
        self.add_edges(edges)
        return PersistSummary(nodes=len(nodes), edges=len(edges))

    def subgraph(self, node_ids: Iterable[str], *, depth: int = 0) -> Subgraph:
        """Extract the induced subgraph over *node_ids*, optionally expanded.

        - **Algorithm**: Optionally grows the seed set by *depth* hops in both
          directions (via :meth:`get_neighbors`), then returns every seed/expanded
          node plus the edges whose endpoints are both in that set.
        - **Correctness choice**: Implemented once here in terms of the backend
          primitives, so Neo4j and NetworkX return identical subgraphs.
        """
        ids: set[str] = set(node_ids)
        frontier = set(ids)
        for _ in range(max(0, depth)):
            new_frontier: set[str] = set()
            for nid in frontier:
                for neighbour in self.get_neighbors(nid, depth=1, direction="both"):
                    if neighbour.id not in ids:
                        new_frontier.add(neighbour.id)
            ids |= new_frontier
            frontier = new_frontier
            if not frontier:
                break

        nodes = [n for n in (self.get_node(i) for i in ids) if n is not None]
        nodes.sort(key=lambda n: n.id)
        edges = self._edges_among(ids)
        return Subgraph(nodes=nodes, edges=edges)

    # -- Context-manager sugar -------------------------------------------

    def __enter__(self) -> BaseGraphStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend needs it
        """Release any backend resources (no-op by default)."""

    # -- Abstract primitives ---------------------------------------------

    @abstractmethod
    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        """Insert or update *nodes* (idempotent on ``id``)."""

    @abstractmethod
    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        """Insert or update *edges* (idempotent on ``(type, source, target)``)."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every node and relationship from the store."""

    @abstractmethod
    def query(self, cypher: str, **params: object) -> list[dict]:
        """Run a raw Cypher query, returning a list of row dicts.

        Only the Neo4j backend supports this; the NetworkX fallback raises
        :class:`NotImplementedError` (use the structured helpers instead).
        """

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the node with *node_id*, or ``None`` if absent."""

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        direction: Direction = "both",
        edge_types: Iterable[str] | None = None,
    ) -> list[GraphNode]:
        """Return nodes within *depth* hops of *node_id*.

        Args:
            depth: Maximum number of hops (>= 1).
            direction: ``"out"`` (successors), ``"in"`` (predecessors), or
                ``"both"``.
            edge_types: Restrict traversal to these relationship types; ``None``
                traverses every type.
        """

    @abstractmethod
    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: Iterable[str] | None = None,
        max_depth: int = 15,
    ) -> list[GraphNode] | None:
        """Return the shortest directed path ``source -> target`` as nodes.

        ``None`` when no path exists (or an endpoint is missing). *edge_types*
        restricts which relationship types the path may traverse.
        """

    @abstractmethod
    def node_count(self) -> int:
        """Return the total number of nodes."""

    @abstractmethod
    def edge_count(self) -> int:
        """Return the total number of relationships."""

    @abstractmethod
    def _edges_among(self, node_ids: set[str]) -> list[GraphEdge]:
        """Return every edge whose source and target are both in *node_ids*."""


# ---------------------------------------------------------------------------
# NetworkX fallback backend
# ---------------------------------------------------------------------------


class NetworkXGraphStore(BaseGraphStore):
    """In-memory :class:`BaseGraphStore` backed by ``networkx.MultiDiGraph``.

    Requires no server, so it powers the unit tests and any environment where
    Neo4j is unavailable. It supports every structured operation; only raw
    :meth:`query` (Cypher) is unavailable.
    """

    backend = "networkx"

    def __init__(self) -> None:
        """Create an empty in-memory graph."""
        import networkx as nx

        self._nx = nx
        self._g = nx.MultiDiGraph()
        # Keep the rich GraphNode objects so lookups return full metadata.
        self._node_objs: dict[str, GraphNode] = {}

    # -- Mutation ---------------------------------------------------------

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        """Insert or update nodes in the in-memory graph."""
        for node in nodes:
            self._g.add_node(node.id, label=node.label, **node.properties)
            self._node_objs[node.id] = node

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        """Insert edges keyed by relationship type (so parallel types coexist)."""
        for edge in edges:
            self._g.add_edge(
                edge.source,
                edge.target,
                key=edge.type,
                type=edge.type,
                **edge.properties,
            )

    def clear(self) -> None:
        """Drop all nodes and edges."""
        self._g.clear()
        self._node_objs.clear()

    # -- Read -------------------------------------------------------------

    def query(self, cypher: str, **params: object) -> list[dict]:
        """Unsupported on the NetworkX backend."""
        raise NotImplementedError(
            "Raw Cypher queries require the Neo4j backend. Use get_neighbors(), "
            "shortest_path(), or subgraph() on the NetworkX fallback."
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the stored :class:`GraphNode` for *node_id*."""
        return self._node_objs.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        direction: Direction = "both",
        edge_types: Iterable[str] | None = None,
    ) -> list[GraphNode]:
        """Breadth-first neighbour search up to *depth* hops."""
        if node_id not in self._g:
            return []
        depth = max(1, depth)
        allowed = set(edge_types) if edge_types is not None else None

        visited = {node_id}
        frontier = {node_id}
        found: set[str] = set()
        for _ in range(depth):
            next_frontier: set[str] = set()
            for current in frontier:
                for neighbour in self._adjacent(current, direction, allowed):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.add(neighbour)
                        found.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return [self._node_objs[i] for i in sorted(found) if i in self._node_objs]

    def _adjacent(
        self, node_id: str, direction: Direction, allowed: set[str] | None
    ) -> Iterator[str]:
        """Yield neighbour ids of *node_id* filtered by direction and edge type."""
        if direction in ("out", "both"):
            for _, target, key in self._g.out_edges(node_id, keys=True):
                if allowed is None or key in allowed:
                    yield target
        if direction in ("in", "both"):
            for source, _, key in self._g.in_edges(node_id, keys=True):
                if allowed is None or key in allowed:
                    yield source

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: Iterable[str] | None = None,
        max_depth: int = 15,
    ) -> list[GraphNode] | None:
        """Directed shortest path via ``networkx.shortest_path``."""
        if source_id not in self._g or target_id not in self._g:
            return None
        graph = self._filtered_digraph(edge_types)
        try:
            path_ids = self._nx.shortest_path(graph, source_id, target_id)
        except (self._nx.NetworkXNoPath, self._nx.NodeNotFound):
            return None
        # Honour the hop cap so both backends agree (Neo4j uses ``*..max_depth``).
        if len(path_ids) - 1 > max(1, max_depth):
            return None
        return [self._node_objs[i] for i in path_ids if i in self._node_objs]

    def _filtered_digraph(self, edge_types: Iterable[str] | None):
        """Return the graph to route over, filtered to *edge_types* if given."""
        if edge_types is None:
            return self._g
        allowed = set(edge_types)
        filtered = self._nx.DiGraph()
        filtered.add_nodes_from(self._g.nodes())
        for source, target, key in self._g.edges(keys=True):
            if key in allowed:
                filtered.add_edge(source, target)
        return filtered

    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        """Number of relationships in the graph."""
        return self._g.number_of_edges()

    def _edges_among(self, node_ids: set[str]) -> list[GraphEdge]:
        """Return edges whose endpoints are both within *node_ids*."""
        out: list[GraphEdge] = []
        for source, target, key, data in self._g.edges(keys=True, data=True):
            if source in node_ids and target in node_ids:
                props = {k: v for k, v in data.items() if k != "type"}
                out.append(
                    GraphEdge(type=key, source=source, target=target, properties=props)
                )
        return out


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


class Neo4jGraphStore(BaseGraphStore):
    """A :class:`BaseGraphStore` backed by a live Neo4j database.

    - **Bulk insert**: nodes and edges are written with batched ``UNWIND``
      transactions (``batch_size`` rows each), so 10k+ nodes load in a handful of
      round-trips rather than one query per node.
    - **Indexing**: a uniqueness constraint on ``:CodeNode(id)`` (created on
      connect) makes ``MERGE`` and every ``MATCH ... {id: ...}`` index-backed.
    - **Resilience**: connection and every read/write are wrapped in
      :meth:`_with_retry` (exponential backoff) on top of the driver's own
      managed-transaction retries, so transient blips are ridden out.

    Prefer the :func:`GraphStore` factory, which can fall back to
    :class:`NetworkXGraphStore` when Neo4j is unreachable.
    """

    backend = "neo4j"

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
        batch_size: int = 1000,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        """Open a driver, verify connectivity (with retry), and ensure the schema.

        Raises:
            neo4j.exceptions.Neo4jError: If the database cannot be reached after
                ``max_retries`` attempts, or authentication fails.
        """
        from neo4j import GraphDatabase

        self._database = database
        self._batch_size = max(1, batch_size)
        self._max_retries = max(1, max_retries)
        self._retry_backoff = retry_backoff

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            self._with_retry(self._driver.verify_connectivity)
            self._ensure_constraints()
        except Exception:
            # Never leak an unusable driver (avoids the destructor warning and
            # frees the connection pool) before propagating the failure.
            self._driver.close()
            raise

    # -- Retry / execution helpers ---------------------------------------

    def _with_retry(self, func, *args, **kwargs):
        """Call *func*, retrying transient Neo4j errors with exponential backoff."""
        from neo4j.exceptions import (
            ServiceUnavailable,
            SessionExpired,
            TransientError,
        )

        delay = self._retry_backoff
        for attempt in range(1, self._max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (ServiceUnavailable, SessionExpired, TransientError) as exc:
                if attempt >= self._max_retries:
                    raise
                logger.warning(
                    "Neo4j operation failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

    def _read(self, cypher: str, transform, **params):
        """Run a read query, applying *transform* to the records inside the tx."""

        def work(tx):
            return transform(list(tx.run(cypher, **params)))

        with self._driver.session(database=self._database) as session:
            return self._with_retry(session.execute_read, work)

    def _write(self, cypher: str, transform=None, **params):
        """Run a write query, applying *transform* to the records inside the tx."""

        def work(tx):
            result = list(tx.run(cypher, **params))
            return transform(result) if transform is not None else None

        with self._driver.session(database=self._database) as session:
            return self._with_retry(session.execute_write, work)

    def _ensure_constraints(self) -> None:
        """Create the uniqueness constraint that indexes every node by id."""
        self._write(
            f"CREATE CONSTRAINT code_node_id IF NOT EXISTS "
            f"FOR (n:{SHARED_LABEL}) REQUIRE n.id IS UNIQUE"
        )

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    # -- Mutation ---------------------------------------------------------

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        """Bulk-insert nodes with one batched ``UNWIND MERGE`` per label."""
        by_label: dict[str, list[dict]] = {}
        for node in nodes:
            row = {"id": node.id, "props": {**node.properties, "id": node.id}}
            by_label.setdefault(node.label, []).append(row)

        for label, rows in by_label.items():
            safe_label = _safe_ident(label)
            cypher = (
                f"UNWIND $rows AS row "
                f"MERGE (n:{SHARED_LABEL} {{id: row.id}}) "
                f"SET n += row.props, n:`{safe_label}`"
            )
            for chunk in _chunks(rows, self._batch_size):
                self._write(cypher, rows=chunk)

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        """Bulk-insert edges with one batched ``UNWIND MATCH MERGE`` per type."""
        by_type: dict[str, list[dict]] = {}
        for edge in edges:
            row = {
                "source": edge.source,
                "target": edge.target,
                "props": dict(edge.properties),
            }
            by_type.setdefault(edge.type, []).append(row)

        for edge_type, rows in by_type.items():
            safe_type = _safe_ident(edge_type)
            cypher = (
                f"UNWIND $rows AS row "
                f"MATCH (a:{SHARED_LABEL} {{id: row.source}}) "
                f"MATCH (b:{SHARED_LABEL} {{id: row.target}}) "
                f"MERGE (a)-[r:`{safe_type}`]->(b) "
                f"SET r += row.props"
            )
            for chunk in _chunks(rows, self._batch_size):
                self._write(cypher, rows=chunk)

    def clear(self) -> None:
        """Delete all nodes/relationships in batches (safe for large graphs)."""
        cypher = (
            f"MATCH (n:{SHARED_LABEL}) WITH n LIMIT $batch "
            f"DETACH DELETE n RETURN count(n) AS deleted"
        )
        while True:
            deleted = self._write(
                cypher,
                transform=lambda recs: recs[0]["deleted"] if recs else 0,
                batch=self._batch_size,
            )
            if not deleted:
                break

    # -- Read -------------------------------------------------------------

    def query(self, cypher: str, **params: object) -> list[dict]:
        """Run an arbitrary Cypher query and return a list of row dicts."""
        return self._read(cypher, lambda recs: [r.data() for r in recs], **params)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the node with *node_id*, or ``None``."""
        return self._read(
            f"MATCH (n:{SHARED_LABEL} {{id: $id}}) RETURN n",
            lambda recs: self._node_to_graphnode(recs[0]["n"]) if recs else None,
            id=node_id,
        )

    def get_neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        direction: Direction = "both",
        edge_types: Iterable[str] | None = None,
    ) -> list[GraphNode]:
        """Variable-length neighbour traversal via Cypher."""
        depth = max(1, depth)
        rel = self._rel_pattern(edge_types)
        left, right = self._arrows(direction)
        cypher = (
            f"MATCH (a:{SHARED_LABEL} {{id: $id}}) "
            f"MATCH (a){left}[r{rel}*1..{depth}]{right}(b:{SHARED_LABEL}) "
            f"WHERE b.id <> $id "
            f"RETURN DISTINCT b"
        )
        return self._read(
            cypher,
            lambda recs: [self._node_to_graphnode(r["b"]) for r in recs],
            id=node_id,
        )

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_types: Iterable[str] | None = None,
        max_depth: int = 15,
    ) -> list[GraphNode] | None:
        """Directed ``shortestPath`` via Cypher."""
        max_depth = max(1, max_depth)
        rel = self._rel_pattern(edge_types)
        cypher = (
            f"MATCH (a:{SHARED_LABEL} {{id: $a}}), (b:{SHARED_LABEL} {{id: $b}}) "
            f"MATCH p = shortestPath((a)-[r{rel}*..{max_depth}]->(b)) "
            f"RETURN p"
        )
        return self._read(
            cypher,
            lambda recs: (
                [self._node_to_graphnode(n) for n in recs[0]["p"].nodes]
                if recs
                else None
            ),
            a=source_id,
            b=target_id,
        )

    def node_count(self) -> int:
        """Count all nodes."""
        return self._read(
            f"MATCH (n:{SHARED_LABEL}) RETURN count(n) AS c",
            lambda recs: recs[0]["c"] if recs else 0,
        )

    def edge_count(self) -> int:
        """Count all relationships."""
        return self._read(
            f"MATCH (:{SHARED_LABEL})-[r]->(:{SHARED_LABEL}) RETURN count(r) AS c",
            lambda recs: recs[0]["c"] if recs else 0,
        )

    def _edges_among(self, node_ids: set[str]) -> list[GraphEdge]:
        """Return edges whose endpoints are both within *node_ids*."""
        cypher = (
            f"MATCH (a:{SHARED_LABEL})-[r]->(b:{SHARED_LABEL}) "
            f"WHERE a.id IN $ids AND b.id IN $ids "
            f"RETURN a.id AS source, type(r) AS type, b.id AS target, "
            f"properties(r) AS props"
        )
        return self._read(
            cypher,
            lambda recs: [
                GraphEdge(
                    type=r["type"],
                    source=r["source"],
                    target=r["target"],
                    properties=dict(r["props"]),
                )
                for r in recs
            ],
            ids=list(node_ids),
        )

    # -- Cypher construction helpers -------------------------------------

    @staticmethod
    def _node_to_graphnode(node) -> GraphNode:
        """Convert a ``neo4j.graph.Node`` into a :class:`GraphNode`."""
        props = dict(node)
        specific = [label for label in node.labels if label != SHARED_LABEL]
        label = specific[0] if specific else SHARED_LABEL
        return GraphNode(id=props.get("id", ""), label=label, properties=props)

    @staticmethod
    def _rel_pattern(edge_types: Iterable[str] | None) -> str:
        """Return a relationship filter like ``:`CALLS`|`IMPORTS```, or ``""``."""
        if edge_types is None:
            return ""
        parts = [f"`{_safe_ident(t)}`" for t in edge_types]
        return f":{'|'.join(parts)}" if parts else ""

    @staticmethod
    def _arrows(direction: Direction) -> tuple[str, str]:
        """Return the ``(left, right)`` arrow tokens for a traversal direction."""
        return {
            "out": ("-", "->"),
            "in": ("<-", "-"),
            "both": ("-", "-"),
        }[direction]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def GraphStore(  # noqa: N802 - factory intentionally named like a class
    uri: str | None = None,
    *,
    user: str | None = None,
    password: str | None = None,
    backend: Literal["auto", "neo4j", "networkx"] = "auto",
    database: str | None = None,
    batch_size: int = 1000,
    max_retries: int = 3,
    retry_backoff: float = 0.5,
    fallback_to_networkx: bool = True,
) -> BaseGraphStore:
    """Construct a graph store, choosing (or falling back to) a backend.

    - ``backend="networkx"`` -- always the in-memory fallback (no server).
    - ``backend="neo4j"`` -- always Neo4j; connection details default to
      ``settings`` when omitted; raises if the database is unreachable.
    - ``backend="auto"`` (default) -- Neo4j when a *uri* is given (falling back
      to NetworkX on connection failure if *fallback_to_networkx*), else the
      in-memory fallback.

    Args:
        uri: Bolt URI (e.g. ``"bolt://localhost:7687"``). ``None`` selects the
            in-memory backend under ``auto``.
        user: Username; defaults to ``settings.neo4j_user``.
        password: Password; defaults to ``settings.neo4j_password``.
        backend: Backend selection strategy (see above).
        database: Optional Neo4j database name.
        batch_size: Rows per bulk-insert transaction.
        max_retries: Connection/operation retry attempts.
        retry_backoff: Initial backoff (seconds), doubled each retry.
        fallback_to_networkx: Under ``auto``, degrade to NetworkX instead of
            raising when Neo4j cannot be reached.

    Returns:
        A ready-to-use :class:`BaseGraphStore`.
    """
    selected = backend.lower()
    if selected == "networkx":
        return NetworkXGraphStore()
    if selected not in ("auto", "neo4j"):
        raise ValueError(f"Unknown backend: {backend!r}")
    if selected == "auto" and uri is None:
        return NetworkXGraphStore()

    resolved_uri, resolved_user, resolved_password = _resolve_credentials(
        uri, user, password
    )
    try:
        return Neo4jGraphStore(
            resolved_uri,
            resolved_user,
            resolved_password,
            database=database,
            batch_size=batch_size,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
    except Exception as exc:  # noqa: BLE001 - fall back on any connection failure
        if selected == "neo4j" or not fallback_to_networkx:
            raise
        logger.warning(
            "Neo4j unavailable (%s: %s); falling back to in-memory NetworkX store.",
            type(exc).__name__,
            exc,
        )
        return NetworkXGraphStore()


def _resolve_credentials(
    uri: str | None, user: str | None, password: str | None
) -> tuple[str, str, str]:
    """Fill missing Neo4j connection details from ``settings``."""
    from src.reporag.config import settings

    resolved_uri = uri if uri is not None else settings.neo4j_uri
    resolved_user = user if user is not None else settings.neo4j_user
    if password is not None:
        resolved_password = password
    else:
        secret = settings.neo4j_password
        resolved_password = (
            secret.get_secret_value()
            if hasattr(secret, "get_secret_value")
            else str(secret)
        )
    return resolved_uri, resolved_user, resolved_password

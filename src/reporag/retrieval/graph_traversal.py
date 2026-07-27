"""Graph-based retrieval over the code knowledge graph.

This module is the retrieval-facing adapter for Issue 18.  The graph store
already owns persistence and low-level traversal; :class:`GraphRetriever` adds
symbol-name resolution, retrieval-style ranking, path expansion, and conversion
to the shared :class:`reporag.retrieval.vector_search.RetrievalResult` schema.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from reporag.config import settings
from reporag.graph.neo4j_store import (
    GraphStore,
    GraphStoreProtocol,
    NetworkXGraphStore,
)
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

GraphDirection = Literal["out", "in", "both"]

CALL_EDGE = "CALLS"
IMPORT_EDGE = "IMPORTS"
INHERIT_EDGE = "INHERITS"
CONTAIN_EDGE = "CONTAINS"
DEFAULT_EDGE_TYPES = (CALL_EDGE, IMPORT_EDGE, INHERIT_EDGE, CONTAIN_EDGE)
_RELATIONSHIP_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class GraphRetrievalResult(RetrievalResult):
    """A graph hit represented as a normal retrieval result plus traversal data."""

    depth: int = 0
    source_symbol: str | None = None
    relationship: str | None = None
    direction: GraphDirection = "both"


@dataclass
class GraphPath:
    """A symbol path returned by :meth:`GraphRetriever.find_paths`."""

    symbols: list[str]
    nodes: list[dict[str, Any]]
    results: list[GraphRetrievalResult]
    score: float
    edges: list[dict[str, Any]] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Number of graph hops in the path."""
        return max(0, len(self.symbols) - 1)


@dataclass
class GraphSubgraph:
    """An induced subgraph plus retrieval-result views of its nodes."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    results: list[GraphRetrievalResult]


class GraphRetriever:
    """Retrieve structural context from Neo4j or the NetworkX fallback.

    Args:
        graph_store: Optional pre-built graph store. Tests and in-memory
            pipelines should pass a :class:`NetworkXGraphStore`; production can
            omit this and pass ``neo4j_uri`` instead.
        neo4j_uri: Bolt URI for Neo4j. When omitted, ``settings.neo4j_uri`` is
            used unless ``use_settings_uri`` is ``False``.
        username: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.
        fallback: When ``True``, the graph-store factory falls back to
            NetworkX if Neo4j is unavailable.
        use_settings_uri: When ``False`` and no ``neo4j_uri`` is provided,
            construct an empty NetworkX store. This is convenient for tests.
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol | None = None,
        *,
        neo4j_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        fallback: bool = True,
        use_settings_uri: bool = True,
    ) -> None:
        self.store = graph_store or GraphStore(
            uri=(
                neo4j_uri
                if neo4j_uri is not None
                else (settings.neo4j_uri if use_settings_uri else None)
            ),
            username=username or settings.neo4j_user,
            password=password or settings.neo4j_password.get_secret_value(),
            database=database,
            fallback=fallback,
        )

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        symbol: str,
        *,
        depth: int = 1,
        direction: GraphDirection = "both",
        edge_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[GraphRetrievalResult]:
        """Return N-hop graph neighbors as common retrieval results.

        ``symbol`` may be a ``symbol_id``, fully qualified name, bare symbol
        name, or module node id such as ``"module:pkg.service"`` when the
        backing store can resolve bare names locally.
        """
        self._validate_depth(depth)
        self._validate_direction(direction)
        normalized_edge_types = self._normalize_edge_types(edge_types)

        start_id = self._resolve_symbol_id(symbol)
        if start_id is None:
            return []

        nodes = self.store.get_neighbors(
            start_id,
            edge_types=normalized_edge_types,
            depth=depth,
            direction=direction,
        )
        results = [
            self._node_to_result(
                node,
                depth=self._distance(
                    start_id,
                    self._node_id(node),
                    direction,
                    edge_types=normalized_edge_types,
                ),
                source_symbol=start_id,
                direction=direction,
                relationship=self._relationship_between(start_id, self._node_id(node)),
            )
            for node in nodes
        ]
        results.sort(
            key=lambda result: (
                -result.score,
                result.depth,
                result.symbol_name or "",
                result.file_path,
                result.start_line or 0,
            )
        )
        return results[:top_k] if top_k is not None else results

    def get_callers(
        self,
        symbol: str,
        *,
        depth: int = 1,
        top_k: int | None = None,
    ) -> list[GraphRetrievalResult]:
        """Return functions that call ``symbol`` within ``depth`` hops."""
        return self.get_neighbors(
            symbol,
            depth=depth,
            direction="in",
            edge_types=[CALL_EDGE],
            top_k=top_k,
        )

    def get_callees(
        self,
        symbol: str,
        *,
        depth: int = 1,
        top_k: int | None = None,
    ) -> list[GraphRetrievalResult]:
        """Return functions called by ``symbol`` within ``depth`` hops."""
        return self.get_neighbors(
            symbol,
            depth=depth,
            direction="out",
            edge_types=[CALL_EDGE],
            top_k=top_k,
        )

    def find_shortest_path(
        self,
        source_symbol: str,
        target_symbol: str,
        *,
        max_depth: int = 5,
        direction: GraphDirection = "both",
        edge_types: list[str] | None = None,
    ) -> GraphPath | None:
        """Return the shortest path between two symbols, if one exists."""
        paths = self.find_paths(
            source_symbol,
            target_symbol,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
            limit=1,
        )
        return paths[0] if paths else None

    def find_paths(
        self,
        source_symbol: str,
        target_symbol: str,
        *,
        max_depth: int = 5,
        direction: GraphDirection = "both",
        edge_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[GraphPath]:
        """Return simple paths between two symbols, shortest first.

        For the NetworkX backend this uses ``networkx.all_simple_paths``.
        For Neo4j it issues a bounded Cypher path query.  If a custom store
        cannot enumerate paths, the method falls back to its shortest-path
        helper and returns either zero or one path.
        """
        self._validate_depth(max_depth)
        self._validate_direction(direction)
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")

        edge_types = self._normalize_edge_types(edge_types)
        source_id = self._resolve_symbol_id(source_symbol)
        target_id = self._resolve_symbol_id(target_symbol)
        if source_id is None or target_id is None:
            return []

        if isinstance(self.store, NetworkXGraphStore):
            node_paths = self._networkx_paths(
                source_id,
                target_id,
                max_depth=max_depth,
                direction=direction,
                edge_types=edge_types,
                limit=limit,
            )
        else:
            node_paths = self._cypher_paths(
                source_id,
                target_id,
                max_depth=max_depth,
                direction=direction,
                edge_types=edge_types,
                limit=limit,
            )
            if node_paths is None:
                shortest = self.store.shortest_path(
                    source_id, target_id, edge_types=edge_types
                )
                node_paths = [shortest] if 1 < len(shortest) <= max_depth + 1 else []

        return [
            self._nodes_to_path(path_nodes, edge_types=edge_types)
            for path_nodes in node_paths
            if path_nodes
        ]

    def extract_subgraph(
        self,
        symbols: list[str],
        *,
        include_neighbors: bool = True,
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> GraphSubgraph:
        """Return the induced subgraph around the requested symbols.

        When ``include_neighbors`` is true, the induced node set includes each
        seed plus its N-hop neighbors.  Set it to ``False`` for a strict induced
        subgraph over exactly the resolved seed symbols.
        """
        self._validate_depth(depth)
        edge_types = self._normalize_edge_types(edge_types)

        node_ids: list[str] = []
        for symbol in symbols:
            symbol_id = self._resolve_symbol_id(symbol)
            if symbol_id is not None:
                node_ids.append(symbol_id)
                if include_neighbors:
                    for result in self.get_neighbors(
                        symbol_id,
                        depth=depth,
                        direction="both",
                        edge_types=edge_types,
                    ):
                        neighbor_id = result.metadata.get("symbol_id")
                        if isinstance(neighbor_id, str):
                            node_ids.append(neighbor_id)

        unique_ids = list(dict.fromkeys(node_ids))
        nodes, edges = self.store.subgraph(unique_ids)
        if edge_types is not None:
            allowed = set(edge_types)
            edges = [edge for edge in edges if edge.get("type") in allowed]

        results = [
            self._node_to_result(node, depth=0, relationship=None) for node in nodes
        ]
        results.sort(key=lambda result: (result.file_path, result.start_line or 0))
        return GraphSubgraph(nodes=nodes, edges=edges, results=results)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search graph neighbors for a symbol-like query.

        This mirrors the vector/BM25 ``search`` surface for fusion callers.
        It treats ``query`` as a symbol identifier/name, retrieves its local
        graph neighborhood, and returns common ``RetrievalResult`` objects.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")
        return self.get_neighbors(
            query,
            depth=depth,
            direction="both",
            edge_types=edge_types,
            top_k=top_k,
        )

    def close(self) -> None:
        """Close the underlying graph store."""
        self.store.close()

    # ------------------------------------------------------------------
    # Path backends
    # ------------------------------------------------------------------

    def _networkx_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int,
        direction: GraphDirection,
        edge_types: list[str] | None,
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        import networkx as nx

        graph = self.store.graph
        if source_id not in graph or target_id not in graph:
            return []

        if edge_types is not None:
            allowed = set(edge_types)
            filtered = nx.DiGraph()
            filtered.add_nodes_from(graph.nodes(data=True))
            filtered.add_edges_from(
                (u, v, data)
                for u, v, data in graph.edges(data=True)
                if data.get("type") in allowed
            )
            graph = filtered

        if direction == "in":
            traversal_graph = graph.reverse(copy=False)
        elif direction == "both":
            traversal_graph = graph.to_undirected(as_view=True)
        else:
            traversal_graph = graph

        try:
            paths_iter = nx.all_simple_paths(
                traversal_graph,
                source=source_id,
                target=target_id,
                cutoff=max_depth,
            )
            path_ids = sorted(paths_iter, key=lambda path: (len(path), path))[:limit]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return [
            [dict(self.store.graph.nodes[node_id]) for node_id in path]
            for path in path_ids
        ]

    def _cypher_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int,
        direction: GraphDirection,
        edge_types: list[str] | None,
        limit: int,
    ) -> list[list[dict[str, Any]]] | None:
        rel = self._cypher_relationship(edge_types, max_depth)
        if direction == "out":
            pattern = f"(a)-{rel}->(b)"
        elif direction == "in":
            pattern = f"(a)<-{rel}-(b)"
        else:
            pattern = f"(a)-{rel}-(b)"

        cypher = f"""
            MATCH (a {{symbol_id: $src}}), (b {{symbol_id: $tgt}})
            MATCH p = {pattern}
            WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n))
            RETURN [n IN nodes(p) | properties(n)] AS path_nodes
            ORDER BY length(p) ASC
            LIMIT $limit
        """
        try:
            result = self.store.query(
                cypher,
                {"src": source_id, "tgt": target_id, "limit": limit},
            )
        except NotImplementedError:
            return None
        return [row["path_nodes"] for row in result.records if row.get("path_nodes")]

    # ------------------------------------------------------------------
    # Conversion and scoring
    # ------------------------------------------------------------------

    def _nodes_to_path(
        self,
        nodes: list[dict[str, Any]],
        *,
        edge_types: list[str] | None,
    ) -> GraphPath:
        results = [
            self._node_to_result(node, depth=index, relationship=None)
            for index, node in enumerate(nodes)
        ]
        symbols = [self._display_symbol(node) for node in nodes]
        edges = self._path_edges(nodes, edge_types=edge_types)
        score = 1.0 / max(1, len(nodes) - 1)
        return GraphPath(
            symbols=symbols, nodes=nodes, results=results, score=score, edges=edges
        )

    def _node_to_result(
        self,
        node: dict[str, Any],
        *,
        depth: int,
        relationship: str | None,
        source_symbol: str | None = None,
        direction: GraphDirection = "both",
    ) -> GraphRetrievalResult:
        score = 1.0 / (depth + 1)
        symbol_name = self._display_symbol(node)
        node_type = node.get("type") or node.get("label")
        signature = node.get("signature") or ""
        docstring = node.get("docstring") or ""
        chunk_parts = [
            part
            for part in (
                f"{node_type} {symbol_name}".strip(),
                signature,
                docstring,
            )
            if part
        ]
        return GraphRetrievalResult(
            score=score,
            file_path=str(node.get("file_path") or ""),
            start_line=self._optional_int(node.get("start_line")),
            end_line=self._optional_int(node.get("end_line")),
            symbol_name=symbol_name,
            chunk_text="\n".join(chunk_parts),
            metadata=dict(node),
            depth=depth,
            source_symbol=source_symbol,
            relationship=relationship,
            direction=direction,
        )

    @staticmethod
    def _display_symbol(node: dict[str, Any]) -> str:
        return str(
            node.get("qualified_name")
            or node.get("symbol_id")
            or node.get("name")
            or ""
        )

    @staticmethod
    def _node_id(node: dict[str, Any]) -> str:
        return str(node.get("symbol_id") or node.get("qualified_name") or "")

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    # ------------------------------------------------------------------
    # Symbol resolution and graph inspection
    # ------------------------------------------------------------------

    def _resolve_symbol_id(self, symbol: str) -> str | None:
        symbol = symbol.strip()
        if not symbol:
            return None

        if isinstance(self.store, NetworkXGraphStore):
            graph = self.store.graph
            if symbol in graph:
                return symbol
            matches = [
                node_id
                for node_id, data in graph.nodes(data=True)
                if data.get("qualified_name") == symbol
                or data.get("name") == symbol
                or data.get("symbol_id") == symbol
            ]
            if len(matches) > 1:
                logger.warning(
                    "GraphRetriever: symbol %r is ambiguous; using %r from %r",
                    symbol,
                    matches[0],
                    matches,
                )
            return matches[0] if matches else None

        try:
            rows = self.store.query(
                """
                MATCH (n)
                WHERE n.symbol_id = $symbol
                   OR n.qualified_name = $symbol
                   OR n.name = $symbol
                RETURN n.symbol_id AS symbol_id
                ORDER BY n.qualified_name ASC, n.symbol_id ASC
                LIMIT 2
                """,
                {"symbol": symbol},
            ).records
        except NotImplementedError:
            return symbol
        if len(rows) > 1:
            logger.warning("GraphRetriever: symbol %r is ambiguous in Neo4j", symbol)
        return str(rows[0]["symbol_id"]) if rows else None

    def _distance(
        self,
        source_id: str,
        target_id: str,
        direction: GraphDirection,
        *,
        edge_types: list[str] | None,
    ) -> int:
        if not target_id or source_id == target_id:
            return 0
        if not isinstance(self.store, NetworkXGraphStore):
            return 1

        import networkx as nx

        graph = self.store.graph
        if edge_types is not None:
            allowed = set(edge_types)
            filtered = nx.DiGraph()
            filtered.add_nodes_from(graph.nodes(data=True))
            filtered.add_edges_from(
                (u, v, data)
                for u, v, data in graph.edges(data=True)
                if data.get("type") in allowed
            )
            graph = filtered
        if direction == "in":
            traversal_graph = graph.reverse(copy=False)
        elif direction == "both":
            traversal_graph = graph.to_undirected(as_view=True)
        else:
            traversal_graph = graph
        try:
            return int(nx.shortest_path_length(traversal_graph, source_id, target_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1

    def _relationship_between(
        self,
        source_id: str,
        target_id: str,
    ) -> str | None:
        if not target_id or not isinstance(self.store, NetworkXGraphStore):
            return None
        graph = self.store.graph
        if graph.has_edge(source_id, target_id):
            return str(graph.edges[source_id, target_id].get("type") or "")
        if graph.has_edge(target_id, source_id):
            return str(graph.edges[target_id, source_id].get("type") or "")
        return None

    def _path_edges(
        self,
        nodes: list[dict[str, Any]],
        *,
        edge_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(self.store, NetworkXGraphStore):
            return []

        graph = self.store.graph
        allowed = set(edge_types) if edge_types is not None else None
        edges: list[dict[str, Any]] = []
        node_ids = [self._node_id(node) for node in nodes]
        for source, target in zip(node_ids, node_ids[1:], strict=False):
            edge_data = None
            edge_source = source
            edge_target = target
            if graph.has_edge(source, target):
                edge_data = graph.edges[source, target]
            elif graph.has_edge(target, source):
                edge_data = graph.edges[target, source]
                edge_source = target
                edge_target = source

            if edge_data is None:
                continue
            if allowed is not None and edge_data.get("type") not in allowed:
                continue
            edge = {"source": edge_source, "target": edge_target}
            edge.update(edge_data)
            edges.append(edge)
        return edges

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_depth(depth: int) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

    @staticmethod
    def _validate_direction(direction: str) -> None:
        if direction not in {"out", "in", "both"}:
            raise ValueError(
                "direction must be one of 'out', 'in', or 'both', " f"got {direction!r}"
            )

    @staticmethod
    def _normalize_edge_types(edge_types: list[str] | None) -> list[str] | None:
        if edge_types is None:
            return None
        normalized = [edge_type.upper() for edge_type in edge_types]
        invalid = [
            edge_type
            for edge_type in normalized
            if not _RELATIONSHIP_RE.fullmatch(edge_type)
        ]
        if invalid:
            raise ValueError(f"Invalid graph edge type(s): {invalid!r}")
        return normalized

    @staticmethod
    def _cypher_relationship(edge_types: list[str] | None, max_depth: int) -> str:
        if edge_types:
            joined = "|".join(edge_types)
            return f"[:{joined}*1..{max_depth}]"
        return f"[*1..{max_depth}]"

"""Graph-based retrieval.

Uses the code knowledge graph for structural queries: N-hop neighbors,
shortest paths between symbols, and subgraph extraction. Converts graph
results to the common RetrievalResult schema.
"""

# TODO: Implement in Issue 18
# - get_neighbors(symbol, depth=N): return N-hop callers/callees
# - find_paths(from_symbol, to_symbol, max_depth): shortest + all paths
# - extract_subgraph(symbol_set): induced subgraph with all connecting edges
# - Convert graph results to RetrievalResult schema
# - NetworkX fallback if Neo4j unavailable

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reporag.graph.neo4j_store import GraphStore, GraphStoreProtocol
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Valid directions for traversal -- validated once at call boundaries.
_VALID_DIRECTIONS: frozenset[str] = frozenset({"in", "out", "both"})

# Hard cap on depth to prevent accidental full-graph scans.
_MAX_ALLOWED_DEPTH: int = 10

# Hard cap on max_depth for path queries.
_MAX_ALLOWED_PATH_DEPTH: int = 15


@dataclass
class GraphRetrievalResult(RetrievalResult):
    """A retrieval result representing a single neighbor/node in the code graph.

    Extends RetrievalResult to include the depth (distance) at which the node was found.
    """

    depth: int = 1


@dataclass
class GraphPathResult(RetrievalResult):
    """A retrieval result representing a path connecting two symbols in the graph.

    Extends RetrievalResult to include the sequence of symbols in the path.
    """

    symbols: list[str] = field(default_factory=list)


def _read_file_lines(
    file_path: str, start_line: int | None, end_line: int | None
) -> str:
    """Read a bounded range of lines from a file.

    Uses ``itertools.islice`` so only lines up to ``end_line`` are read --
    never loads the entire file into memory.

    Returns empty string if the lines cannot be read or the file is absent.
    """
    if not file_path or start_line is None or end_line is None:
        return ""
    if start_line < 1 or end_line < start_line:
        return ""
    try:
        path = Path(file_path)
        if not path.is_file():
            return ""
        with open(path, encoding="utf-8", errors="ignore") as f:
            # Skip (start_line - 1) lines, then read (end_line - start_line + 1) lines.
            lines = list(itertools.islice(f, start_line - 1, end_line))
        return "".join(lines)
    except Exception as exc:
        logger.debug("Failed to read file lines from %s: %s", file_path, exc)
        return ""


def _build_chunk_text(props: dict[str, Any]) -> str:
    """Extract chunk text from node properties.

    Tries file content first; falls back to signature + docstring.
    """
    file_path = props.get("file_path", "")
    start_line = props.get("start_line")
    end_line = props.get("end_line")
    text = _read_file_lines(file_path, start_line, end_line)
    if not text:
        sig = props.get("signature") or ""
        doc = props.get("docstring") or ""
        text = f"{sig}\n{doc}".strip() if sig or doc else ""
    return text


def _symbol_name_from_props(props: dict[str, Any]) -> str | None:
    """Extract the best human-readable symbol name from node properties."""
    return props.get("qualified_name") or props.get("name") or props.get("symbol_id")


class GraphRetriever:
    """Performs graph-based retrieval over the code knowledge graph.

    Supports querying structural relationships such as call graph neighbors,
    import dependencies, shortest paths, and induced subgraphs.

    The backend (Neo4j or NetworkX) is detected once at construction time and
    cached, so individual query methods have zero per-call overhead for backend
    detection.

    Args:
        neo4j_uri: Bolt URI for Neo4j. If ``None``, always falls back to NetworkX.
        username: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.
        fallback: If ``True``, falls back to NetworkX if Neo4j connection fails.
        store: Pre-instantiated store (useful for testing).
    """

    def __init__(
        self,
        neo4j_uri: str | None = None,
        username: str = "neo4j",
        password: str = "reporag123",
        database: str = "neo4j",
        fallback: bool = True,
        store: GraphStoreProtocol | None = None,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            self.store = GraphStore(
                uri=neo4j_uri,
                username=username,
                password=password,
                database=database,
                fallback=fallback,
            )

        # Detect backend once so query methods pay no per-call import cost.
        try:
            from reporag.graph.neo4j_store import Neo4jGraphStore

            self._is_neo4j: bool = isinstance(self.store, Neo4jGraphStore)
        except ImportError:
            self._is_neo4j = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_depth(self, depth: int, param: str = "depth") -> None:
        """Raise ValueError for invalid depth values."""
        if depth < 1:
            raise ValueError(f"{param} must be >= 1, got {depth!r}")
        if depth > _MAX_ALLOWED_DEPTH:
            raise ValueError(
                f"{param} must be <= {_MAX_ALLOWED_DEPTH} to avoid full-graph scans, "
                f"got {depth!r}"
            )

    def _validate_direction(self, direction: str) -> None:
        """Raise ValueError for unrecognised direction strings."""
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)!r}, "
                f"got {direction!r}"
            )

    def _validate_edge_types(self, edge_types: list[str] | None) -> None:
        """Raise ValueError if any edge type label is empty or None."""
        if edge_types is None:
            return
        if not edge_types:
            raise ValueError("edge_types must be None or a non-empty list")
        for label in edge_types:
            if not label or not isinstance(label, str):
                raise ValueError(
                    f"Every edge type must be a non-empty string, got {label!r}"
                )

    def _nx_graph(self) -> Any:
        """Return the underlying NetworkX DiGraph, or raise RuntimeError."""
        if not hasattr(self.store, "_g"):
            raise RuntimeError(
                "The graph store does not expose a NetworkX graph via ._g. "
                "This is an internal error -- please report it."
            )
        return self.store._g

    def _resolve_symbol_ids(self, symbol: str) -> list[str]:
        """Resolve a symbol name or ID to a list of matching symbol_ids in the graph."""
        if not symbol:
            return []

        if self._is_neo4j:
            cypher = """
                MATCH (n:_Node)
                WHERE n.symbol_id = $symbol OR n.qualified_name = $symbol OR n.name = $symbol
                RETURN DISTINCT n.symbol_id AS symbol_id
            """
            result = self.store.query(cypher, {"symbol": symbol})
            return [row["symbol_id"] for row in result.records if "symbol_id" in row]

        # NetworkX path: direct-ID check first (O(1)), then property scan (O(N))
        g = self._nx_graph()
        ids: list[str] = []
        if symbol in g:
            ids.append(symbol)
        for node_id, data in g.nodes(data=True):
            if (
                data.get("qualified_name") == symbol or data.get("name") == symbol
            ) and node_id not in ids:
                ids.append(node_id)
        return ids

    def _get_neighbors_networkx(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None,
        depth: int,
        direction: str,
    ) -> list[tuple[dict[str, Any], int]]:
        """BFS over NetworkX DiGraph with per-hop depth tracking."""
        g = self._nx_graph()
        if node_id not in g:
            return []

        if direction == "out":
            graph = g
        elif direction == "in":
            graph = g.reverse(copy=False)
        else:
            graph = g.to_undirected(as_view=True)

        # Pre-compute edge type set for O(1) membership tests per hop.
        allowed_types: frozenset[str] | None = (
            frozenset(edge_types) if edge_types is not None else None
        )

        visited: dict[str, int] = {node_id: 0}
        queue: list[str] = [node_id]

        for d in range(1, depth + 1):
            next_queue: list[str] = []
            for u in queue:
                for v in graph.neighbors(u):
                    if v in visited:
                        continue
                    if allowed_types is not None:
                        # Look up the original directed edge for type checking.
                        if direction == "in":
                            edge_data = g.edges.get((v, u)) or {}
                        elif direction == "out":
                            edge_data = g.edges.get((u, v)) or {}
                        else:
                            edge_data = g.edges.get((u, v)) or g.edges.get((v, u)) or {}
                        if edge_data.get("type") not in allowed_types:
                            continue
                    visited[v] = d
                    next_queue.append(v)
            queue = next_queue
            if not queue:
                break

        visited.pop(node_id)
        return [(dict(g.nodes[nid]), d) for nid, d in visited.items()]

    def _get_neighbors_with_depth(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None,
        depth: int,
        direction: str,
    ) -> list[tuple[dict[str, Any], int]]:
        """Dispatch neighbor query to the active backend (Neo4j or NetworkX)."""
        if self._is_neo4j:
            if direction == "out":
                rel_pattern = "-[r*1..{d}]->"
            elif direction == "in":
                rel_pattern = "<-[r*1..{d}]-"
            else:
                rel_pattern = "-[r*1..{d}]-"

            rel_clause = rel_pattern.format(d=depth)
            if edge_types:
                types_str = "|".join(edge_types)
                rel_clause = rel_clause.replace("[r*", f"[r:{types_str}*")

            cypher = f"""
                MATCH (n:_Node) WHERE n.symbol_id = $id
                MATCH p = (n){rel_clause}(m)
                WHERE n <> m
                WITH m, min(length(p)) AS shortest_depth
                RETURN properties(m) AS node, shortest_depth AS depth
            """
            result = self.store.query(cypher, {"id": node_id})
            return [
                (row["node"], int(row["depth"]))
                for row in result.records
                if "node" in row
            ]

        return self._get_neighbors_networkx(
            node_id, edge_types=edge_types, depth=depth, direction=direction
        )

    def _to_retrieval_result(
        self, props: dict[str, Any], depth: int
    ) -> GraphRetrievalResult:
        """Convert node property dict to a GraphRetrievalResult."""
        # depth is already validated by callers; guard anyway.
        score = 1.0 / max(depth, 1)
        return GraphRetrievalResult(
            score=score,
            file_path=props.get("file_path") or "",
            start_line=props.get("start_line"),
            end_line=props.get("end_line"),
            symbol_name=_symbol_name_from_props(props),
            chunk_text=_build_chunk_text(props),
            metadata=props,
            depth=depth,
        )

    def _get_neighbors_filtered(
        self,
        symbol: str,
        depth: int,
        direction: str,
        edge_types: list[str] | None,
    ) -> list[GraphRetrievalResult]:
        """Core implementation: resolve symbol and run filtered BFS/graph traversal."""
        self._validate_depth(depth)
        self._validate_direction(direction)
        self._validate_edge_types(edge_types)

        resolved_ids = self._resolve_symbol_ids(symbol)
        if not resolved_ids:
            return []

        resolved_set = set(resolved_ids)
        min_depths: dict[str, int] = {}
        node_props: dict[str, dict[str, Any]] = {}

        for start_id in resolved_ids:
            for props, d in self._get_neighbors_with_depth(
                start_id, edge_types=edge_types, depth=depth, direction=direction
            ):
                sid = props.get("symbol_id")
                if not sid or sid in resolved_set:
                    continue
                if sid not in min_depths or d < min_depths[sid]:
                    min_depths[sid] = d
                    node_props[sid] = props

        results = [
            self._to_retrieval_result(node_props[sid], d)
            for sid, d in min_depths.items()
        ]
        results.sort(key=lambda x: (x.depth, x.symbol_name or ""))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_callers(self, symbol: str, depth: int = 1) -> list[GraphRetrievalResult]:
        """Return symbols that call *symbol* up to *depth* hops (incoming CALLS edges)."""
        return self._get_neighbors_filtered(
            symbol, depth=depth, direction="in", edge_types=["CALLS"]
        )

    def get_callees(self, symbol: str, depth: int = 1) -> list[GraphRetrievalResult]:
        """Return symbols called by *symbol* up to *depth* hops (outgoing CALLS edges)."""
        return self._get_neighbors_filtered(
            symbol, depth=depth, direction="out", edge_types=["CALLS"]
        )

    def get_neighbors(
        self,
        symbol: str,
        depth: int = 1,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> list[GraphRetrievalResult]:
        """Return neighbor symbols reachable up to *depth* hops in the graph."""
        return self._get_neighbors_filtered(
            symbol, depth=depth, direction=direction, edge_types=edge_types
        )

    def _find_paths_networkx(
        self,
        src_id: str,
        tgt_id: str,
        max_depth: int,
        shortest_only: bool,
    ) -> list[list[dict[str, Any]]]:
        """Query paths on NetworkX backend."""
        import networkx as nx

        g = self._nx_graph()
        if src_id not in g or tgt_id not in g:
            return []

        graph = g.to_undirected(as_view=True)

        if shortest_only:
            try:
                path_nodes = nx.shortest_path(graph, source=src_id, target=tgt_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
            if len(path_nodes) - 1 > max_depth:
                return []
            return [[dict(g.nodes[nid]) for nid in path_nodes]]

        paths: list[list[dict[str, Any]]] = []
        try:
            for path_nodes in nx.all_simple_paths(
                graph, source=src_id, target=tgt_id, cutoff=max_depth
            ):
                paths.append([dict(g.nodes[nid]) for nid in path_nodes])
        except nx.NodeNotFound:
            pass
        return paths

    def _find_paths_neo4j(
        self,
        src_id: str,
        tgt_id: str,
        max_depth: int,
        shortest_only: bool,
    ) -> list[list[dict[str, Any]]]:
        """Query paths on Neo4j backend."""
        if shortest_only:
            path_nodes = self.store.shortest_path(src_id, tgt_id)
            if not path_nodes or len(path_nodes) - 1 > max_depth:
                return []
            return [path_nodes]

        cypher = f"""
            MATCH (a:_Node) WHERE a.symbol_id = $src
            MATCH (b:_Node) WHERE b.symbol_id = $tgt
            MATCH p = (a)-[*1..{max_depth}]-(b)
            RETURN [n IN nodes(p) | properties(n)] AS path_nodes
        """
        result = self.store.query(cypher, {"src": src_id, "tgt": tgt_id})
        paths: list[list[dict[str, Any]]] = []
        for row in result.records:
            nodes = row.get("path_nodes") or []
            if not nodes:
                continue
            # Discard non-simple paths (any repeated symbol_id).
            seen: set[str] = set()
            simple = True
            for n in nodes:
                sid = n.get("symbol_id")
                if sid in seen:
                    simple = False
                    break
                if sid:
                    seen.add(sid)
            if simple:
                paths.append(nodes)
        return paths

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
        shortest_only: bool = True,
    ) -> list[GraphPathResult]:
        """Find paths between two symbols up to *max_depth* hops.

        Args:
            source: Name, qualified name, or symbol_id of the source symbol.
            target: Name, qualified name, or symbol_id of the target symbol.
            max_depth: Maximum number of hops allowed (1 <= max_depth <= 15).
            shortest_only: If ``True``, return only the single shortest path.

        Returns:
            A deduplicated list of :class:`GraphPathResult` sorted by path
            length ascending, then alphabetically by symbol sequence.

        Raises:
            ValueError: If *max_depth* is out of the allowed range.
        """
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")
        if max_depth > _MAX_ALLOWED_PATH_DEPTH:
            raise ValueError(
                f"max_depth must be <= {_MAX_ALLOWED_PATH_DEPTH}, got {max_depth!r}"
            )

        src_ids = self._resolve_symbol_ids(source)
        tgt_ids = self._resolve_symbol_ids(target)
        if not src_ids or not tgt_ids:
            return []

        raw_paths: list[list[dict[str, Any]]] = []
        for src_id in src_ids:
            for tgt_id in tgt_ids:
                if self._is_neo4j:
                    raw_paths.extend(
                        self._find_paths_neo4j(src_id, tgt_id, max_depth, shortest_only)
                    )
                else:
                    raw_paths.extend(
                        self._find_paths_networkx(
                            src_id, tgt_id, max_depth, shortest_only
                        )
                    )

        seen_path_keys: set[tuple[str, ...]] = set()
        results: list[GraphPathResult] = []
        for path_nodes in raw_paths:
            if not path_nodes:
                continue
            symbols = [_symbol_name_from_props(n) or "" for n in path_nodes]
            path_key = tuple(symbols)
            if path_key in seen_path_keys:
                continue
            seen_path_keys.add(path_key)

            tgt_props = path_nodes[-1]
            results.append(
                GraphPathResult(
                    score=1.0 / len(symbols),
                    file_path=tgt_props.get("file_path") or "",
                    start_line=tgt_props.get("start_line"),
                    end_line=tgt_props.get("end_line"),
                    symbol_name=_symbol_name_from_props(tgt_props),
                    chunk_text=_build_chunk_text(tgt_props),
                    metadata={"path_symbols": symbols, "path_nodes": path_nodes},
                    symbols=symbols,
                )
            )

        results.sort(key=lambda x: (len(x.symbols), -x.score, x.symbols))
        return results

    def get_subgraph(
        self,
        symbols: list[str],
        depth: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract the induced subgraph around a set of starting symbols.

        Resolves all starting symbols in one pass, expands each N hops to
        collect the neighbourhood, then delegates to the backend's
        ``subgraph()`` for a single efficient induced-subgraph query.

        Args:
            symbols: List of symbol names or IDs to expand around.
            depth: Expansion depth for neighbourhood collection (>= 1).

        Returns:
            ``(nodes, edges)`` tuple of property dicts.

        Raises:
            ValueError: If *depth* is out of the allowed range.
        """
        self._validate_depth(depth)
        if not symbols:
            return [], []

        # Resolve all starting symbols
        start_ids: set[str] = set()
        for sym in symbols:
            if sym:
                start_ids.update(self._resolve_symbol_ids(sym))

        if not start_ids:
            return [], []

        # Expand neighbourhood: one backend call per starting node.
        all_ids = set(start_ids)
        for sid in start_ids:
            for props, _ in self._get_neighbors_with_depth(
                sid, edge_types=None, depth=depth, direction="both"
            ):
                nid = props.get("symbol_id")
                if nid:
                    all_ids.add(nid)

        return self.store.subgraph(list(all_ids))

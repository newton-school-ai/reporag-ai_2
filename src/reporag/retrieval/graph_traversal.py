"""Graph-based retrieval.

Answers structural questions that vector search and BM25 cannot: "What
functions call ``authenticate_user``?" or "Trace the path from the router
to the database layer." Vector search blurs structurally-distinct but
semantically-similar symbols together; graph traversal follows typed edges
(``CALLS``, ``IMPORTS``, ``INHERITS``) in the knowledge graph built by
Issues 9, 10, and 12.

Design
------
:class:`GraphRetriever` is a thin query-side wrapper around the
:class:`~reporag.graph.neo4j_store.GraphStoreProtocol` interface -- all
graph construction and edge resolution happens in the upstream pipeline.
This module only:

* Delegates traversal to the backend via the three protocol methods
  (``get_neighbors``, ``shortest_path``, ``subgraph``) so that the
  caller is insulated from Neo4j Cypher or NetworkX BFS details,
* Converts raw node-property dicts into the unified
  :class:`~reporag.retrieval.vector_search.RetrievalResult` schema so
  downstream fusion (:mod:`reporag.retrieval.fusion`, Issue 19) can
  treat all three retrieval paths interchangeably,
* Synthesises ``chunk_text`` from the ``signature`` and ``docstring``
  node properties, since the graph store does not hold full source text,
* Maps traversal hop distance to a strictly-positive pseudo-score
  ``1.0 / (distance + 1.0)`` (1.0 for the source, 0.5 for 1 hop, 0.33
  for 2 hops, ...) that is compatible with Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reporag.graph.neo4j_store import GraphStore, GraphStoreProtocol
from reporag.graph.symbol_table import SymbolTable
from reporag.retrieval.vector_search import RetrievalResult


class SymbolNotFoundError(Exception):
    pass


class AmbiguousSymbolError(Exception):
    pass


@dataclass
class GraphPaths:
    """Paths between two symbols.

    Attributes:
        shortest: The shortest path as a list of RetrievalResult.
        all_paths: All simple paths up to max_depth, each as a list of RetrievalResult.
    """

    shortest: list[RetrievalResult]
    all_paths: list[list[RetrievalResult]]


@dataclass
class GraphSubgraph:
    """Nodes and edges of an induced subgraph.

    Attributes:
        nodes: One :class:`~reporag.retrieval.vector_search.RetrievalResult`
            per node in the induced subgraph, all at distance 0 (score 1.0).
        edges: Raw edge dicts from the underlying store; each dict
            contains at least ``source``, ``target``, and ``type`` keys.
    """

    nodes: list[RetrievalResult]
    edges: list[dict[str, Any]]


class GraphRetriever:
    """Graph-based retriever for structural repository queries.

    Uses a GraphStore backend (Neo4j or NetworkX fallback) to traverse
    the code knowledge graph. Results are normalized into the common
    RetrievalResult schema for downstream fusion.
    """

    def __init__(
        self,
        store: GraphStoreProtocol | None = None,
        neo4j_uri: str | None = None,
        fallback: bool = True,
        symbol_table: SymbolTable | None = None,
    ) -> None:
        """Initialize the GraphRetriever.

        Args:
            store: An existing GraphStoreProtocol implementation.
            neo4j_uri: If store is not provided, connects to this URI.
            fallback: If true, falls back to NetworkXGraphStore on failure.
            symbol_table: Optional SymbolTable to resolve names to ids.
        """
        if store is not None:
            self._store = store
        else:
            uri = neo4j_uri or "bolt://localhost:7687"
            self._store = GraphStore(uri=uri, fallback=fallback)
        self.symbol_table = symbol_table

    def _resolve(self, symbol: str) -> str:
        """Resolve a symbol name to a symbol_id using the symbol table."""
        if not self.symbol_table:
            return symbol
        record = self.symbol_table.lookup_qualified(symbol)
        if record:
            return record.symbol_id
        records = self.symbol_table.lookup(symbol)
        if not records:
            raise SymbolNotFoundError(f"Symbol {symbol!r} not found in symbol table.")
        if len(records) > 1:
            raise AmbiguousSymbolError(f"Symbol {symbol!r} is ambiguous.")
        return records[0].symbol_id

    def _node_to_result(
        self, node: dict[str, Any], distance: int = 0
    ) -> RetrievalResult:
        """Convert a graph node dictionary to a RetrievalResult.

        Uses distance to compute a pseudo-score (1.0 / (distance + 1)).
        Synthesizes chunk_text from signature and docstring if available.
        """
        signature = node.get("signature", "")
        docstring = node.get("docstring", "")
        name = node.get("name", "")

        chunk_parts = []
        if signature:
            chunk_parts.append(signature)
        elif name:
            chunk_parts.append(name)

        if docstring:
            chunk_parts.append(f'"""{docstring}"""')

        chunk_text = "\n".join(chunk_parts)
        if not chunk_text:
            chunk_text = name

        metadata = dict(node)
        # We don't want to duplicate fields that have dedicated attributes,
        # but the schema allows keeping them in metadata too.

        # Distance is the number of hops from the query node.
        # We give 1.0 to the node itself (dist 0), 0.5 to dist 1, 0.33 to dist 2, etc.
        # All call sites pass non-negative values (depth arg or enumerate index).
        score = 1.0 / (distance + 1.0)

        return RetrievalResult(
            score=score,
            file_path=node.get("file_path", ""),
            start_line=node.get("start_line"),
            end_line=node.get("end_line"),
            symbol_name=node.get("name"),
            chunk_text=chunk_text,
            metadata=metadata,
        )

    def _get_neighbors_ring_by_ring(
        self,
        symbol_id: str,
        depth: int,
        direction: str,
        edge_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Helper to compute exact hop distances by querying ring-by-ring."""
        hop_by_id: dict[str, int] = {}
        node_by_id: dict[str, Any] = {}
        for hop in range(1, depth + 1):
            ring = self._store.get_neighbors(
                symbol_id, edge_types=edge_types, depth=hop, direction=direction
            )
            for node in ring:
                nid = node.get("symbol_id")
                if nid is not None and nid not in hop_by_id:
                    hop_by_id[nid] = hop
                    node_by_id[nid] = node
        return [
            self._node_to_result(node_by_id[nid], distance=hop_by_id[nid])
            for nid in hop_by_id
        ]

    def get_neighbors(
        self, symbol: str, depth: int = 1, direction: str = "both"
    ) -> list[RetrievalResult]:
        """Return N-hop neighbors of the given symbol in the graph.

        Args:
            symbol: The symbol name or symbol_id of the starting node.
            depth: Maximum number of hops.
            direction: 'out', 'in', or 'both'.

        Returns:
            List of RetrievalResult.
        """
        symbol_id = self._resolve(symbol)
        return self._get_neighbors_ring_by_ring(symbol_id, depth, direction)

    def get_callers(self, symbol: str, depth: int = 2) -> list[RetrievalResult]:
        """Return symbols that call the given symbol within ``depth`` hops.

        Each result's score reflects its true hop distance -- a direct caller
        (hop 1) scores higher than a transitive caller (hop 2).

        Args:
            symbol: The symbol name or symbol_id of the target node.
            depth: Maximum number of hops.

        Returns:
            List of RetrievalResult representing the callers.
        """
        symbol_id = self._resolve(symbol)
        return self._get_neighbors_ring_by_ring(
            symbol_id, depth, "in", edge_types=["CALLS"]
        )

    def find_paths(self, source: str, target: str, max_depth: int = 5) -> GraphPaths:
        """Find the shortest path and all simple paths between two symbols.

        Args:
            source: Starting symbol name or symbol_id.
            target: Ending symbol name or symbol_id.
            max_depth: Paths longer than this are excluded from both shortest
                and all_paths.

        Returns:
            A GraphPaths containing the shortest path and all valid simple paths.
        """
        source_id = self._resolve(source)
        target_id = self._resolve(target)

        shortest_nodes = self._store.shortest_path(source_id, target_id)
        if shortest_nodes and (len(shortest_nodes) - 1) > max_depth:
            shortest_nodes = []

        shortest = [
            self._node_to_result(n, distance=i) for i, n in enumerate(shortest_nodes)
        ]

        all_paths = []
        if shortest_nodes:
            # We use DFS over get_neighbors to find all paths up to max_depth
            # without breaking the GraphStoreProtocol boundaries.
            nodes_cache: dict[str, Any] = {}
            source_nodes, _ = self._store.subgraph([source_id])
            if source_nodes:
                nodes_cache[source_id] = source_nodes[0]

                # Stack holds: (current_id, path_of_ids, set_of_visited_ids)
                stack = [(source_id, [source_id], {source_id})]

                while stack:
                    curr, path, visited = stack.pop()
                    if curr == target_id:
                        all_paths.append(
                            [
                                self._node_to_result(nodes_cache[nid], distance=i)
                                for i, nid in enumerate(path)
                            ]
                        )
                        continue

                    if len(path) - 1 >= max_depth:
                        continue

                    ring = self._store.get_neighbors(curr, depth=1, direction="both")
                    for node in ring:
                        nid = node.get("symbol_id")
                        if nid:
                            nodes_cache[nid] = node
                            if nid not in visited:
                                new_visited = set(visited)
                                new_visited.add(nid)
                                stack.append((nid, path + [nid], new_visited))

        # Optional: reverse so shorter paths tend to appear first if needed
        all_paths.reverse()
        return GraphPaths(shortest=shortest, all_paths=all_paths)

    def extract_subgraph(self, symbols: list[str]) -> GraphSubgraph:
        """Extract the induced subgraph over the given symbols.

        Returns:
            A :class:`GraphSubgraph` with one :class:`RetrievalResult` per node
            (at distance 0, score 1.0) and the raw edge dicts connecting those
            nodes.  Each edge dict contains at least ``source``, ``target``,
            and ``type`` keys as provided by the underlying store.
        """
        symbol_ids = [self._resolve(sym) for sym in symbols]
        nodes, edges = self._store.subgraph(symbol_ids)
        return GraphSubgraph(
            nodes=[self._node_to_result(n, distance=0) for n in nodes],
            edges=edges,
        )

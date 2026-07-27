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

Known limitations
-----------------
* ``find_paths`` returns only the **shortest** path.
  ``GraphStoreProtocol.shortest_path`` does not expose an all-paths
  variant; enumerating all paths is left for a future protocol extension.
* ``max_depth`` accepted by ``find_paths`` is not forwarded to the
  backend because ``GraphStoreProtocol.shortest_path`` has no depth
  bound parameter.  It is kept in the signature for API symmetry with
  the Issue 18 acceptance criteria.
* All nodes returned by ``get_neighbors`` and ``get_callers`` receive
  the same pseudo-score (``1.0 / (depth + 1.0)``) because the
  ``GraphStoreProtocol`` returns a flat list without per-node hop counts.
"""

from __future__ import annotations

from typing import Any

from reporag.graph.neo4j_store import GraphStore, GraphStoreProtocol
from reporag.retrieval.vector_search import RetrievalResult


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
    ) -> None:
        """Initialize the GraphRetriever.

        Args:
            store: An existing GraphStoreProtocol implementation.
            neo4j_uri: If store is not provided, connects to this URI.
            fallback: If true, falls back to NetworkXGraphStore on failure.
        """
        if store is not None:
            self._store = store
        else:
            uri = neo4j_uri or "bolt://localhost:7687"
            self._store = GraphStore(uri=uri, fallback=fallback)

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

    def get_neighbors(
        self, symbol_id: str, depth: int = 1, direction: str = "both"
    ) -> list[RetrievalResult]:
        """Return N-hop neighbors of the given symbol in the graph.

        Args:
            symbol_id: The symbol_id of the starting node.
            depth: Maximum number of hops.
            direction: 'out', 'in', or 'both'.

        Returns:
            List of RetrievalResult.
        """
        nodes = self._store.get_neighbors(symbol_id, depth=depth, direction=direction)
        # Note: get_neighbors in the store doesn't return the exact distance,
        # but we can assume depth is the max. For simplicity, we just assign the depth as distance
        # or we could compute accurate shortest paths. We'll use depth for the pseudo-score.
        return [self._node_to_result(n, distance=depth) for n in nodes]

    def get_callers(self, symbol_id: str, depth: int = 2) -> list[RetrievalResult]:
        """Return symbols that call the given symbol within `depth` hops.

        Args:
            symbol_id: The symbol_id of the target node.
            depth: Maximum number of hops.

        Returns:
            List of RetrievalResult representing the callers.
        """
        # Call edges are caller -> callee. So to find callers, we traverse 'in' edges.
        nodes = self._store.get_neighbors(
            symbol_id, edge_types=["CALLS"], depth=depth, direction="in"
        )
        return [self._node_to_result(n, distance=depth) for n in nodes]

    def find_paths(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[RetrievalResult]:
        """Find the shortest path between two symbols.

        Args:
            source_id: Starting symbol_id.
            target_id: Ending symbol_id.
            max_depth: Included for API compatibility with acceptance criteria,
                       though shortest_path finds the shortest without depth bound in NetworkX.

        Returns:
            List of RetrievalResult for the nodes in the path, in order.
            Returns empty list if no path is found.
        """
        # Note: Issue #18 requests "shortest + all paths", but GraphStoreProtocol
        # currently only supports `shortest_path`. We document this limitation here.
        nodes = self._store.shortest_path(source_id, target_id)
        # For a path, distance can be the index in the path.
        return [self._node_to_result(n, distance=i) for i, n in enumerate(nodes)]

    def extract_subgraph(self, symbol_ids: list[str]) -> list[RetrievalResult]:
        """Extract the induced subgraph over the given symbol_ids.

        Returns the nodes in the subgraph. (Edges are retrieved by the underlying
        store, but RetrievalResult only represents nodes. If edge representation is
        needed, they can be stored in the metadata, but typically RAG only needs chunks).
        """
        nodes, _edges = self._store.subgraph(symbol_ids)
        return [self._node_to_result(n, distance=0) for n in nodes]

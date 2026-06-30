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

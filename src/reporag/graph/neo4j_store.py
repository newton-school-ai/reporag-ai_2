"""Neo4j graph store with Cypher query layer.

Persists the code knowledge graph (call graph + dependency graph + symbol
table) in Neo4j. Provides Cypher query helpers for neighbors, shortest
path, and subgraph extraction. Includes NetworkX fallback for testing.
"""

# TODO: Implement in Issue 12
# - Neo4j driver wrapper: connect, create_nodes, create_edges, query, clear
# - Node types: Function, Class, Module (with properties from symbol table)
# - Edge types: CALLS, IMPORTS, INHERITS, CONTAINS
# - Cypher helpers: get_neighbors(node, depth), shortest_path(a, b), subgraph(nodes)
# - Bulk insert with batch transactions for 10K+ nodes
# - NetworkX fallback implementing the same interface
# - Connection error handling with retry logic

"""Call graph builder.

Walks ASTs to identify function call expressions and resolves them to
target symbols. Builds directed edges: caller -> callee with call site
metadata.
"""

# TODO: Implement in Issue 9
# - Identify function_call nodes in AST
# - Resolve calls to target symbols (same file via scope, cross-file via imports)
# - Handle: method calls (self.method), chained calls, constructor calls
# - Build directed edges with metadata: caller, callee, call_site_line, call_site_file
# - Return list of CallEdge dataclass objects

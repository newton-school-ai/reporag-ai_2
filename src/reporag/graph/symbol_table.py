"""Global symbol table / registry.

Central lookup index. Given a symbol name, returns the defining file,
line range, type, and signature. Supports lookup by exact name, fully
qualified name, regex pattern, and file path.
"""

# TODO: Implement in Issue 11
# - Register all symbols with fully qualified names (module.class.method)
# - Lookup by: exact name (all matches), qualified name (unique), regex, file path
# - Disambiguate same-name symbols across files/classes
# - Serializable to/from JSON for persistence and debugging

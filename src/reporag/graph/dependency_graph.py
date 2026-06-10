"""Import dependency graph builder.

Builds directed edges representing module-level import relationships.
Resolves relative imports, handles star imports, detects circular dependencies.
"""

# TODO: Implement in Issue 10
# - Build edges: importing_module -> imported_module
# - Resolve relative imports (from .utils import helper)
# - Handle: import X, from X import Y, from X import *
# - Detect and flag circular import chains
# - Edge metadata: source_module, target_module, import_type, imported_names

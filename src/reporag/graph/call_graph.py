"""Call graph builder from AST.

Walks tree-sitter ASTs to identify function call expressions, resolves them
to their target symbols (same-file or cross-file via import aliases), and
builds a directed edge list:

    caller -> callee  with metadata: call_site_file, call_site_line

Handles:
- Direct calls:          foo()
- Method calls (self):   self.method()
- Attribute calls (obj): obj.method()
- Chained calls:         a.b.c()
- Constructor calls:     MyClass()
- Cross-file calls:      math.sin()  (resolved through import symbols)
- Recursive calls:       function calling itself
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """A directed call edge from one symbol to another.

    Attributes:
        caller: Qualified name of the calling function/method.
        callee: Resolved name of the called function/method.
        call_site_file: Absolute path of the file containing the call.
        call_site_line: 1-based line number of the call expression.
        raw_call_text: Verbatim text of the call as it appears in source.
    """

    caller: str
    callee: str
    call_site_file: str
    call_site_line: int
    raw_call_text: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode(node: Node) -> str:
    """Decode a tree-sitter node's text to str."""
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _extract_call_name(call_node: Node) -> str | None:
    """Extract the callee name string from a ``call`` AST node.

    Handles:
    - ``identifier``        -> ``"foo"``
    - ``attribute``         -> ``"self.method"`` / ``"obj.method"``
    - Chained attributes    -> ``"a.b.c"``
    - Any other expression  -> raw text of the function field (fallback)

    Returns ``None`` when the call node has no recognisable function child.
    """
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None

    if func_node.type == "identifier":
        return _decode(func_node)

    if func_node.type == "attribute":
        # Walk attribute chain bottom-up: a.b.c -> ["c", "b", "a"] -> "a.b.c"
        parts: list[str] = []
        current: Node | None = func_node
        while current is not None and current.type == "attribute":
            attr = current.child_by_field_name("attribute")
            if attr:
                parts.append(_decode(attr))
            current = current.child_by_field_name("object")
        if current is not None:
            parts.append(_decode(current))
        parts.reverse()
        return ".".join(parts)

    # Fallback: grab whatever text the function expression holds
    text = _decode(func_node)
    return text if text else None


# ---------------------------------------------------------------------------
# Scope tracker -- maps (file, line) -> enclosing function qualified name
# ---------------------------------------------------------------------------


@dataclass
class _ScopeRange:
    """A line span inside a file belonging to a function or method."""

    qualified_name: str
    file_path: str
    start_line: int
    end_line: int


def _build_scope_ranges(symbols: list) -> list[_ScopeRange]:
    """Return one :class:`_ScopeRange` for every function/method symbol."""
    ranges: list[_ScopeRange] = []
    for sym in symbols:
        if sym.type in ("function", "method"):
            qname = sym.qualified_name or sym.name
            ranges.append(
                _ScopeRange(
                    qualified_name=qname,
                    file_path=sym.file_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
            )
    return ranges


def _find_caller(
    file_path: str,
    line: int,
    scope_ranges: list[_ScopeRange],
) -> str | None:
    """Return the qualified name of the innermost function containing *line*.

    Among all overlapping scopes we choose the one with the greatest
    ``start_line`` -- that is the most deeply nested (innermost) function.
    Returns ``None`` for module-level calls (no enclosing function).
    """
    candidates = [
        sr
        for sr in scope_ranges
        if sr.file_path == file_path and sr.start_line <= line <= sr.end_line
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda sr: sr.start_line).qualified_name


# ---------------------------------------------------------------------------
# Import resolver -- maps local alias -> fully qualified name
# ---------------------------------------------------------------------------


def _build_import_alias_map(symbols: list) -> dict[str, str]:
    """Build ``{local_alias: resolved_name}`` from import symbols in one file.

    Examples
    --------
    ``import os``                 -> ``{"os": "os"}``
    ``import numpy as np``        -> ``{"np": "numpy"}``
    ``from math import sin``      -> ``{"sin": "math.sin"}``
    ``from .utils import helper`` -> ``{"helper": ".utils.helper"}``
    ``from collections import *`` -> skipped (wildcard)
    """
    alias_map: dict[str, str] = {}
    for sym in symbols:
        if sym.type != "import":
            continue
        if sym.is_wildcard_import:
            continue  # cannot resolve statically

        local_name: str = sym.name
        source: str = sym.import_source or ""

        if sym.import_alias:
            # ``import X as alias`` or ``from M import X as alias``
            # The extractor already stores source as "M.X" for from-imports.
            alias_map[local_name] = source
        else:
            # ``import os`` -> source == "os", local_name == "os"
            # ``from math import sin`` -> source == "math", local_name == "sin"
            if source and local_name != source:
                # from-import: member of a module
                alias_map[local_name] = f"{source}.{local_name}"
            else:
                alias_map[local_name] = local_name or source
    return alias_map


# ---------------------------------------------------------------------------
# AST walker -- collects every ``call`` node in a tree
# ---------------------------------------------------------------------------


def _collect_call_nodes(tree: Tree) -> list[Node]:
    """Return every ``call`` node in *tree* via iterative DFS.

    We always recurse into every node because calls can be nested (e.g.
    ``f(g())``).
    """
    result: list[Node] = []
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            result.append(node)
        stack.extend(reversed(node.children))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CallGraphBuilder:
    """Build a directed call graph from tree-sitter ASTs and extracted symbols.

    Quick usage::

        builder = CallGraphBuilder()
        edges = builder.build_from_symbols(all_symbols, file_asts)
        for e in edges:
            print(f"{e.caller} -> {e.callee}  ({e.call_site_file}:{e.call_site_line})")

    Or from raw source strings (parses internally)::

        edges = builder.build_from_source({"a.py": source_a, "b.py": source_b})
    """

    # ------------------------------------------------------------------
    # Primary API: pre-parsed symbols + ASTs
    # ------------------------------------------------------------------

    def build_from_symbols(
        self,
        symbols: list,
        file_asts: dict[str, Tree],
    ) -> list[CallEdge]:
        """Build call edges from pre-extracted symbols and parsed ASTs.

        Parameters
        ----------
        symbols:
            Flat list of :class:`~src.reporag.ingestion.symbol_extractor.Symbol`
            objects (functions, methods, imports) from all files.
        file_asts:
            ``{file_path: Tree}`` -- one tree per source file.

        Returns
        -------
        list[CallEdge]
            Directed edges; may be empty when no inter-function calls exist.

        Algorithm
        ---------
        1. Build a scope-range index from every function/method symbol so we
           can look up ``"who is calling?"`` by (file, line).
        2. Build a per-file import-alias map so ``np.zeros`` resolves to
           ``numpy.zeros``, ``sin`` resolves to ``math.sin``, etc.
        3. Collect every ``call`` AST node in every file.
        4. For each call: identify the caller via scope lookup, resolve the
           callee via the alias map, and emit a :class:`CallEdge`.
        """
        if not symbols or not file_asts:
            return []

        scope_ranges = _build_scope_ranges(symbols)

        # Group import symbols by file for per-file alias resolution
        imports_by_file: dict[str, list] = {}
        for sym in symbols:
            if sym.type == "import":
                imports_by_file.setdefault(sym.file_path, []).append(sym)

        edges: list[CallEdge] = []

        for file_path, tree in file_asts.items():
            file_imports = imports_by_file.get(file_path, [])
            alias_map = _build_import_alias_map(file_imports)

            for call_node in _collect_call_nodes(tree):
                edge = self._make_edge(call_node, file_path, scope_ranges, alias_map)
                if edge is not None:
                    edges.append(edge)

        return edges

    # ------------------------------------------------------------------
    # Convenience API: raw source strings
    # ------------------------------------------------------------------

    def build_from_source(
        self,
        source_map: dict[str, str],
        language: str = "python",
    ) -> list[CallEdge]:
        """Build call edges from ``{file_path: source_code}`` in one step.

        Internally creates an :class:`~src.reporag.ingestion.parser.ASTParser`
        and :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`,
        then delegates to :meth:`build_from_symbols`.

        Parameters
        ----------
        source_map:
            Mapping of file path to source code string.
        language:
            Tree-sitter language name (default ``"python"``).
        """
        # Lazy import to avoid circular dependency at module level
        from src.reporag.ingestion.parser import ASTParser
        from src.reporag.ingestion.symbol_extractor import SymbolExtractor

        parser = ASTParser()
        extractor = SymbolExtractor(parser=parser)

        all_symbols: list = []
        file_asts: dict[str, Tree] = {}

        for file_path, source in source_map.items():
            tree = parser.parse(source, language=language)
            file_asts[file_path] = tree
            syms = extractor.extract_from_tree(
                tree, file_path, source, language=language
            )
            all_symbols.extend(syms)

        return self.build_from_symbols(all_symbols, file_asts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_edge(
        self,
        call_node: Node,
        file_path: str,
        scope_ranges: list[_ScopeRange],
        alias_map: dict[str, str],
    ) -> CallEdge | None:
        """Attempt to produce a :class:`CallEdge` for a single ``call`` node.

        Returns ``None`` when:

        - The call has no resolvable function name.
        - The call is at module scope (no enclosing function/method).
        """
        raw_name = _extract_call_name(call_node)
        if not raw_name:
            return None

        call_line = call_node.start_point[0] + 1  # 1-based

        caller = _find_caller(file_path, call_line, scope_ranges)
        if caller is None:
            return None  # module-level call -- skip

        callee = _resolve_callee(raw_name, alias_map)
        raw_text = _decode(call_node)

        return CallEdge(
            caller=caller,
            callee=callee,
            call_site_file=file_path,
            call_site_line=call_line,
            raw_call_text=raw_text,
        )


# ---------------------------------------------------------------------------
# Callee resolution (module-level so it can be unit-tested independently)
# ---------------------------------------------------------------------------


def _resolve_callee(raw_name: str, alias_map: dict[str, str]) -> str:
    """Resolve *raw_name* through *alias_map* to a fully qualified name.

    Strategy
    --------
    1. Split on the first ``.``: ``"np.zeros"`` -> root=``"np"``, rest=``".zeros"``.
    2. If root is in the map, replace it: ``"np"`` -> ``"numpy"`` -> ``"numpy.zeros"``.
    3. Else if the entire name is a map key, replace it directly (covers bare
       from-imports: ``"sin"`` -> ``"math.sin"``).
    4. Otherwise return the raw name unchanged.

    Examples
    --------
    ``("sin",      {"sin": "math.sin"})``  -> ``"math.sin"``
    ``("np.zeros", {"np": "numpy"})``      -> ``"numpy.zeros"``
    ``("self.run", {})``                   -> ``"self.run"``
    ``("os.path.join", {"os": "os"})``     -> ``"os.path.join"``
    """
    parts = raw_name.split(".", 1)
    root = parts[0]
    rest = f".{parts[1]}" if len(parts) > 1 else ""

    if root in alias_map:
        return f"{alias_map[root]}{rest}"

    if raw_name in alias_map:
        return alias_map[raw_name]

    return raw_name

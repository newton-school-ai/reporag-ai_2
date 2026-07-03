"""Call graph builder from AST.

Walks tree-sitter ASTs to identify function call expressions, resolves them
to their target symbols (same-file or cross-file via import aliases), and
builds a directed edge list::

    caller -> callee  (call_site_file, call_site_line)

Handles
-------
- Direct calls:            foo()
- Method calls (self):     self.method()
- Attribute calls (obj):   obj.method()
- Chained calls:           a.b.c()
- Super calls:             super().__init__()
- Constructor calls:       MyClass()
- Cross-file calls:        math.sin()  (resolved via import alias map)
- Recursive calls:         function calling itself
- Nested functions:        innermost scope wins via start_line tiebreak
- Async / await calls:     await get_data()
- Module-level calls:      skipped (no enclosing function scope)

Public API
----------
- :class:`CallEdge`         -- data model for one directed edge
- :class:`CallGraph`        -- queryable container; wraps a list of edges
- :class:`CallGraphBuilder` -- builds a :class:`CallGraph` from symbols + ASTs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model: CallEdge
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """A directed call edge from one symbol to another.

    Attributes
    ----------
    caller:
        Qualified name of the calling function/method, e.g. ``"Engine.start"``
        or ``"outer.<locals>.inner"``.  Mirrors the ``qualified_name`` field of
        :class:`~src.reporag.ingestion.symbol_extractor.Symbol`.
    callee:
        Resolved name of the called function/method.  Import aliases are
        expanded so ``np.zeros`` becomes ``numpy.zeros`` and ``sin`` (from
        ``from math import sin``) becomes ``math.sin``.
    call_site_file:
        Absolute (or relative) path of the file that contains the call
        expression.
    call_site_line:
        1-based line number of the call expression inside *call_site_file*.
    raw_call_text:
        Verbatim source text of the full call expression, truncated to 200
        chars.  Useful for debugging and for Neo4j edge properties.
    """

    caller: str
    callee: str
    call_site_file: str
    call_site_line: int
    raw_call_text: str = field(default="")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of all edge fields.

        Suitable for storage as a Qdrant payload, Neo4j edge property bag,
        or JSONL debugging output.
        """
        return {
            "caller": self.caller,
            "callee": self.callee,
            "call_site_file": self.call_site_file,
            "call_site_line": self.call_site_line,
            "raw_call_text": self.raw_call_text,
        }


# ---------------------------------------------------------------------------
# Queryable container: CallGraph
# ---------------------------------------------------------------------------


class CallGraph:
    """Queryable container for a set of :class:`CallEdge` objects.

    Wraps a flat list of edges and provides O(1) lookup helpers that Issue 12
    (Neo4j store) and Issue 18 (graph retrieval) depend on.

    Parameters
    ----------
    edges:
        Initial list of edges; may be empty.
    """

    def __init__(self, edges: list[CallEdge] | None = None) -> None:
        self._edges: list[CallEdge] = list(edges or [])
        # Lazy-built indexes, invalidated on mutation
        self._callees_index: dict[str, list[CallEdge]] | None = None
        self._callers_index: dict[str, list[CallEdge]] | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_edge(self, edge: CallEdge) -> None:
        """Append *edge* and invalidate lookup indexes."""
        self._edges.append(edge)
        self._callees_index = None
        self._callers_index = None

    def extend(self, edges: list[CallEdge]) -> None:
        """Extend with multiple edges at once."""
        self._edges.extend(edges)
        self._callees_index = None
        self._callers_index = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def edges(self) -> list[CallEdge]:
        """All edges in insertion order (read-only view)."""
        return list(self._edges)

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self):
        return iter(self._edges)

    def __bool__(self) -> bool:
        return bool(self._edges)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def callees_of(self, caller: str) -> list[CallEdge]:
        """Return all edges where *caller* is the calling function.

        Parameters
        ----------
        caller:
            Qualified name (exact match, case-sensitive).
        """
        if self._callees_index is None:
            self._callees_index = {}
            for e in self._edges:
                self._callees_index.setdefault(e.caller, []).append(e)
        return self._callees_index.get(caller, [])

    def callers_of(self, callee: str) -> list[CallEdge]:
        """Return all edges where *callee* is the called function.

        Parameters
        ----------
        callee:
            Resolved callee name (exact match, case-sensitive).
        """
        if self._callers_index is None:
            self._callers_index = {}
            for e in self._edges:
                self._callers_index.setdefault(e.callee, []).append(e)
        return self._callers_index.get(callee, [])

    def unique_callers(self) -> list[str]:
        """Return sorted list of all unique caller qualified names."""
        return sorted({e.caller for e in self._edges})

    def unique_callees(self) -> list[str]:
        """Return sorted list of all unique callee names."""
        return sorted({e.callee for e in self._edges})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the full graph."""
        return {
            "edge_count": len(self._edges),
            "edges": [e.to_dict() for e in self._edges],
        }

    def __repr__(self) -> str:
        return f"CallGraph(edges={len(self._edges)})"


# ---------------------------------------------------------------------------
# Internal: AST helpers
# ---------------------------------------------------------------------------


def _decode(node: Node) -> str:
    """Safely decode a tree-sitter node's bytes to UTF-8 str."""
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _extract_call_name(call_node: Node) -> str | None:
    """Extract a dotted callee name from a ``call`` AST node.

    Resolves the ``function`` field of the call node:

    - ``identifier``              -> ``"foo"``
    - ``attribute``               -> ``"self.method"`` / ``"obj.method"``
    - Chained ``attribute``       -> ``"a.b.c"``
    - ``attribute`` where object  is itself a ``call`` (e.g. ``super().__init__``)
                                  -> ``"<call>.__init__"`` (caller text kept)
    - Any other expression        -> raw text of the function node (fallback)

    Returns ``None`` if no function child exists.
    """
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None

    if func_node.type == "identifier":
        return _decode(func_node)

    if func_node.type == "attribute":
        return _walk_attribute(func_node)

    # Fallback: grab whatever text exists for the function expression
    text = _decode(func_node)
    return text if text else None


def _walk_attribute(node: Node) -> str:
    """Recursively walk an ``attribute`` node into a dotted name.

    When the object side is itself a ``call`` (e.g. ``super()``),
    we represent it as the raw call text, keeping things readable::

        super().__init__() -> "super().__init__"

    This preserves the semantic intent while staying representable as a
    plain string.
    """
    parts: list[str] = []
    current: Node | None = node
    while current is not None and current.type == "attribute":
        attr = current.child_by_field_name("attribute")
        if attr:
            parts.append(_decode(attr))
        obj = current.child_by_field_name("object")
        if obj is not None and obj.type == "call":
            # e.g. super() in super().__init__
            parts.append(_decode(obj))
            current = None
        else:
            current = obj
    if current is not None:
        parts.append(_decode(current))
    parts.reverse()
    return ".".join(parts)


def _collect_call_nodes(tree: Tree) -> list[Node]:
    """Return every ``call`` AST node via iterative DFS.

    Always recurses into all children so nested calls like ``f(g())``
    are both captured.
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
# Internal: scope tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ScopeRange:
    """Line span of a function or method symbol inside one file."""

    qualified_name: str
    file_path: str
    start_line: int
    end_line: int


def _build_scope_ranges(symbols: list) -> list[_ScopeRange]:
    """Build one :class:`_ScopeRange` per function/method symbol."""
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
    line: int,
    file_scope_ranges: list[_ScopeRange],
) -> str | None:
    """Return the innermost enclosing function's qualified name at *line*.

    Among all scopes that contain *line*, the one with the highest
    ``start_line`` is the most deeply nested (innermost), matching Python's
    scoping rules.  Returns ``None`` for module-level calls.
    """
    candidates = [
        sr for sr in file_scope_ranges if sr.start_line <= line <= sr.end_line
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda sr: sr.start_line).qualified_name


# ---------------------------------------------------------------------------
# Internal: import resolver
# ---------------------------------------------------------------------------


def _build_import_alias_map(symbols: list) -> dict[str, str]:
    """Build ``{local_name: fully_qualified_name}`` from a file's import symbols.

    Rules
    -----
    ``import os``                 -> ``{"os": "os"}``
    ``import numpy as np``        -> ``{"np": "numpy"}``
    ``from math import sin``      -> ``{"sin": "math.sin"}``
    ``from math import cos as c`` -> ``{"c": "math.cos"}``
    ``from .utils import helper`` -> ``{"helper": ".utils.helper"}``
    ``from collections import *`` -> skipped (unresolvable statically)
    """
    alias_map: dict[str, str] = {}
    for sym in symbols:
        if sym.type != "import":
            continue
        if sym.is_wildcard_import:
            continue  # cannot resolve individual names statically

        local_name: str = sym.name
        source: str = sym.import_source or ""

        if sym.import_alias:
            # ``import X as Y``  or  ``from M import X as Y``
            # The extractor already stores import_source as "M.X" for from-aliased.
            alias_map[local_name] = source
        else:
            # ``import os`` -> source == "os", local == "os"  (same -> just map)
            # ``from math import sin`` -> source == "math", local == "sin"
            if source and local_name != source:
                alias_map[local_name] = f"{source}.{local_name}"
            else:
                alias_map[local_name] = local_name or source
    return alias_map


def _resolve_callee(raw_name: str, alias_map: dict[str, str]) -> str:
    """Expand *raw_name* using *alias_map* to get a fully qualified callee.

    Strategy
    --------
    1. Try the whole name first (from-imports: ``"sin"`` -> ``"math.sin"``).
    2. Split on first ``.`` and try the root (module aliases: ``"np.zeros"``
       -> root=``"np"`` -> ``"numpy"`` -> ``"numpy.zeros"``).
    3. Return *raw_name* unchanged if nothing matches.

    Examples
    --------
    ``("sin",       {"sin": "math.sin"})`` -> ``"math.sin"``
    ``("np.zeros",  {"np": "numpy"})``     -> ``"numpy.zeros"``
    ``("self.run",  {})``                  -> ``"self.run"``
    ``("os.path.join", {"os": "os"})``     -> ``"os.path.join"``
    """
    # Whole-name lookup first (bare from-imports like `sin`)
    if raw_name in alias_map:
        return alias_map[raw_name]

    # Root-prefix lookup (module alias like `np.zeros`)
    parts = raw_name.split(".", 1)
    root = parts[0]
    rest = f".{parts[1]}" if len(parts) > 1 else ""
    if root in alias_map:
        return f"{alias_map[root]}{rest}"

    return raw_name


# ---------------------------------------------------------------------------
# Public API: CallGraphBuilder
# ---------------------------------------------------------------------------


class CallGraphBuilder:
    """Build a directed call graph from tree-sitter ASTs and extracted symbols.

    The builder is stateless -- create once and reuse across many repos.

    Quick usage::

        builder = CallGraphBuilder()

        # From pre-parsed symbols and ASTs (primary API):
        graph = builder.build(all_symbols, file_asts)

        # From raw source strings (convenience):
        graph = builder.build_from_source({"a.py": src_a, "b.py": src_b})

        # From file paths on disk:
        graph = builder.build_from_files(["src/a.py", "src/b.py"])

        # Query the result:
        for edge in graph.callees_of("Engine.start"):
            print(f"{edge.caller} -> {edge.callee} @ line {edge.call_site_line}")
    """

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def build(
        self,
        symbols: list,
        file_asts: dict[str, Tree],
    ) -> CallGraph:
        """Build a :class:`CallGraph` from pre-extracted symbols and ASTs.

        This is the primary entry point when the ingestion pipeline has
        already run :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
        and :class:`~src.reporag.ingestion.parser.ASTParser`.

        Parameters
        ----------
        symbols:
            Flat list of all
            :class:`~src.reporag.ingestion.symbol_extractor.Symbol` objects
            (functions, methods, imports) from every file in the repo.
        file_asts:
            ``{file_path: Tree}`` -- one tree-sitter ``Tree`` per file.

        Returns
        -------
        CallGraph
            A queryable call graph; empty when there are no calls.

        Algorithm
        ---------
        1. Build a *scope-range index* from every function/method symbol.
           Each entry maps (file_path, start_line, end_line) -> qualified_name.
        2. Build a *per-file import alias map*: resolves ``np`` -> ``numpy``,
           ``sin`` -> ``math.sin``, etc.
        3. Walk every file's AST collecting all ``call`` nodes (iterative DFS).
        4. For each ``call`` node:
           a. Extract the raw callee name via :func:`_extract_call_name`.
           b. Look up the innermost enclosing function in the scope index
              -- this is the caller.  Skip module-level calls.
           c. Resolve the callee through the alias map.
           d. Emit a :class:`CallEdge`.
        """
        graph = CallGraph()
        if not symbols or not file_asts:
            return graph

        scope_ranges = _build_scope_ranges(symbols)

        # Group scope ranges by file for O(1) file lookup during call resolution
        scopes_by_file: dict[str, list[_ScopeRange]] = {}
        for sr in scope_ranges:
            scopes_by_file.setdefault(sr.file_path, []).append(sr)

        # Group import symbols by file for per-file alias resolution
        imports_by_file: dict[str, list] = {}
        for sym in symbols:
            if sym.type == "import":
                imports_by_file.setdefault(sym.file_path, []).append(sym)

        for file_path, tree in file_asts.items():
            alias_map = _build_import_alias_map(imports_by_file.get(file_path, []))
            file_scopes = scopes_by_file.get(file_path, [])
            for call_node in _collect_call_nodes(tree):
                edge = self._make_edge(call_node, file_path, file_scopes, alias_map)
                if edge is not None:
                    graph.add_edge(edge)

        return graph

    # Legacy compat: build_from_symbols -> build
    def build_from_symbols(
        self,
        symbols: list,
        file_asts: dict[str, Tree],
    ) -> list[CallEdge]:
        """Backward-compatible shim: returns a plain list of :class:`CallEdge`.

        Prefer :meth:`build` for new code -- it returns a :class:`CallGraph`
        with O(1) lookup helpers.
        """
        return self.build(symbols, file_asts).edges

    # ------------------------------------------------------------------
    # Convenience: from raw source strings
    # ------------------------------------------------------------------

    def build_from_source(
        self,
        source_map: dict[str, str],
        language: str = "python",
    ) -> CallGraph:
        """Build a :class:`CallGraph` from ``{file_path: source_code}``.

        Internally creates an :class:`~src.reporag.ingestion.parser.ASTParser`
        and :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`,
        parses each file, and delegates to :meth:`build`.

        Parameters
        ----------
        source_map:
            Mapping of ``file_path -> source_code`` string.
        language:
            Tree-sitter language name (default ``"python"``).
        """
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

        return self.build(all_symbols, file_asts)

    # ------------------------------------------------------------------
    # Convenience: from file paths on disk
    # ------------------------------------------------------------------

    def build_from_files(
        self,
        file_paths: list[str | Path],
        language: str | None = None,
    ) -> CallGraph:
        """Build a :class:`CallGraph` by reading and parsing files from disk.

        Parameters
        ----------
        file_paths:
            List of paths to source files.
        language:
            Override the inferred language for all files.  When ``None``,
            the language is inferred from each file's extension via
            ``settings.extension_map``.
        """
        from src.reporag.ingestion.parser import ASTParser
        from src.reporag.ingestion.symbol_extractor import SymbolExtractor

        parser = ASTParser()
        extractor = SymbolExtractor(parser=parser)

        all_symbols: list = []
        file_asts: dict[str, Tree] = {}

        for fp in file_paths:
            fpath = Path(fp)
            lang = language
            if lang is None:
                from src.reporag.config import settings

                lang = settings.extension_map.get(fpath.suffix.lower())
                if lang is None:
                    logger.debug("Skipping unsupported extension: %s", fpath.suffix)
                    continue
            try:
                source_bytes = fpath.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            tree = parser.parse(source_bytes, language=lang)
            str_path = str(fpath)
            file_asts[str_path] = tree
            syms = extractor.extract_from_tree(
                tree, str_path, source_bytes, language=lang
            )
            all_symbols.extend(syms)

        return self.build(all_symbols, file_asts)

    # ------------------------------------------------------------------
    # Internal: per-node edge factory
    # ------------------------------------------------------------------

    def _make_edge(
        self,
        call_node: Node,
        file_path: str,
        file_scopes: list[_ScopeRange],
        alias_map: dict[str, str],
    ) -> CallEdge | None:
        """Attempt to convert a ``call`` AST node into a :class:`CallEdge`.

        Returns ``None`` when:
        - The call has no resolvable function name.
        - The call is at module scope (no enclosing function/method).
        """
        raw_name = _extract_call_name(call_node)
        if not raw_name:
            return None

        call_line = call_node.start_point[0] + 1  # convert to 1-based

        caller = _find_caller(call_line, file_scopes)
        if caller is None:
            return None  # module-level call -- skip

        callee = _resolve_callee(raw_name, alias_map)
        raw_text = _decode(call_node)[:200]  # cap at 200 chars

        return CallEdge(
            caller=caller,
            callee=callee,
            call_site_file=file_path,
            call_site_line=call_line,
            raw_call_text=raw_text,
        )

"""Call graph builder.

Walks tree-sitter ASTs to identify function call expressions and resolves
them to target symbols. Builds directed edges: caller -> callee with call
site metadata.

Supported call forms
--------------------
* Direct function calls:         ``helper()``
* Method calls:                  ``self.method()`` / ``obj.method()``
* Chained calls:                 ``get_service().run()``
* Constructor calls:             ``MyClass()`` (resolved when local symbol is a class)
* Cross-file calls:              resolved via the import map provided by the caller

Resolution strategy
-------------------
1. Build a local symbol index from the file's own definitions.
   - A *qualified-name index* (``dict[str, Symbol]``) for exact qualified lookups.
   - A *simple-name index* (``dict[str, list[Symbol]]``) that preserves all
     symbols sharing the same bare name without overwriting.
2. Overlay an import map that maps alias -> (module, original_name) for
   names imported from other files.
3. For each ``call_expression`` / ``call`` node in the AST, extract the
   callee name and resolve it through dedicated helpers:
   - :func:`_resolve_local`  - same-file direct and constructor calls.
   - :func:`_resolve_method` - ``self.`` / ``cls.`` calls with class-aware lookup.
   - :func:`_resolve_import` - calls whose root name appears in the import map.
   - :func:`_resolve_unknown` - unresolvable calls (built-ins, third-party).

If a callee cannot be resolved (e.g. a built-in or a third-party library
call), the edge is still recorded with ``callee_file=None`` so that callers
can inspect all outgoing calls - not just in-project ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """A single directed edge in the call graph.

    Attributes:
        caller: Qualified name of the function/method that contains the call.
        callee: Qualified name of the called symbol (best-effort resolved).
        caller_file: Absolute path to the file that contains the call site.
        callee_file: Absolute path to the file that *defines* the callee,
            or ``None`` when the callee could not be resolved to a project
            file (e.g. stdlib / third-party).
        call_site_line: 1-based line number of the call expression.
        call_type: One of ``"direct"``, ``"method"``, ``"constructor"``,
            ``"chained"``, or ``"unknown"``.
        is_recursive: ``True`` when ``caller == callee`` (same qualified name).
    """

    caller: str
    callee: str
    caller_file: str
    callee_file: str | None
    call_site_line: int
    call_type: str = "direct"
    is_recursive: bool = False


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


class _ImportEntry(NamedTuple):
    """Resolved import entry."""

    module: str  # e.g. "src.reporag.ingestion.parser"
    original_name: str  # e.g. "ASTParser"  (same as alias when no alias)
    callee_file: str | None  # resolved absolute path, or None


class _LocalIndex(NamedTuple):
    """Dual-index structure for local symbol lookup.

    Attributes:
        by_qualified: Maps fully-qualified name -> Symbol (unique per qualified name).
        by_name: Maps bare name -> list[Symbol] (preserves all same-name symbols).
    """

    by_qualified: dict[str, Symbol]
    by_name: dict[str, list[Symbol]]


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------


def _build_local_index(symbols: list[Symbol]) -> _LocalIndex:
    """Build a dual local symbol index from extracted symbols.

    - ``by_qualified`` indexes each symbol by its fully-qualified name so
      that lookups like ``"MyClass.save"`` always resolve to the correct
      symbol, even when another class defines a same-named method.
    - ``by_name`` indexes each symbol by its bare (unqualified) name,
      collecting *all* symbols sharing that name into a list to avoid
      silent overwrites.

    Only ``"function"``, ``"method"``, and ``"class"`` symbols are indexed;
    imports and other symbol types are handled separately.

    Args:
        symbols: All symbols extracted from a single source file.

    Returns:
        A :class:`_LocalIndex` with both sub-indexes populated.
    """
    by_qualified: dict[str, Symbol] = {}
    by_name: dict[str, list[Symbol]] = {}

    for sym in symbols:
        if sym.type not in ("function", "method", "class"):
            continue

        # Qualified-name index (unique: qualified names must not collide)
        if sym.qualified_name:
            by_qualified[sym.qualified_name] = sym

        # Simple-name index (multi-valued: preserve duplicates)
        by_name.setdefault(sym.name, []).append(sym)

    return _LocalIndex(by_qualified=by_qualified, by_name=by_name)


def _build_import_map(
    symbols: list[Symbol],
    file_path: str,
    project_root: str | None = None,
) -> dict[str, _ImportEntry]:
    """Build alias -> _ImportEntry from import symbols.

    Only processes symbols with ``type == "import"``.

    Args:
        symbols: All symbols extracted from the file (imports included).
        file_path: Absolute path of the current file (for relative resolution).
        project_root: Optional root used to convert module paths to file paths.

    Returns:
        A mapping from the imported alias to its :class:`_ImportEntry`.
    """
    import_map: dict[str, _ImportEntry] = {}
    for sym in symbols:
        if sym.type != "import":
            continue

        alias = sym.name  # e.g. "ASTParser" or "s" (aliased)
        module = sym.import_source or ""  # e.g. "src.reporag.ingestion.parser"
        original = sym.import_alias or alias  # original symbol name in the module

        callee_file: str | None = None
        if project_root and module:
            callee_file = _module_to_file(module, project_root, file_path)

        import_map[alias] = _ImportEntry(
            module=module,
            original_name=original,
            callee_file=callee_file,
        )
    return import_map


# ---------------------------------------------------------------------------
# Module-path helpers
# ---------------------------------------------------------------------------


def _module_to_file(
    module: str,
    project_root: str,
    current_file: str,
) -> str | None:
    """Attempt to map a module string to an absolute .py file path.

    Handles absolute dotted paths (``src.reporag.ingestion.parser``) and
    simple relative markers (``.`` / ``..``).

    Args:
        module: Dotted module path, possibly starting with ``.`` for relative.
        project_root: Absolute path to the project root directory.
        current_file: Absolute path to the file containing the import.

    Returns:
        Resolved absolute file path, or ``None`` if the file does not exist.
    """
    # Relative import
    if module.startswith("."):
        levels = len(module) - len(module.lstrip("."))
        rest = module.lstrip(".")
        base = Path(current_file).parent
        for _ in range(levels - 1):
            base = base.parent
        candidate = base / rest.replace(".", "/") if rest else base
        for suffix in (".py", "/__init__.py"):
            p = Path(str(candidate) + suffix)
            if p.exists():
                return str(p)
        return None

    # Absolute dotted path
    parts = module.split(".")
    candidate = Path(project_root, *parts)
    for suffix in (".py", "/__init__.py"):
        p = Path(str(candidate) + suffix)
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# AST call-node helpers
# ---------------------------------------------------------------------------


def _decode(node: Node) -> str:
    """Decode a tree-sitter node's text safely to a UTF-8 string."""
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _extract_callee_name(call_node: Node) -> tuple[str, str]:
    """Return ``(callee_text, call_type)`` from a tree-sitter ``call`` node.

    tree-sitter Python grammar: ``call`` nodes have a ``function`` field
    which can be:

    - ``identifier``  -> direct call (e.g. ``helper()``)
    - ``attribute``   -> method / chained call (e.g. ``self.save()``)
    - ``call``        -> chained call result (e.g. ``get_x()()``)

    The ``call_type`` returned here is a *preliminary* classification.
    The final ``"constructor"`` type is determined later in
    :func:`_resolve_local` by inspecting the resolved symbol's type.

    Args:
        call_node: A tree-sitter ``call`` AST node.

    Returns:
        A 2-tuple of ``(callee_str, call_type)``.
    """
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return ("", "unknown")

    if func_node.type == "identifier":
        return (_decode(func_node), "direct")

    if func_node.type == "attribute":
        attr_node = func_node.child_by_field_name("attribute")
        obj_node = func_node.child_by_field_name("object")
        attr = _decode(attr_node) if attr_node else ""
        obj = _decode(obj_node) if obj_node else ""

        call_type = "chained" if (obj_node and obj_node.type == "call") else "method"
        return (f"{obj}.{attr}" if obj else attr, call_type)

    if func_node.type == "call":
        # Doubly-nested call: get_service()()
        inner_name, _ = _extract_callee_name(func_node)
        return (inner_name, "chained")

    # Fallback for unusual node shapes
    return (_decode(func_node), "unknown")


def _collect_call_nodes(tree: Tree) -> list[Node]:
    """Iterative DFS to collect all ``call`` nodes in the AST.

    Recurses into every child so that nested calls (inside comprehensions,
    lambdas, default arguments, etc.) are never missed.

    Args:
        tree: A parsed tree-sitter ``Tree``.

    Returns:
        All ``call`` nodes in DFS pre-order.
    """
    results: list[Node] = []
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            results.append(node)
        stack.extend(reversed(node.children))
    return results


# ---------------------------------------------------------------------------
# Enclosing-scope resolution
# ---------------------------------------------------------------------------


@dataclass
class _ScopeFrame:
    """Line-range record for a single function or method scope."""

    qualified_name: str
    start_line: int
    end_line: int
    # Qualified name of the enclosing class, if any (derived from parent_symbol).
    parent_class: str | None = None


def _build_scope_frames(symbols: list[Symbol]) -> list[_ScopeFrame]:
    """Convert the symbol list into scope frames, sorted for innermost lookup.

    Only ``"function"`` and ``"method"`` symbols produce frames.  Each frame
    also carries the ``parent_class`` derived from the symbol's
    ``parent_symbol`` field so that ``self.`` / ``cls.`` calls can be
    resolved within the correct class namespace.

    Args:
        symbols: All symbols extracted from a source file.

    Returns:
        List of :class:`_ScopeFrame` objects sorted by
        ``(start_line asc, end_line desc)`` so that innermost scopes sort
        last and can be found with a simple linear scan.
    """
    # Build a quick lookup: qualified_name -> Symbol type to identify classes
    class_names: set[str] = {
        sym.qualified_name
        for sym in symbols
        if sym.type == "class" and sym.qualified_name
    }

    frames: list[_ScopeFrame] = []
    for sym in symbols:
        if sym.type not in ("function", "method") or not sym.qualified_name:
            continue

        # parent_class is the parent_symbol when that parent is a class
        parent_class: str | None = None
        if sym.parent_symbol and sym.parent_symbol in class_names:
            parent_class = sym.parent_symbol

        frames.append(
            _ScopeFrame(
                qualified_name=sym.qualified_name,
                start_line=sym.start_line,
                end_line=sym.end_line,
                parent_class=parent_class,
            )
        )

    # Sort so that when two frames overlap, the innermost (smaller span, later
    # start) comes last in iteration and can be picked as the "best" match.
    frames.sort(key=lambda f: (f.start_line, -f.end_line))
    return frames


def _find_enclosing_scope(
    call_line: int,
    frames: list[_ScopeFrame],
    file_path: str,
) -> _ScopeFrame | None:
    """Return the innermost :class:`_ScopeFrame` that contains *call_line*.

    Args:
        call_line: 1-based line number of the call expression.
        frames: All scope frames for the file, pre-sorted.
        file_path: Used only for the fallback module-scope label.

    Returns:
        The innermost matching frame, or ``None`` for module-level calls.
    """
    best: _ScopeFrame | None = None

    for frame in frames:
        if frame.start_line <= call_line <= frame.end_line and (
            best is None
            or (frame.start_line >= best.start_line and frame.end_line <= best.end_line)
        ):
            best = frame

    return best


def _scope_label(frame: _ScopeFrame | None, file_path: str) -> str:
    """Return the caller string for an edge.

    Args:
        frame: The enclosing scope frame, or ``None`` for module-level.
        file_path: Source file path used for the module-level fallback label.

    Returns:
        Qualified function name, or ``"<module:filename>"`` at module level.
    """
    if frame is not None:
        return frame.qualified_name
    return f"<module:{Path(file_path).name}>"


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_local(
    raw_callee: str,
    local: _LocalIndex,
    current_file: str,
) -> tuple[str, str | None, str] | None:
    """Try to resolve *raw_callee* against the local symbol index.

    Handles both direct calls (``helper()``) and constructor calls (when the
    resolved local symbol has ``type == "class"``).  Constructor detection is
    based on the *actual symbol type*, not on naming conventions.

    Args:
        raw_callee: Raw callee text from the AST (bare name only, no dots).
        local: The :class:`_LocalIndex` for the current file.
        current_file: File path returned as ``callee_file`` on success.

    Returns:
        A 3-tuple ``(qualified_name, callee_file, call_type)`` if resolved,
        or ``None`` if the name is not in the local index.
    """
    # Qualified-name exact hit (e.g. "MyClass.save" from a dotted local ref)
    if raw_callee in local.by_qualified:
        sym = local.by_qualified[raw_callee]
        call_type = "constructor" if sym.type == "class" else "direct"
        return (sym.qualified_name or sym.name, current_file, call_type)

    # Simple-name hit: prefer unambiguous single match
    matches = local.by_name.get(raw_callee)
    if matches:
        # When multiple symbols share the name, pick the first (module-level)
        # as a best-effort default.  Ambiguous cases remain resolvable via
        # the qualified-name index once the caller's parent class is known.
        sym = matches[0]
        call_type = "constructor" if sym.type == "class" else "direct"
        return (sym.qualified_name or sym.name, current_file, call_type)

    return None


def _resolve_method(
    method_name: str,
    parent_class: str | None,
    local: _LocalIndex,
    current_file: str,
) -> tuple[str, str | None] | None:
    """Try to resolve a ``self.`` / ``cls.`` call against the local index.

    Resolution order:
    1. ``ParentClass.method_name`` in the qualified-name index (preferred --
       avoids ambiguity when multiple classes define the same method name).
    2. Bare ``method_name`` in the simple-name index (fallback).

    Args:
        method_name: The attribute name after the ``self.`` / ``cls.`` prefix.
        parent_class: Qualified name of the class that owns the calling method,
            or ``None`` when the caller is a module-level function.
        local: The :class:`_LocalIndex` for the current file.
        current_file: File path returned as ``callee_file`` on success.

    Returns:
        A 2-tuple ``(qualified_name, callee_file)`` if resolved, or ``None``.
    """
    # 1. Class-scoped lookup: prefer "ClassName.method_name"
    if parent_class:
        qualified_key = f"{parent_class}.{method_name}"
        if qualified_key in local.by_qualified:
            sym = local.by_qualified[qualified_key]
            return (sym.qualified_name or qualified_key, current_file)

    # 2. Bare-name fallback
    matches = local.by_name.get(method_name)
    if not matches:
        return None

    # Unambiguous single match
    if len(matches) == 1:
        sym = matches[0]
        return (sym.qualified_name or method_name, current_file)

    # Multiple methods share the same name and we cannot determine
    # which one is correct without class context.
    return None


def _resolve_import(
    raw_callee: str,
    import_map: dict[str, _ImportEntry],
) -> tuple[str, str | None] | None:
    """Try to resolve *raw_callee* via the import map.

    Checks whether the root identifier of *raw_callee* is a known import
    alias and, if so, constructs the fully-qualified callee name by
    substituting the module path.

    Args:
        raw_callee: Raw callee text, e.g. ``"os.path.join"`` or ``"compute"``.
        import_map: The import map built from :func:`_build_import_map`.

    Returns:
        A 2-tuple ``(qualified_name, callee_file)`` if the root is imported,
        or ``None`` otherwise.
    """
    root = raw_callee.split(".")[0]
    entry = import_map.get(root)
    if entry is None:
        return None

    tail = raw_callee[len(root) :]  # e.g. ".method" or ""
    qualified = f"{entry.module}.{entry.original_name}{tail}".strip(".")
    return (qualified, entry.callee_file)


def _resolve_unknown(raw_callee: str) -> tuple[str, str | None]:
    """Fallback resolver for unresolvable callees (built-ins, third-party).

    Args:
        raw_callee: The unresolved callee text.

    Returns:
        The raw callee string with ``callee_file=None``.
    """
    return (raw_callee, None)


def _resolve_callee(
    raw_callee: str,
    call_type: str,
    local: _LocalIndex,
    import_map: dict[str, _ImportEntry],
    current_file: str,
    parent_class: str | None = None,
) -> tuple[str, str | None, str]:
    """Orchestrate callee resolution across all available strategies.

    Resolution order:
    1. ``self.`` / ``cls.`` method call -> :func:`_resolve_method`.
    2. Bare or qualified local name    -> :func:`_resolve_local`.
    3. Imported name (root in map)     -> :func:`_resolve_import`.
    4. Unresolvable                    -> :func:`_resolve_unknown`.

    The ``call_type`` may be upgraded to ``"constructor"`` by
    :func:`_resolve_local` when the resolved symbol is a ``class``.

    Args:
        raw_callee: The raw callee text extracted from the AST.
        call_type: Preliminary call type from :func:`_extract_callee_name`.
        local: Local symbol index.
        import_map: Import alias map.
        current_file: Path of the file being analysed.
        parent_class: Qualified name of the caller's enclosing class, or
            ``None``.  Used only for ``self.`` / ``cls.`` resolution.

    Returns:
        A 3-tuple ``(qualified_callee, callee_file, final_call_type)``.
    """
    if not raw_callee:
        return ("", None, call_type)

    # --- 1. self. / cls. method call ---
    if call_type == "method" and raw_callee.startswith(("self.", "cls.")):
        method_name = raw_callee.split(".", 1)[1]
        result = _resolve_method(method_name, parent_class, local, current_file)
        if result is not None:
            qname, callee_file = result
            return (qname, callee_file, "method")

    # --- 2. Local symbol (direct / constructor) ---
    result_local = _resolve_local(raw_callee, local, current_file)
    if result_local is not None:
        qname, callee_file, resolved_type = result_local
        return (qname, callee_file, resolved_type)

    # --- 3. Import map ---
    result_import = _resolve_import(raw_callee, import_map)
    if result_import is not None:
        qname, callee_file = result_import
        return (qname, callee_file, call_type)

    # --- 4. Unresolved ---
    qname, callee_file = _resolve_unknown(raw_callee)
    return (qname, callee_file, call_type)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CallGraphBuilder:
    """Builds a call graph for one or more Python source files.

    Usage::

        builder = CallGraphBuilder()
        edges = builder.extract_from_source(source_code, file_path="myfile.py")

    For cross-file resolution pass ``project_root``::

        edges = builder.extract_from_file("path/to/file.py", project_root="/repo")
    """

    def __init__(self, parser: ASTParser | None = None) -> None:
        """Initialise with an optional shared :class:`ASTParser`."""
        self._parser = parser or ASTParser()
        self._extractor = SymbolExtractor(self._parser)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def extract_from_source(
        self,
        source: str | bytes,
        file_path: str = "<string>",
        language: str = "python",
        project_root: str | None = None,
    ) -> list[CallEdge]:
        """Extract call edges from raw source code.

        Args:
            source: Python source as string or bytes.
            file_path: Logical file path used in edge metadata.
            language: Language name (currently only ``"python"`` is supported).
            project_root: Optional absolute path to the repository root used
                for cross-file callee resolution.

        Returns:
            A list of :class:`CallEdge` objects, one per call site.
        """
        src_bytes = source.encode("utf-8") if isinstance(source, str) else source
        tree: Tree = self._parser.parse(src_bytes, language=language)
        symbols: list[Symbol] = self._extractor.extract_from_tree(
            tree, file_path, src_bytes, language=language
        )
        return self._build_edges(tree, symbols, file_path, project_root)

    def extract_from_file(
        self,
        file_path: str | Path,
        language: str = "python",
        project_root: str | None = None,
    ) -> list[CallEdge]:
        """Extract call edges from a file on disk.

        Args:
            file_path: Path to the source file.
            language: Language name.
            project_root: Optional repository root for cross-file resolution.

        Returns:
            A list of :class:`CallEdge` objects.
        """
        fpath = Path(file_path)
        src_bytes = fpath.read_bytes()
        return self.extract_from_source(
            src_bytes,
            file_path=str(fpath),
            language=language,
            project_root=project_root,
        )

    def build_graph(
        self,
        file_paths: list[str | Path],
        language: str = "python",
        project_root: str | None = None,
    ) -> list[CallEdge]:
        """Extract call edges from multiple files and merge into one graph.

        Args:
            file_paths: List of source file paths.
            language: Language name.
            project_root: Optional repository root.

        Returns:
            Merged list of :class:`CallEdge` objects across all files.
        """
        all_edges: list[CallEdge] = []
        for fp in file_paths:
            all_edges.extend(
                self.extract_from_file(fp, language=language, project_root=project_root)
            )
        return all_edges

    # ------------------------------------------------------------------
    # Core edge extraction logic
    # ------------------------------------------------------------------

    def _build_edges(
        self,
        tree: Tree,
        symbols: list[Symbol],
        file_path: str,
        project_root: str | None,
    ) -> list[CallEdge]:
        """Core routine: walk call nodes and emit :class:`CallEdge` objects.

        Args:
            tree: Parsed tree-sitter ``Tree`` for the file.
            symbols: All symbols extracted from the file.
            file_path: Absolute (or logical) path of the file.
            project_root: Optional project root for cross-file resolution.

        Returns:
            List of :class:`CallEdge` objects.
        """
        local = _build_local_index(symbols)
        import_map = _build_import_map(symbols, file_path, project_root)
        scope_frames = _build_scope_frames(symbols)

        edges: list[CallEdge] = []
        for call_node in _collect_call_nodes(tree):
            raw_callee, call_type = _extract_callee_name(call_node)
            if not raw_callee:
                continue

            call_line = call_node.start_point[0] + 1  # 1-based
            enclosing = _find_enclosing_scope(call_line, scope_frames, file_path)
            caller = _scope_label(enclosing, file_path)
            parent_class = enclosing.parent_class if enclosing else None

            callee_qname, callee_file, final_type = _resolve_callee(
                raw_callee,
                call_type,
                local,
                import_map,
                file_path,
                parent_class=parent_class,
            )

            # A call is recursive only when the fully-qualified caller and
            # callee are identical.
            is_recursive = bool(callee_qname) and caller == callee_qname
            edges.append(
                CallEdge(
                    caller=caller,
                    callee=callee_qname or raw_callee,
                    caller_file=file_path,
                    callee_file=callee_file,
                    call_site_line=call_line,
                    call_type=final_type,
                    is_recursive=is_recursive,
                )
            )

        return edges

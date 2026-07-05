"""Import dependency graph builder.

Builds a directed graph of module-level import relationships: each edge is
``importing_module -> imported_module``. This is the counterpart to the
call graph (Issue 9): the call graph tracks *who calls what*, this module
tracks *who imports what* -- together they form the two halves of the
project's code knowledge graph (see :mod:`src.reporag.graph.call_graph`).

Dependency on Issue 7
----------------------
This module depends on Issue 7's :mod:`~src.reporag.ingestion.symbol_extractor`
in two ways:

1. :meth:`DependencyGraphBuilder.build_from_symbols` takes Issue 7's
   ``Symbol`` objects (``type="import"``) directly as input -- the lower-
   level entry point for callers (e.g. the ingestion pipeline) that have
   already run symbol extraction and don't want to re-parse.
2. :meth:`DependencyGraphBuilder.build_from_files` /
   :meth:`~DependencyGraphBuilder.build_from_sources` construct a
   :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
   internally and use it as their source of truth for *which* symbols are
   imports; edge construction itself walks the AST once more at the
   statement level (see "Why not build_from_symbols by default" below).

Why not build_from_symbols by default: Issue 7's ``Symbol`` schema has no
field recording whether a name came from ``import x`` or
``from x import y`` -- both produce ``type="import"`` -- and for an
*aliased* from-import it folds the member name into ``import_source``
(``from os import path as p`` yields ``import_source="os.path"``, which is
byte-for-byte identical to what ``import os.path as p`` produces). That
makes it impossible to always recover the correct target module from a
``Symbol`` alone for aliased imports of unresolved (external) modules.
``build_from_symbols`` resolves this with a best-effort heuristic (see its
docstring) and is documented as such; ``build_from_files`` /
``build_from_sources`` instead walk the AST directly at the statement
level, which never has this ambiguity, so they remain the accurate default
path.

What it does:

* Walks tree-sitter ASTs directly for ``import_statement`` /
  ``import_from_statement`` nodes. Edges are grouped per *statement*, not
  per imported name: ``from x import a, b`` produces one edge to ``x``
  carrying both names, while ``import a, b`` produces two edges -- one per
  target module -- since each names a different dependency.
* Resolves absolute imports (``import a.b.c``) and relative imports
  (``from . import x``, ``from ..pkg import y``) to project files, reusing
  the dotted-module suffix index already built for the call graph
  (:class:`src.reporag.graph.call_graph._ModuleIndex`) so both graphs agree
  on how a module name maps to a file.
* Flags star imports (``from x import *``) with a warning, since the bound
  names can't be statically determined.
* Detects circular import chains via iterative DFS cycle detection over the
  resolved (file-level) import graph.

Usage::

    from src.reporag.graph.dependency_graph import DependencyGraphBuilder

    builder = DependencyGraphBuilder()
    result = builder.build_from_files([
        "examples/sample_repo/app.py",
        "examples/sample_repo/auth.py",
        "examples/sample_repo/db.py",
    ])
    for edge in result.edges:
        print(f"{edge.source} -> {edge.target} ({edge.import_type})")
    for cycle in result.cycles:
        print(cycle)

    # Or from Issue 7's already-extracted symbols (lower-level API):
    result = builder.build_from_symbols(symbols_by_file)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from tree_sitter import Node, Tree

from src.reporag.graph.call_graph import _ModuleIndex
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

logger = logging.getLogger(__name__)

ImportType = Literal["import", "from_import"]
"""Statement-level import form.

- ``"import"``      -- ``import x`` / ``import x.y`` / ``import x as y``.
- ``"from_import"`` -- ``from x import y`` (absolute or relative; see
  :attr:`DependencyEdge.is_relative`).
"""


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class DependencyEdge:
    """One directed ``source -> target`` module dependency.

    Attributes:
        source: File path of the importing module.
        target: File path of the imported module if it resolved to a
            project file, otherwise the raw dotted module name (external /
            stdlib import, or an import that could not be resolved).
        source_module: Dotted module name of ``source``.
        target_module: Dotted module name as written in the import
            statement (absolute, or with leading dots if relative).
        import_type: ``"import"`` or ``"from_import"``.
        imported_names: Names bound by the statement, as ``(name, alias)``
            pairs. Empty for a plain ``import x`` statement (the module
            itself is the bound name). ``[("*", None)]`` for a star import.
        line: 1-based line of the import statement.
        is_relative: ``True`` for ``from .`` / ``from ..`` style imports.
        relative_level: Number of leading dots (0 for absolute imports).
        is_wildcard: ``True`` for ``from x import *``.
        resolved: ``True`` if ``target`` points to a project file rather
            than a raw (unresolved) module name.
    """

    source: str
    target: str
    source_module: str
    target_module: str
    import_type: ImportType
    imported_names: list[tuple[str, str | None]] = field(default_factory=list)
    line: int = 0
    is_relative: bool = False
    relative_level: int = 0
    is_wildcard: bool = False
    resolved: bool = False


@dataclass
class CircularImportChain:
    """A cycle detected in the resolved (file-level) import graph.

    ``files`` lists the cycle in traversal order with the start file
    repeated at the end (e.g. ``["a.py", "b.py", "a.py"]``); ``modules`` is
    the same chain expressed as dotted module names.
    """

    files: list[str]
    modules: list[str]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return " -> ".join(self.modules)


@dataclass
class DependencyGraphResult:
    """The full output of a dependency-graph build."""

    edges: list[DependencyEdge]
    cycles: list[CircularImportChain] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_cycles(self) -> bool:
        """Return ``True`` if at least one circular import was detected."""
        return bool(self.cycles)


# ---------------------------------------------------------------------------
# Raw (pre-resolution) import extraction
# ---------------------------------------------------------------------------


@dataclass
class _RawImport:
    """An import statement as written, before resolving to a project file."""

    import_type: ImportType
    module: str  # dotted module text as written; leading dots if relative
    names: list[tuple[str, str | None]]
    line: int
    is_wildcard: bool = False


def _node_text(node: Node) -> str:
    """Decode a tree-sitter node's captured source text."""
    raw = node.text
    return raw.decode("utf-8", errors="replace") if raw else ""


def _extract_python_imports(tree: Tree) -> list[_RawImport]:
    """Walk *tree* (iterative DFS) collecting every import statement.

    - **Why it exists**: Mirrors the grammar handling in
      :mod:`src.reporag.ingestion.symbol_extractor`, but groups names by
      *statement* rather than emitting one entry per imported name -- a
      dependency edge is one row per target module per statement, not one
      row per bound name.
    - **Algorithm**: Iterative stack-based DFS over the tree (no recursion,
      so deeply nested source can't blow the call stack). On
      ``import_statement`` / ``import_from_statement`` nodes it stops
      descending and hands off to a dedicated per-statement parser;
      everything else is pushed onto the stack for further traversal.
    - **Edge cases**: A malformed ``from import x`` (no module name node)
      is skipped for resolution purposes but its children are still
      walked, so it can't hide a well-formed nested import.
    """
    raw: list[_RawImport] = []
    stack: list[Node] = [tree.root_node]

    while stack:
        node = stack.pop()

        if node.type == "import_statement":
            raw.extend(_parse_import_statement(node))
            continue

        if node.type == "import_from_statement":
            parsed = _parse_import_from_statement(node)
            if parsed is not None:
                raw.append(parsed)
                continue
            # No resolvable module name -- fall through to walk children
            # so we don't silently drop anything nested inside.

        stack.extend(reversed(node.children))

    return raw


def _parse_import_statement(node: Node) -> list[_RawImport]:
    """Parse a plain ``import_statement`` node into one entry per target.

    ``import a, b as c`` names two independent dependencies, so it yields
    two :class:`_RawImport` entries sharing the same line.
    """
    line = node.start_point[0] + 1
    entries: list[_RawImport] = []
    for child in node.named_children:
        if child.type == "dotted_name":
            name = _node_text(child)
            entries.append(_RawImport("import", name, [(name, None)], line))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is not None and alias_node is not None:
                src_name = _node_text(name_node)
                alias = _node_text(alias_node)
                entries.append(
                    _RawImport("import", src_name, [(src_name, alias)], line)
                )
    return entries


def _parse_import_from_statement(node: Node) -> _RawImport | None:
    """Parse an ``import_from_statement`` node into a single entry.

    Returns ``None`` if the statement has no discoverable module-name node
    (malformed source), so the caller can fall back to a generic walk.
    """
    module_node = node.child_by_field_name("module_name")
    if module_node is None:
        for child in node.named_children:
            if child.type in ("dotted_name", "relative_import"):
                module_node = child
                break
    if module_node is None:
        return None

    line = node.start_point[0] + 1
    module = _node_text(module_node)
    has_wildcard = any(c.type == "wildcard_import" for c in node.named_children)

    if has_wildcard:
        return _RawImport("from_import", module, [("*", None)], line, is_wildcard=True)

    names: list[tuple[str, str | None]] = []
    for child in node.named_children:
        # NOTE: tree-sitter Node objects are re-wrapped on each access, so
        # identity (`is`) comparison is unreliable -- compare with `==`
        # (byte-range + type equality), matching the symbol_extractor
        # convention.
        if child == module_node:
            continue
        if child.type in ("dotted_name", "identifier"):
            names.append((_node_text(child), None))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is not None and alias_node is not None:
                names.append((_node_text(name_node), _node_text(alias_node)))

    return _RawImport("from_import", module, names, line)


_IMPORT_EXTRACTORS = {
    "python": _extract_python_imports,
}


# ---------------------------------------------------------------------------
# Module-name helper (kept in sync with _ModuleIndex._parts)
# ---------------------------------------------------------------------------


def _module_name(file_path: str) -> str:
    """Return the dotted module name for *file_path*.

    Mirrors :meth:`src.reporag.graph.call_graph._ModuleIndex._parts` so the
    module names attached to :class:`DependencyEdge` and
    :class:`CircularImportChain` line up with how ``_ModuleIndex`` itself
    would name the same file.
    """
    pure = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(pure.parts[:-1]) + [pure.stem]
    if pure.stem == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p not in ("", ".", "/"))


# ---------------------------------------------------------------------------
# DependencyGraphBuilder
# ---------------------------------------------------------------------------


class DependencyGraphBuilder:
    """Builds a directed module-dependency graph from parsed source files.

    A single instance can be reused across an entire repository; it caches
    one :class:`~src.reporag.ingestion.parser.ASTParser` so grammars load
    at most once per language.

    Args:
        parser: Optional pre-built :class:`ASTParser` (inject in tests to
            avoid re-loading grammars).
        extractor: Optional pre-built
            :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
            (Issue 7). Not used by the AST-walking path itself, but shared
            so a caller building both a call graph and a dependency graph
            for the same repository pays the grammar-loading cost once.
    """

    def __init__(
        self,
        parser: ASTParser | None = None,
        extractor: SymbolExtractor | None = None,
    ) -> None:
        """Initialise the builder and its shared parser/extractor."""
        self._parser = parser if parser is not None else ASTParser()
        self._extractor = (
            extractor if extractor is not None else SymbolExtractor(self._parser)
        )

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def build_from_files(
        self,
        paths: Iterable[str | Path],
        *,
        include_external: bool = True,
    ) -> DependencyGraphResult:
        """Parse *paths* from disk and build the dependency graph.

        Files whose language has no registered import extractor are
        skipped with a debug log rather than raising, so a mixed-language
        repository is processed on a best-effort basis.

        Args:
            paths: Iterable of source file paths.
            include_external: When ``True`` (default), imports that don't
                resolve to a project file (stdlib / third-party packages)
                are still emitted as edges, with ``target`` set to the raw
                module name and ``resolved=False``. When ``False``, only
                intra-project edges are returned.

        Returns:
            A :class:`DependencyGraphResult`.
        """
        trees_by_file: dict[str, Tree] = {}
        file_list: list[str] = []

        for path in paths:
            fpath = Path(path)
            language = self._infer_language(fpath)
            if language is None or language not in _IMPORT_EXTRACTORS:
                logger.debug("Skipping %s: no import extractor for language", fpath)
                continue
            try:
                source = fpath.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            key = str(fpath)
            trees_by_file[key] = self._parser.parse(source, language=language)
            file_list.append(key)

        return self._build(trees_by_file, file_list, include_external=include_external)

    def build_from_sources(
        self,
        sources: Mapping[str, str | bytes],
        *,
        language: str = "python",
        include_external: bool = True,
    ) -> DependencyGraphResult:
        """Build the dependency graph from in-memory ``{file_path: source}``.

        Ideal for tests and for ingesting a repository already held in
        memory.

        Args:
            sources: Mapping of file path label -> source code.
            language: Language of every source (single-language batch).
            include_external: See :meth:`build_from_files`.
        """
        trees_by_file: dict[str, Tree] = {}
        for file_path, source in sources.items():
            source_bytes = source.encode("utf-8") if isinstance(source, str) else source
            trees_by_file[file_path] = self._parser.parse(
                source_bytes, language=language
            )
        return self._build(
            trees_by_file, list(sources), include_external=include_external
        )

    def build_from_trees(
        self,
        file_asts: Mapping[str, Tree],
        *,
        include_external: bool = True,
    ) -> DependencyGraphResult:
        """Build the dependency graph from already-parsed trees.

        Lower-level entry point for callers (e.g. the ingestion pipeline)
        that have already parsed every file and don't want to re-parse.
        """
        return self._build(
            dict(file_asts), list(file_asts), include_external=include_external
        )

    def build_from_symbols(
        self,
        symbols_by_file: Mapping[str, Iterable[Symbol]],
        *,
        include_external: bool = True,
    ) -> DependencyGraphResult:
        """Build the dependency graph directly from Issue 7's ``Symbol`` output.

        This is the API referenced by Issue 10's dependency on Issue 7: it
        takes the ``Symbol`` list a
        :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
        already produced for each file and turns the ``type="import"``
        symbols into :class:`DependencyEdge` objects, without re-parsing
        or re-walking any AST.

        Limitation -- read before relying on this for aliased imports:
        Issue 7's ``Symbol`` schema has no field for *which statement form*
        produced a name (``import x`` vs. ``from x import y`` both yield
        ``type="import"``), and an aliased from-import folds the member
        into ``import_source`` (``from os import path as p`` ->
        ``import_source="os.path"``) -- identical to what
        ``import os.path as p`` produces. For an aliased import this method
        therefore tries the dotted ``import_source`` as a whole module
        first (the "plain import" reading); if that doesn't resolve to a
        project file it retries with the last dot-segment split off as a
        from-import member (the "from-import" reading). If *neither*
        resolves (typically: an aliased import of an external package),
        the "plain import" reading is kept as the reported target/imported
        name, since there is no further signal to disambiguate.
        :meth:`build_from_files` / :meth:`build_from_sources` don't have
        this limitation because they read the statement kind directly off
        the AST, so prefer those when precision on aliased external
        imports matters.

        Args:
            symbols_by_file: Mapping of file path -> the ``Symbol`` list
                extracted for that file (as returned by
                ``SymbolExtractor.extract_from_file`` / ``extract_from_tree``).
            include_external: See :meth:`build_from_files`.

        Returns:
            A :class:`DependencyGraphResult`. Unlike the AST-walking path,
            edges here are merged by ``(source, target, import_type,
            is_relative, is_wildcard, resolved)`` after resolution, since
            ``Symbol`` carries no statement boundary to group by -- two
            *separate* ``from x import a`` / ``from x import b`` statements
            in the same file are therefore reported as one merged edge.
        """
        file_list = list(symbols_by_file)
        module_index = _ModuleIndex(file_list)
        raw_edges: list[DependencyEdge] = []
        warnings: list[str] = []

        for file_path, symbols in symbols_by_file.items():
            source_module = _module_name(file_path)
            for sym in symbols:
                if sym.type != "import":
                    continue
                raw_edges.extend(
                    self._edges_for_symbol(
                        sym, file_path, source_module, module_index, include_external
                    )
                )
                if sym.is_wildcard_import:
                    warnings.append(
                        f"{file_path}:{sym.start_line}: star import from "
                        f"'{sym.import_source}' -- bound names cannot be "
                        f"statically determined"
                    )

        edges = _merge_symbol_edges(raw_edges)
        cycles = _detect_cycles(edges)
        return DependencyGraphResult(edges=edges, cycles=cycles, warnings=warnings)

    @staticmethod
    def _edges_for_symbol(
        sym: Symbol,
        file_path: str,
        source_module: str,
        module_index: _ModuleIndex,
        include_external: bool,
    ) -> list[DependencyEdge]:
        """Resolve one Issue-7 ``Symbol`` into zero or more raw edges.

        See :meth:`DependencyGraphBuilder.build_from_symbols` for the
        aliased-import disambiguation heuristic used here.
        """

        def resolve(module: str) -> list[str]:
            level = len(module) - len(module.lstrip("."))
            if level > 0:
                return module_index.resolve_relative(module, file_path)
            return module_index.resolve_absolute(module)

        def emit(
            module: str,
            names: list[tuple[str, str | None]],
            import_type: ImportType,
            is_wildcard: bool,
        ) -> list[DependencyEdge]:
            level = len(module) - len(module.lstrip("."))
            is_relative = level > 0
            targets = resolve(module)
            if targets:
                return [
                    DependencyEdge(
                        source=file_path,
                        target=t,
                        source_module=source_module,
                        target_module=module,
                        import_type=import_type,
                        imported_names=list(names),
                        line=sym.start_line,
                        is_relative=is_relative,
                        relative_level=level,
                        is_wildcard=is_wildcard,
                        resolved=True,
                    )
                    for t in targets
                ]
            if include_external:
                return [
                    DependencyEdge(
                        source=file_path,
                        target=module,
                        source_module=source_module,
                        target_module=module,
                        import_type=import_type,
                        imported_names=list(names),
                        line=sym.start_line,
                        is_relative=is_relative,
                        relative_level=level,
                        is_wildcard=is_wildcard,
                        resolved=False,
                    )
                ]
            return []

        if sym.is_wildcard_import:
            module = sym.import_source or ""
            return emit(module, [("*", None)], "from_import", True)

        source = sym.import_source or sym.name

        if not sym.import_alias:
            # No alias: name == import_source for a plain `import x`
            # (possibly dotted); anything else is `from x import y`.
            if sym.name == source:
                return emit(source, [(sym.name, None)], "import", False)
            return emit(source, [(sym.name, None)], "from_import", False)

        # Aliased: ambiguous between `import x.y as z` and
        # `from x import y as z` -- try the "plain import" reading first.
        plain_edges = emit(source, [(source, sym.import_alias)], "import", False)
        if any(e.resolved for e in plain_edges):
            return plain_edges

        level = len(source) - len(source.lstrip("."))
        bare = source[level:]
        if "." in bare:
            module_part, member = bare.rsplit(".", 1)
            module_candidate = source[:level] + module_part
            from_edges = emit(
                module_candidate, [(member, sym.import_alias)], "from_import", False
            )
            if any(e.resolved for e in from_edges):
                return from_edges

        # Neither reading resolved -- keep the "plain import" guess.
        return plain_edges

    # ------------------------------------------------------------------
    # Internal build pipeline (AST-based; used by build_from_files/sources)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_language(path: Path) -> str | None:
        """Infer a language from a file extension via ``settings.extension_map``."""
        from src.reporag.config import settings  # noqa: PLC0415

        return settings.extension_map.get(path.suffix.lower())

    def _build(
        self,
        trees_by_file: Mapping[str, Tree],
        file_list: list[str],
        *,
        include_external: bool,
    ) -> DependencyGraphResult:
        """Extract, resolve, and assemble edges + cycles for *file_list*."""
        # Reuse the call graph's suffix index so both graphs resolve module
        # names to files identically.
        module_index = _ModuleIndex(file_list)
        edges: list[DependencyEdge] = []
        warnings: list[str] = []

        for file_path in file_list:
            tree = trees_by_file[file_path]
            extractor = _IMPORT_EXTRACTORS.get(self._tree_language(file_path))
            if extractor is None:
                continue

            source_module = _module_name(file_path)
            for raw in extractor(tree):
                level = len(raw.module) - len(raw.module.lstrip("."))
                is_relative = level > 0

                if raw.is_wildcard:
                    warnings.append(
                        f"{file_path}:{raw.line}: star import from "
                        f"'{raw.module}' -- bound names cannot be "
                        f"statically determined"
                    )

                # Expand "from . import service" into ".service" before resolution.
                module_to_resolve = raw.module

                # Handle:
                #     from . import service
                # by resolving ".service" instead of just "."
                if (
                    is_relative
                    and raw.module.rstrip(".") == ""
                    and len(raw.names) == 1
                    and raw.names[0][0] != "*"
                ):
                    module_to_resolve = raw.module + raw.names[0][0]

                if is_relative:
                    targets = module_index.resolve_relative(
                        module_to_resolve,
                        file_path,
                    )
                else:
                    targets = module_index.resolve_absolute(module_to_resolve)

                if targets:
                    for target_file in targets:
                        edges.append(
                            self._make_edge(
                                file_path,
                                target_file,
                                source_module,
                                module_to_resolve,
                                raw,
                                level,
                                is_relative,
                                resolved=True,
                            )
                        )
                elif include_external:
                    edges.append(
                        self._make_edge(
                            file_path,
                            module_to_resolve,
                            source_module,
                            module_to_resolve,
                            raw,
                            level,
                            is_relative,
                            resolved=False,
                        )
                    )

        cycles = _detect_cycles(edges)
        return DependencyGraphResult(edges=edges, cycles=cycles, warnings=warnings)

    @staticmethod
    def _tree_language(file_path: str) -> str:
        """Infer the language of a tree keyed by *file_path* (defaults to python)."""
        from src.reporag.config import settings  # noqa: PLC0415

        return settings.extension_map.get(Path(file_path).suffix.lower(), "python")

    @staticmethod
    def _make_edge(
        source: str,
        target: str,
        source_module: str,
        target_module: str,
        raw: _RawImport,
        level: int,
        is_relative: bool,
        *,
        resolved: bool,
    ) -> DependencyEdge:
        """Assemble a :class:`DependencyEdge` from a resolved/unresolved raw import."""
        return DependencyEdge(
            source=source,
            target=target,
            source_module=source_module,
            target_module=target_module,
            import_type=raw.import_type,
            imported_names=list(raw.names),
            line=raw.line,
            is_relative=is_relative,
            relative_level=level,
            is_wildcard=raw.is_wildcard,
            resolved=resolved,
        )


# ---------------------------------------------------------------------------
# Symbol-derived edge merging (build_from_symbols only)
# ---------------------------------------------------------------------------


def _merge_symbol_edges(edges: list[DependencyEdge]) -> list[DependencyEdge]:
    """Merge per-name edges from ``build_from_symbols`` into per-module edges.

    - **Why it exists**: Issue 7's ``Symbol`` schema has no statement
      boundary, so :meth:`DependencyGraphBuilder._edges_for_symbol` emits
      one raw edge per imported *name* -- ``from x import a, b`` yields two
      raw edges to the same target file. This merges them back into one
      edge per module, matching the AST-walking path's per-statement
      grouping.
    - **Algorithm**: Groups by ``(source, target, import_type,
      is_relative, is_wildcard, resolved)``; within a group, unions the
      ``imported_names`` lists (order-preserving, de-duplicated) and keeps
      the earliest ``line``.
    - **Edge cases**: Two genuinely separate ``from x import a`` /
      ``from x import b`` statements on different lines collapse into one
      merged edge -- an accepted precision loss versus the AST path, since
      ``Symbol`` alone can't tell them apart from one combined statement.
    """
    merged: dict[tuple[str, str, str, bool, bool, bool], DependencyEdge] = {}
    order: list[tuple[str, str, str, bool, bool, bool]] = []

    for edge in edges:
        key = (
            edge.source,
            edge.target,
            edge.import_type,
            edge.is_relative,
            edge.is_wildcard,
            edge.resolved,
        )
        if key not in merged:
            merged[key] = edge
            order.append(key)
            continue
        existing = merged[key]
        for name_pair in edge.imported_names:
            if name_pair not in existing.imported_names:
                existing.imported_names.append(name_pair)
        existing.line = min(existing.line, edge.line)

    return [merged[key] for key in order]


# ---------------------------------------------------------------------------
# Circular import detection
# ---------------------------------------------------------------------------


def _detect_cycles(edges: Iterable[DependencyEdge]) -> list[CircularImportChain]:
    """Detect cycles in the resolved (file-level) import graph.

    - **Why it exists**: Circular imports are a common source of
      ``ImportError: cannot import name`` bugs in Python; surfacing the
      full chain lets a reviewer see the loop instead of just the
      symptom.
    - **Algorithm**: Iterative DFS with white/gray/black colouring over the
      subgraph of edges that resolved to a project file -- unresolved /
      external edges can't participate in a project-internal cycle, and
      recursion is avoided so a large, deeply-connected repository can't
      blow the call stack. A back-edge to a node still coloured gray (on
      the current DFS path) is a cycle; the chain is read off the explicit
      path stack.
    - **Edge cases**: A self-import (a module importing itself, e.g. via a
      relative wildcard resolving to its own file) is reported as a
      length-1 cycle.
    - **Correctness choice**: Each distinct cycle is keyed by the frozen
      set of files involved, so the same loop discovered from different
      DFS roots -- or via multiple import statements between the same two
      files -- is reported only once.
    """
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if not edge.resolved:
            continue
        adjacency.setdefault(edge.source, [])
        if edge.target not in adjacency[edge.source]:
            adjacency[edge.source].append(edge.target)
        adjacency.setdefault(edge.target, [])

    for node in adjacency:
        adjacency[node].sort()

    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(adjacency, white)
    cycles: list[CircularImportChain] = []
    seen_keys: set[frozenset[str]] = set()

    for start in sorted(adjacency):
        if color[start] != white:
            continue

        color[start] = gray
        path: list[str] = [start]
        frames: list[tuple[str, Iterable[str]]] = [(start, iter(adjacency[start]))]

        while frames:
            node, neighbors = frames[-1]
            advanced = False

            for neighbor in neighbors:
                if color[neighbor] == white:
                    color[neighbor] = gray
                    path.append(neighbor)
                    frames.append((neighbor, iter(adjacency[neighbor])))
                    advanced = True
                    break
                if color[neighbor] == gray:
                    idx = path.index(neighbor)
                    chain_files = [*path[idx:], neighbor]
                    key = frozenset(chain_files[:-1])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        cycles.append(
                            CircularImportChain(
                                files=chain_files,
                                modules=[_module_name(f) for f in chain_files],
                            )
                        )
                # black neighbors are fully explored dead ends -- skip.

            if not advanced:
                frames.pop()
                path.pop()
                color[node] = black

    return cycles

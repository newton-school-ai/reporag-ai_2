"""Import dependency graph builder.

Builds directed edges representing module-level import relationships.
Resolves relative imports, handles star imports, and detects circular
import chains.

Each edge captures one unique module-to-module dependency::

    source_module -> target_module  (import_type, imported_names)

Multiple names imported from the same module (``from math import sin, cos``)
are **merged into a single edge** so the graph stays compact: one edge per
(source_file, target_module) pair.

Resolution is *best-effort* and *static*:

* Absolute imports (``import os``, ``from flask import Flask``) resolve to
  a project file when the module lives inside the repository, otherwise the
  edge is kept with ``resolved=False`` and ``target_file=None``.
* Relative imports (``from .sibling import helper``) are rebuilt into an
  absolute module path using the importing file's package tree, then looked
  up in the project index.
* Wildcard imports (``from X import *``) are captured with a ``"*"`` entry
  in ``imported_names`` and an ``is_wildcard`` flag on the edge.
* Circular import chains are detected by a DFS reachability check and
  flagged with ``is_circular=True`` on each involved edge.

Usage::

    from src.reporag.graph.dependency_graph import DependencyGraphBuilder

    builder = DependencyGraphBuilder()

    # From files on disk:
    edges = builder.build_from_files([
        "src/reporag/graph/call_graph.py",
        "src/reporag/ingestion/parser.py",
    ])
    for e in edges:
        print(f"{e.source_module} -> {e.target_module} ({e.import_type})")

    # From in-memory source strings:
    edges = builder.build_from_sources({"a.py": source_a, "b.py": source_b})

    # From pre-extracted Symbol objects:
    edges = builder.build_from_symbols(all_symbols)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.reporag.graph.call_graph import _ModuleIndex
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

logger = logging.getLogger(__name__)

ImportType = Literal["import", "from_import", "wildcard"]
"""Discriminator for how a module dependency was expressed.

- ``"import"``      -- plain ``import X`` or ``import X as Y``.
- ``"from_import"`` -- ``from M import X`` or ``from M import X as Y``.
- ``"wildcard"``    -- ``from M import *``.
"""


# ---------------------------------------------------------------------------
# DependencyEdge dataclass
# ---------------------------------------------------------------------------


@dataclass
class DependencyEdge:
    """A directed module-level import edge: source_module -> target_module.

    Attributes:
        source_module:  Dotted module name of the importing file
                        (e.g. ``"reporag.graph.dependency_graph"``).
        target_module:  Dotted module name of the imported module, resolved
                        to an absolute path where possible
                        (e.g. ``"reporag.ingestion.parser"``).
                        For unresolved external imports this is the raw
                        module name as written in the source.
        source_file:    Absolute or relative path of the importing file.
        import_type:    How the dependency was expressed (see :data:`ImportType`).
        imported_names: Names pulled from the target module.  Empty list for
                        plain ``import X`` edges; ``["*"]`` for wildcard
                        imports; otherwise the list of imported identifiers.
        target_file:    Path of the project file that defines the imported
                        module, or ``None`` when the module is external /
                        third-party or could not be located.
        resolved:       ``True`` when ``target_file`` is not ``None``
                        (derived automatically in :meth:`__post_init__`).
        is_circular:    Whether this edge belongs to a circular import
                        chain (A -> B -> ... -> A).  Set post-build by
                        :meth:`DependencyGraphBuilder._mark_circular`.
    """

    source_module: str
    target_module: str
    source_file: str
    import_type: ImportType
    imported_names: list[str] = field(default_factory=list)
    target_file: str | None = None
    resolved: bool = False
    is_circular: bool = False

    def __post_init__(self) -> None:
        """Derive ``resolved`` from ``target_file`` so they never disagree."""
        self.resolved = self.target_file is not None

    @property
    def is_wildcard(self) -> bool:
        """True if this edge represents a wildcard import (from X import *)."""
        return self.import_type == "wildcard"

    @property
    def has_warning(self) -> bool:
        """True if this dependency has an anti-pattern warning (like a wildcard)."""
        return self.is_wildcard

    def __repr__(self) -> str:
        circ = " [circular]" if self.is_circular else ""
        return (
            f"DependencyEdge({self.source_module} -> {self.target_module}"
            f" [{self.import_type}]{circ})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this edge.

        All values are JSON-primitive types (``str``, ``bool``, ``None``,
        or ``list[str]``).  Suitable for a Neo4j ``IMPORTS`` relationship
        payload (Issue 12) or for writing to JSONL for offline analysis.
        """
        return {
            "source_module": self.source_module,
            "target_module": self.target_module,
            "source_file": self.source_file,
            "target_file": self.target_file,
            "import_type": self.import_type,
            "imported_names": list(self.imported_names),
            "resolved": self.resolved,
            "is_circular": self.is_circular,
            "is_wildcard": self.is_wildcard,
            "has_warning": self.has_warning,
        }


# ---------------------------------------------------------------------------
# Public coordinator: DependencyGraphBuilder
# ---------------------------------------------------------------------------


class DependencyGraphBuilder:
    """Builds a directed import dependency graph from parsed source files.

    A single instance can be reused across repositories.  It caches one
    :class:`~src.reporag.ingestion.parser.ASTParser` (and a
    :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`) so
    grammars load at most once per language.

    Args:
        parser:    Optional pre-built :class:`ASTParser` (inject in tests to
                   avoid re-loading grammars).
        extractor: Optional pre-built :class:`SymbolExtractor`.
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
    ) -> list[DependencyEdge]:
        """Parse *paths* from disk, extract symbols, and build the dependency graph.

        Files whose extension has no registered language are skipped with a
        debug log rather than raising, so mixed-language repositories are
        processed on a best-effort basis.

        Args:
            paths:            Iterable of source file paths.
            include_external: When ``True`` (default), emit edges whose target
                              module could not be resolved to a project file
                              (builtins, third-party packages).  Set to
                              ``False`` to keep only intra-project edges.

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}

        for path in paths:
            fpath = Path(path)
            language = self._infer_language(fpath)
            if language is None:
                logger.debug("Skipping %s: unsupported extension", fpath)
                continue
            try:
                source = fpath.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            key = str(fpath)
            tree = self._parser.parse(source, language=language)
            symbols_by_file[key] = self._extractor.extract_from_tree(
                tree, key, source, language=language
            )

        return self._build(symbols_by_file, include_external=include_external)

    def build_from_sources(
        self,
        sources: Mapping[str, str | bytes],
        *,
        language: str = "python",
        include_external: bool = True,
    ) -> list[DependencyEdge]:
        """Build the dependency graph from in-memory ``{file_path: source}`` mappings.

        Ideal for tests and for ingesting a repository already held in memory.

        Args:
            sources:          Mapping of file path label -> source code.
            language:         Language of every source (single-language batch).
            include_external: See :meth:`build_from_files`.

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}

        for file_path, source in sources.items():
            source_bytes = source.encode("utf-8") if isinstance(source, str) else source
            tree = self._parser.parse(source_bytes, language=language)
            symbols_by_file[file_path] = self._extractor.extract_from_tree(
                tree, file_path, source_bytes, language=language
            )

        return self._build(symbols_by_file, include_external=include_external)

    def build_from_symbols(
        self,
        symbols: Iterable[Symbol],
        *,
        include_external: bool = True,
    ) -> list[DependencyEdge]:
        """Build the dependency graph from pre-extracted :class:`Symbol` objects.

        This is the lower-level API for use when the ingestion pipeline has
        already extracted symbols (so no re-parsing is needed).

        Args:
            symbols:          Flat iterable of :class:`Symbol` objects across
                              all files (each carries its ``file_path``).
            include_external: See :meth:`build_from_files`.

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = defaultdict(list)
        for sym in symbols:
            symbols_by_file[sym.file_path].append(sym)

        return self._build(dict(symbols_by_file), include_external=include_external)

    # ------------------------------------------------------------------
    # Core build
    # ------------------------------------------------------------------

    def _build(
        self,
        symbols_by_file: Mapping[str, list[Symbol]],
        *,
        include_external: bool,
    ) -> list[DependencyEdge]:
        """Resolve every import symbol across all files into :class:`DependencyEdge` objects.

        - **Why it exists**: Central pass that ties together import extraction
          and cross-file module resolution.
        - **Algorithm**:
          1. Build a global :class:`~src.reporag.graph.call_graph._ModuleIndex`
             from all known file paths so relative and absolute imports can
             both be resolved to project files.
          2. For each file, group its import symbols by target module key and
             emit one :class:`DependencyEdge` per unique (source, target) pair.
          3. Run a DFS reachability check to flag circular import chains.
          4. Sort edges deterministically by ``(source_file, target_module)``.
        - **Edge cases**: Files with zero imports contribute no edges.
          Wildcard imports are retained even when ``include_external=False``
          because the source module *is* a project file.
        - **Correctness choice**: One edge per (source_file, target_module)
          pair keeps the graph compact.  ``imported_names`` aggregates all
          names pulled from that module so no information is lost.
        """
        module_index = _ModuleIndex(symbols_by_file.keys())
        edges: list[DependencyEdge] = []

        for file_path, syms in symbols_by_file.items():
            source_module = _file_to_module(file_path)
            file_edges = self._build_file_edges(
                file_path, source_module, syms, module_index
            )
            for edge in file_edges:
                if not edge.resolved and not include_external:
                    continue
                edges.append(edge)

        self._mark_circular(edges)
        edges.sort(key=lambda e: (e.source_file, e.target_module))
        return edges

    def _build_file_edges(
        self,
        file_path: str,
        source_module: str,
        symbols: list[Symbol],
        module_index: _ModuleIndex,
    ) -> list[DependencyEdge]:
        """Produce one :class:`DependencyEdge` per unique target module for *file_path*.

        Groups import symbols by their target module key so that
        ``from math import sin`` and ``from math import cos`` in the same file
        produce a single edge to ``math`` with ``imported_names=["cos", "sin"]``.
        """
        # Group import symbols by the module they target.
        groups: dict[str, list[Symbol]] = defaultdict(list)
        for sym in symbols:
            if sym.type != "import":
                continue
            key = self._module_key(sym)
            if not key:
                continue
            groups[key].append(sym)

        edges: list[DependencyEdge] = []
        for module_key, group in groups.items():
            edge = self._make_edge(
                file_path, source_module, module_key, group, module_index
            )
            if edge is not None:
                edges.append(edge)
        return edges

    def _make_edge(
        self,
        source_file: str,
        source_module: str,
        module_key: str,
        group: list[Symbol],
        module_index: _ModuleIndex,
    ) -> DependencyEdge | None:
        """Build one :class:`DependencyEdge` for a group of same-module imports.

        Resolution strategy
        -------------------
        1. Try to resolve *module_key* directly via :class:`_ModuleIndex`.
        2. If that fails and *module_key* looks like a ``"module.member"`` path
           (e.g. ``from math import sin as s`` stores source as ``"math.sin"``),
           retry with just the parent component (``"math"``).
        3. If still unresolved the edge is external: ``target_file=None``.

        The ``target_module`` on the edge is derived from the resolved file's
        dotted module path when available, falling back to *module_key* for
        external imports.
        """
        import_type = self._import_type(group)
        imported_names = self._imported_names(import_type, group)

        # --- Resolution step 1: resolve module_key directly ---
        target_files = module_index.resolve(module_key, source_file)

        # --- Resolution step 2: retry with parent module (aliased from-import) ---
        if not target_files and "." in module_key and not module_key.startswith("."):
            parent = module_key.rsplit(".", 1)[0]
            if parent:
                target_files = module_index.resolve(parent, source_file)

        target_file: str | None = target_files[0] if target_files else None
        target_module = (
            _file_to_module(target_file) if target_file else module_key.lstrip(".")
        )

        return DependencyEdge(
            source_module=source_module,
            target_module=target_module,
            source_file=source_file,
            import_type=import_type,
            imported_names=imported_names,
            target_file=target_file,
        )

    # ------------------------------------------------------------------
    # Circular import detection
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_circular(edges: list[DependencyEdge]) -> None:
        """Flag edges that are part of a circular import chain.

        - **Why it exists**: Circular imports cause subtle bugs in Python and
          are important to surface in the dependency graph.
        - **Algorithm**: Builds an adjacency map (``source_file -> {target_files}``),
          then uses Tarjan's Strongly Connected Components (SCC) algorithm. Any
          edge where both endpoints belong to the same SCC (of size > 1) is
          part of a cycle and is flagged. This runs in O(V + E) time.
        - **Edge cases**: Self-imports and external edges (``target_file=None``)
          are skipped.
        - **Correctness choice**: An edge is flagged only when a cycle is
          *actually* reachable (sound detection), never by heuristic.
        """
        # Build adjacency: source_file -> set of target_files
        adj: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            if e.target_file and e.source_file != e.target_file:
                adj[e.source_file].add(e.target_file)

        index = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        scc_map: dict[str, int] = {}
        scc_count = 0

        def strongconnect(v: str) -> None:
            nonlocal index, scc_count
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                scc_count += 1
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc_map[w] = scc_count
                    if w == v:
                        break

        for node in adj:
            if node not in indices:
                strongconnect(node)

        for edge in edges:
            if (
                edge.target_file is not None
                and edge.source_file != edge.target_file
                and scc_map.get(edge.source_file) == scc_map.get(edge.target_file)
                and edge.source_file in scc_map
            ):
                edge.is_circular = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _module_key(sym: Symbol) -> str:
        """Extract the target *module* path from an import :class:`Symbol`.

        The goal is to return the dotted name of the **module** being depended
        on -- not the member being imported.  This is used as the grouping key
        so that ``from math import sin`` and ``from math import cos`` both
        produce the key ``"math"`` and are merged into one edge.

        Rules
        -----
        - Wildcard ``from X import *``   -> ``import_source`` (``"X"``)
        - Plain ``import X``             -> ``import_source`` (same as name)
        - ``import X as Y``              -> ``import_source`` (``"X"``)
        - ``from M import X``            -> ``import_source`` (``"M"``)
        - ``from M import X as Y``       -> strip last component of
          ``import_source`` (extractor stores ``"M.X"`` -> returns ``"M"``)
        - Relative ``from .M import X``  -> ``import_source`` (``".M"``)
        - Relative ``from . import X``   -> ``"." + sym.name``
          (X is itself a submodule in the current package)

        .. note::
            ``import X.Y as Z`` is indistinguishable from
            ``from X import Y as Z`` in Symbol alone (both yield
            ``import_source="X.Y"``).  The heuristic strips the last
            component, which is correct for ``from X import Y as Z`` but
            loses ``".Y"`` precision for ``import X.Y as Z`` (very rare).
        """
        if sym.is_wildcard_import:
            return sym.import_source or ""

        source = sym.import_source or ""

        # "import X" or "import X.Y": name and source are identical
        if sym.name == source and not sym.import_alias:
            return source

        # "import X as Y": alias set, source is the pure module name (no member)
        if sym.import_alias and not source.endswith("." + sym.import_alias):
            # Distinguish "import X as Y" (source="X") from
            # "from M import X as Y" (source="M.X")
            # Heuristic: strip last component if source contains "."
            if "." in source and not source.startswith("."):
                return source.rsplit(".", 1)[0]
            return source

        # "from . import X" or "from .. import X": source is just dots, name is the submodule
        if source and set(source) == {"."} and not sym.import_alias:
            return f"{source}{sym.name}"

        # "from M import X" or "from .M import X": source is the module
        return source

    @staticmethod
    def _import_type(group: list[Symbol]) -> ImportType:
        """Determine the :data:`ImportType` for a group of same-module imports.

        Precedence: wildcard > plain import > from-import.
        """
        if any(s.is_wildcard_import for s in group):
            return "wildcard"
        # Plain "import X" or "import X as Y": name matches source directly
        # OR name is the alias and source is a simple (non-dotted) module name
        if any(
            (s.name == (s.import_source or "") and not s.import_alias)
            or (s.import_alias and "." not in (s.import_source or ""))
            for s in group
        ):
            return "import"
        return "from_import"

    @staticmethod
    def _imported_names(import_type: ImportType, group: list[Symbol]) -> list[str]:
        """Build the ``imported_names`` list for an edge.

        - Wildcard  -> ``["*"]``
        - Plain import -> ``[]`` (the module itself is the dependency)
        - From-import -> sorted list of names being imported
        """
        if import_type == "wildcard":
            logger.warning(
                "Wildcard import detected in %s from %s",
                group[0].file_path if group else "unknown",
                group[0].import_source if group else "unknown",
            )
            return ["*"]
        if import_type == "import":
            return []
        # from_import: collect all non-"*" names
        names = []
        for s in group:
            if s.name == "*":
                continue
            if s.import_alias and s.import_source and "." in s.import_source:
                # Extract the original name from the source (e.g., "math.sin" -> "sin")
                names.append(s.import_source.rsplit(".", 1)[-1])
            else:
                names.append(s.name)
        return sorted(set(names))

    @staticmethod
    def _infer_language(path: Path) -> str | None:
        """Infer a language from a file extension via ``settings.extension_map``."""
        from src.reporag.config import settings

        return settings.extension_map.get(path.suffix.lower())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _file_to_module(file_path: str) -> str:
    """Convert a file path to a dotted module name using :class:`_ModuleIndex` parts.

    Examples::

        "src/reporag/graph/call_graph.py"  ->  "src.reporag.graph.call_graph"
        "app.py"                            ->  "app"
        "pkg/__init__.py"                   ->  "pkg"
    """
    parts = _ModuleIndex._parts(file_path)
    return ".".join(parts) if parts else file_path

"""Import dependency graph builder.

Walks tree-sitter ASTs (or already-extracted symbols) to find every import
statement and resolves each one to a directed ``source_module -> target_module``
edge with import metadata (import kind, imported names, call-site line).

The import graph is the second half of the code knowledge graph (the call graph
from Issue 9 being the first).  Together they answer questions pure vector
search cannot: "which modules depend on this one?", "what is the import chain
from A to B?", and "does this repository have circular imports?".

What the resolver handles:

* **Absolute imports** -- ``import os`` / ``from flask import Flask`` map to the
  named module (resolved to a project file when one matches, else flagged
  external).
* **Relative imports** -- ``from .utils import helper`` / ``from ..pkg import x``
  are rebuilt into an absolute dotted module path against the importing file's
  package, so the edge target is a stable, canonical module name.
* **Submodule from-imports** -- ``from . import sibling`` and ``from pkg import
  submod`` create an edge to the *submodule* when the imported name resolves to
  a project file, and to the containing module otherwise.
* **Wildcard imports** -- ``from x import *`` is captured and flagged
  (``is_wildcard`` / ``import_type == "wildcard"``) so downstream consumers can
  warn about the unclear surface it introduces.
* **Circular imports** -- strongly connected components over the project-internal
  edges surface every cycle group; participating edges are marked ``in_cycle``.

Module names are *canonical*: a file's module name is the full dotted path
derived from its location (``examples/sample_repo/auth.py`` ->
``examples.sample_repo.auth``), and a resolved edge target uses the same
canonical form.  This is what makes cross-file linkage and cycle detection line
up regardless of whether the import was written bare (``import auth``) or fully
qualified.

Usage::

    from src.reporag.graph.dependency_graph import DependencyGraphBuilder

    builder = DependencyGraphBuilder()

    # From files on disk (parses the AST directly -- highest fidelity):
    edges = builder.build_from_files([
        "examples/sample_repo/app.py",
        "examples/sample_repo/auth.py",
        "examples/sample_repo/db.py",
    ])
    for e in edges:
        print(f"{e.source_module} -> {e.target_module} ({e.import_type})")

    cycles = DependencyGraphBuilder.detect_cycles(edges)

    # Or from already-extracted symbols (the API referenced in the issue):
    edges = builder.build(symbols_by_file)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tree_sitter import Node, Tree

from src.reporag.graph._modules import ModuleIndex
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public discriminators
# ---------------------------------------------------------------------------

ImportType = Literal["import", "from", "relative", "wildcard"]
"""Syntactic category of an import edge.

- ``"import"``   -- a plain module import (``import os``, ``import os.path``,
  ``import numpy as np``).
- ``"from"``     -- an absolute from-import (``from flask import Flask``).
- ``"relative"`` -- a relative from-import (``from .utils import helper``,
  ``from ..pkg import x``).
- ``"wildcard"`` -- a star import (``from x import *``); takes precedence over
  the other labels because the unclear surface it introduces is the most
  important fact to flag.

The label is a single primary category for display; the orthogonal
``is_relative`` and ``is_wildcard`` booleans on :class:`DependencyEdge` preserve
the full picture (e.g. a relative wildcard ``from .pkg import *`` is
``import_type == "wildcard"`` with ``is_relative is True``).
"""


# ---------------------------------------------------------------------------
# DependencyEdge dataclass
# ---------------------------------------------------------------------------


@dataclass
class DependencyEdge:
    """A directed ``source_module -> target_module`` import edge.

    Attributes:
        source_module:  Canonical dotted module name of the importing file
                        (e.g. ``"examples.sample_repo.app"``).
        target_module:  Canonical dotted module name of the imported module.
                        For a resolved project import this matches the target
                        file's own ``source_module``; for an external import it
                        is the module as written (relative imports are still
                        rendered as an absolute dotted path).
        import_type:    Syntactic category of the import (see :data:`ImportType`).
        imported_names: The specific names pulled from ``target_module`` in a
                        from-import (``["Flask"]``), ``["*"]`` for a wildcard,
                        and empty for a whole-module ``import x``.
        source_file:    Path to the importing file.
        target_file:    Path to the resolved project file, or ``None`` when the
                        target is external / unresolved.
        line:           1-based line of the import statement.
        is_relative:    ``True`` for a relative import (``from . import x``).
        is_wildcard:    ``True`` for a star import.
        resolved:       ``True`` when ``target_file`` is not ``None``; mirror
                        derived in :meth:`__post_init__`.
        is_external:    ``True`` when the target is not a project module; the
                        inverse of ``resolved``, derived in :meth:`__post_init__`.
        in_cycle:       ``True`` when this edge participates in a circular import
                        (both endpoints lie in the same cyclic component).
    """

    source_module: str
    target_module: str
    import_type: ImportType
    imported_names: list[str] = field(default_factory=list)
    source_file: str = ""
    target_file: str | None = None
    line: int = 0
    is_relative: bool = False
    is_wildcard: bool = False
    resolved: bool = False
    is_external: bool = True
    in_cycle: bool = False

    def __post_init__(self) -> None:
        """Keep ``resolved`` / ``is_external`` consistent with ``target_file``."""
        self.resolved = self.target_file is not None
        self.is_external = not self.resolved

    def __repr__(self) -> str:
        star = " *" if self.is_wildcard else ""
        cyc = " (cycle)" if self.in_cycle else ""
        return (
            f"DependencyEdge({self.source_module} -> {self.target_module}"
            f"{star} [{self.import_type}] @ {self.source_file}:{self.line}{cyc})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this edge.

        Suitable for a Neo4j ``IMPORTS`` relationship payload (Issue 12) or for
        writing to JSONL.  Every value is a JSON-primitive type (``str``,
        ``int``, ``bool``, ``None``) or a list of strings.
        """
        return {
            "source_module": self.source_module,
            "target_module": self.target_module,
            "import_type": self.import_type,
            "imported_names": list(self.imported_names),
            "source_file": self.source_file,
            "target_file": self.target_file,
            "line": self.line,
            "is_relative": self.is_relative,
            "is_wildcard": self.is_wildcard,
            "resolved": self.resolved,
            "is_external": self.is_external,
            "in_cycle": self.in_cycle,
        }


# ---------------------------------------------------------------------------
# Internal: a raw import discovered before module resolution
# ---------------------------------------------------------------------------


@dataclass
class _RawImport:
    """A single import statement discovered in source, prior to resolution.

    Attributes:
        module:         The module string exactly as written -- absolute
                        (``"os"``, ``"a.b"``) or relative (``"."``, ``".utils"``,
                        ``"..pkg"``).
        imported_names: Names introduced by a from-import; ``["*"]`` for a
                        wildcard; empty for a plain ``import x``.
        is_wildcard:    ``True`` for ``from module import *``.
        is_from:        ``True`` for any ``from ... import`` (including relative
                        and wildcard); ``False`` for plain ``import x``.
        line:           1-based line of the statement.
    """

    module: str
    imported_names: list[str]
    is_wildcard: bool
    is_from: bool
    line: int

    @property
    def is_relative(self) -> bool:
        """``True`` when the module is written as a relative import."""
        return self.module.startswith(".")


# ---------------------------------------------------------------------------
# Import discovery: from a parsed AST (faithful)
# ---------------------------------------------------------------------------


def _extract_python_imports(tree: Tree) -> list[_RawImport]:
    """Find every import statement in a Python tree-sitter *tree*.

    - **Why it exists**: Parsing the AST directly captures the exact syntactic
      form of each import -- crucially distinguishing ``import a.b as c`` from
      ``from a import b as c`` (which the shared :class:`Symbol` model encodes
      identically).  This keeps ``import_type`` exact on the source-based paths.
    - **Algorithm**: Iterative pre-order DFS.  ``import_statement`` and
      ``import_from_statement`` nodes are parsed in place; their children are not
      re-pushed (imports never nest).  Everything else is traversed so imports
      inside function bodies are captured too.
    - **Edge cases**: Wildcards, relative dots, aliases, and multi-name imports
      are all handled.  A malformed import that yields no module name is skipped.
    - **Correctness choice**: Iterative traversal avoids Python recursion limits
      on deeply nested source.
    """
    raws: list[_RawImport] = []
    stack: list[Node] = [tree.root_node]

    while stack:
        node = stack.pop()
        if node.type == "import_statement":
            for child in node.named_children:
                raw = _parse_plain_import(child)
                if raw is not None:
                    raws.append(raw)
            continue
        if node.type == "import_from_statement":
            raw = _parse_from_import(node)
            if raw is not None:
                raws.append(raw)
            continue
        stack.extend(reversed(node.children))

    return raws


def _parse_plain_import(child: Node) -> _RawImport | None:
    """Parse one entry of an ``import a, b as c`` statement into a raw import."""
    if child.type == "dotted_name":
        return _RawImport(
            module=_node_text(child),
            imported_names=[],
            is_wildcard=False,
            is_from=False,
            line=child.start_point[0] + 1,
        )
    if child.type == "aliased_import":
        name_node = child.child_by_field_name("name")
        if name_node is None:
            return None
        return _RawImport(
            module=_node_text(name_node),
            imported_names=[],
            is_wildcard=False,
            is_from=False,
            line=child.start_point[0] + 1,
        )
    return None


def _parse_from_import(node: Node) -> _RawImport | None:
    """Parse a ``from module import ...`` statement into a single raw import."""
    module_node = node.child_by_field_name("module_name")
    if module_node is None:
        for child in node.named_children:
            if child.type in ("dotted_name", "relative_import"):
                module_node = child
                break
    if module_node is None:
        return None

    module = _node_text(module_node)
    line = node.start_point[0] + 1

    if any(c.type == "wildcard_import" for c in node.named_children):
        return _RawImport(
            module=module,
            imported_names=["*"],
            is_wildcard=True,
            is_from=True,
            line=line,
        )

    names: list[str] = []
    for child in node.named_children:
        # tree-sitter yields a fresh Node wrapper per access, so identity
        # (``is``) never matches; compare the stable node id instead.
        if child.id == module_node.id:
            continue
        name = _imported_name(child)
        if name:
            names.append(name)

    return _RawImport(
        module=module,
        imported_names=names,
        is_wildcard=False,
        is_from=True,
        line=line,
    )


def _imported_name(child: Node) -> str | None:
    """Return the imported name from a from-import child (ignoring any alias)."""
    if child.type in ("dotted_name", "identifier"):
        return _node_text(child)
    if child.type == "aliased_import":
        name_node = child.child_by_field_name("name")
        if name_node is not None:
            return _node_text(name_node)
    return None


def _node_text(node: Node) -> str:
    """Decode a node's source bytes to UTF-8 text."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


# Language-specific import finders. Add new languages here alongside a
# tree-sitter grammar in the parser registry.
_IMPORT_FINDERS = {
    "python": _extract_python_imports,
}


# ---------------------------------------------------------------------------
# Import discovery: from already-extracted symbols (issue API)
# ---------------------------------------------------------------------------


def _raw_imports_from_symbols(symbols: Iterable[Symbol]) -> list[_RawImport]:
    """Reconstruct raw imports from extracted import :class:`Symbol` objects.

    - **Why it exists**: Backs the issue's ``build(symbols_by_file)`` API so the
      dependency graph slots into a pipeline that has already run the symbol
      extractor, without re-parsing.
    - **Algorithm**: Inverts the extractor's encoding -- ``from M import Y`` is
      stored as ``name="Y", source="M"``; ``from M import Y as B`` collapses the
      source to ``"M.Y"``; a plain ``import X`` stores ``name == source``.
    - **Edge cases / known limit**: ``import a.b as c`` and ``from a import b as
      c`` produce *identical* symbols (``name="c", source="a.b", alias="c"``), so
      the symbol path cannot tell them apart and treats the aliased-dotted form
      as a from-import.  The source-based entry points
      (:meth:`~DependencyGraphBuilder.build_from_files` /
      :meth:`~DependencyGraphBuilder.build_from_sources`) parse the AST and
      classify these exactly -- prefer them when that fidelity matters.
    """
    raws: list[_RawImport] = []
    for sym in symbols:
        if sym.type != "import":
            continue
        source = sym.import_source or ""
        if not source:
            continue
        line = sym.start_line

        if sym.is_wildcard_import:
            raws.append(_RawImport(source, ["*"], True, True, line))
            continue

        if sym.import_alias:
            if "." in source:
                module, member = source.rsplit(".", 1)
                raws.append(_RawImport(module, [member], False, True, line))
            else:
                raws.append(_RawImport(source, [], False, False, line))
            continue

        if source.startswith("."):
            raws.append(_RawImport(source, [sym.name], False, True, line))
        elif sym.name == source:
            raws.append(_RawImport(source, [], False, False, line))
        else:
            raws.append(_RawImport(source, [sym.name], False, True, line))

    return raws


# ---------------------------------------------------------------------------
# Strongly connected components (circular-import detection)
# ---------------------------------------------------------------------------


def _internal_adjacency(edges: Iterable[DependencyEdge]) -> dict[str, list[str]]:
    """Build the project-internal module adjacency (resolved edges only).

    External edges are excluded -- their targets' imports are unknown, so they
    can never close a project cycle.  Every endpoint is registered as a node so
    sinks participate in the SCC pass.
    """
    adj: dict[str, set[str]] = {}
    for edge in edges:
        if edge.is_external:
            continue
        adj.setdefault(edge.source_module, set()).add(edge.target_module)
        adj.setdefault(edge.target_module, set())
    return {node: sorted(targets) for node, targets in adj.items()}


def _tarjan_scc(adjacency: Mapping[str, list[str]]) -> list[list[str]]:
    """Return the strongly connected components of *adjacency*.

    Iterative Tarjan (explicit work stack) so very long import chains cannot
    exhaust the Python recursion limit.  Nodes are visited in sorted order so
    the component list is deterministic.
    """
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    scc_stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in sorted(adjacency):
        if root in index_of:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_i = work[-1]
            if child_i == 0:
                index_of[node] = counter
                lowlink[node] = counter
                counter += 1
                scc_stack.append(node)
                on_stack.add(node)

            children = adjacency.get(node, [])
            recursed = False
            for j in range(child_i, len(children)):
                nxt = children[j]
                if nxt not in index_of:
                    work[-1] = (node, j + 1)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[nxt])
            if recursed:
                continue

            if lowlink[node] == index_of[node]:
                component: list[str] = []
                while True:
                    popped = scc_stack.pop()
                    on_stack.discard(popped)
                    component.append(popped)
                    if popped == node:
                        break
                components.append(component)

            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    return components


def _scc_partition(
    edges: Iterable[DependencyEdge],
) -> tuple[dict[str, int], set[int], list[list[str]]]:
    """Partition modules into SCCs and identify which components are cyclic.

    Returns ``(scc_of, cyclic_ids, components)`` where a component is cyclic when
    it has more than one member or a single member with a self-import.
    """
    adjacency = _internal_adjacency(edges)
    components = _tarjan_scc(adjacency)

    scc_of: dict[str, int] = {}
    cyclic_ids: set[int] = set()
    for comp_id, members in enumerate(components):
        for module in members:
            scc_of[module] = comp_id
        is_self_loop = len(members) == 1 and members[0] in adjacency.get(members[0], [])
        if len(members) > 1 or is_self_loop:
            cyclic_ids.add(comp_id)
    return scc_of, cyclic_ids, components


def _ordered_cycle(members: list[str], adjacency: Mapping[str, list[str]]) -> list[str]:
    """Return one ordered cycle path (``a -> b -> c -> a``) through an SCC.

    - **Why it exists**: An SCC's member *set* proves modules are mutually
      dependent; an ordered path shows the actual import chain, which reads far
      better in a "circular import" report.
    - **Algorithm**: Iterative DFS confined to the component, starting from its
      lowest-named module.  The first edge back to a node already on the current
      path closes the cycle; the slice from that node to the end (plus the node
      again) is the chain.
    - **Correctness choice**: Because the input is a genuine SCC, a cycle through
      the start node is guaranteed, so the search always succeeds; the sorted
      fallback is defensive only.
    """
    member_set = set(members)
    start = min(members)
    path: list[str] = [start]
    position: dict[str, int] = {start: 0}
    visited: set[str] = {start}
    stack: list[tuple[str, int]] = [(start, 0)]

    while stack:
        node, child_i = stack[-1]
        children = [c for c in adjacency.get(node, []) if c in member_set]
        advanced = False
        for j in range(child_i, len(children)):
            nxt = children[j]
            if nxt in position:  # back edge onto the current path -> cycle
                return path[position[nxt] :] + [nxt]
            if nxt in visited:
                continue
            stack[-1] = (node, j + 1)
            visited.add(nxt)
            position[nxt] = len(path)
            path.append(nxt)
            stack.append((nxt, 0))
            advanced = True
            break
        if not advanced:
            stack.pop()
            dropped = path.pop()
            del position[dropped]

    return sorted(members)


# ---------------------------------------------------------------------------
# Public coordinator
# ---------------------------------------------------------------------------


class DependencyGraphBuilder:
    """Builds a directed import dependency graph from source files or symbols.

    A single instance can be reused across repositories; it caches one
    :class:`~src.reporag.ingestion.parser.ASTParser` (and a
    :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`) so grammars
    load at most once per language.

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

    def build(
        self, symbols_by_file: Mapping[str, Iterable[Symbol]]
    ) -> list[DependencyEdge]:
        """Build the dependency graph from pre-extracted symbols.

        This is the API referenced in the issue.  Use it when the ingestion
        pipeline has already run the symbol extractor.  See
        :func:`_raw_imports_from_symbols` for the one aliased-dotted import case
        the shared symbol model cannot disambiguate; the source-based entry
        points classify it exactly.

        Args:
            symbols_by_file: Mapping of ``file_path -> symbols`` (each file's
                extracted :class:`Symbol` list).

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        raw_by_file = {
            file_path: _raw_imports_from_symbols(symbols)
            for file_path, symbols in symbols_by_file.items()
        }
        return self._build(raw_by_file)

    def build_from_symbols(self, symbols: Iterable[Symbol]) -> list[DependencyEdge]:
        """Build from a flat iterable of symbols (grouped by ``file_path``).

        Convenience wrapper over :meth:`build` for callers holding one flat
        symbol list across all files.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}
        for sym in symbols:
            symbols_by_file.setdefault(sym.file_path, []).append(sym)
        return self.build(symbols_by_file)

    def build_from_files(self, paths: Iterable[str | Path]) -> list[DependencyEdge]:
        """Parse *paths* from disk and build the dependency graph.

        Files whose language has no registered import finder are skipped with a
        debug log rather than raising, so a mixed-language repository is
        processed on a best-effort basis.

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        raw_by_file: dict[str, list[_RawImport]] = {}
        for path in paths:
            fpath = Path(path)
            language = self._infer_language(fpath)
            finder = _IMPORT_FINDERS.get(language) if language else None
            if finder is None:
                logger.debug("Skipping %s: no import finder for language", fpath)
                continue
            try:
                source = fpath.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            tree = self._parser.parse(source, language=language)
            raw_by_file[str(fpath)] = finder(tree)
        return self._build(raw_by_file)

    def build_from_sources(
        self,
        sources: Mapping[str, str | bytes],
        *,
        language: str = "python",
    ) -> list[DependencyEdge]:
        """Build the dependency graph from in-memory ``{file_path: source}``.

        Ideal for tests and for ingesting a repository already held in memory.

        Returns:
            Deterministically ordered list of :class:`DependencyEdge` objects.
        """
        finder = _IMPORT_FINDERS.get(language)
        if finder is None:
            raise ValueError(f"No import finder registered for language '{language}'.")

        raw_by_file: dict[str, list[_RawImport]] = {}
        for file_path, source in sources.items():
            source_bytes = source.encode("utf-8") if isinstance(source, str) else source
            tree = self._parser.parse(source_bytes, language=language)
            raw_by_file[file_path] = finder(tree)
        return self._build(raw_by_file)

    # ------------------------------------------------------------------
    # Core build
    # ------------------------------------------------------------------

    def _build(
        self, raw_by_file: Mapping[str, list[_RawImport]]
    ) -> list[DependencyEdge]:
        """Resolve every raw import across all files into edges, and flag cycles.

        - **Why it exists**: Central pass tying import discovery, module
          resolution, and cycle detection together.
        - **Algorithm**: Builds one shared :class:`ModuleIndex` over all files,
          turns each raw import into one or more resolved edges, then runs a
          strongly connected component pass to mark edges that sit inside a cycle.
        - **Correctness choice**: Edges are sorted by
          ``(source_file, line, target_module, import_type)`` so output is
          deterministic and diff-friendly for tests and snapshots.
        """
        index = ModuleIndex(raw_by_file.keys())
        edges: list[DependencyEdge] = []

        for file_path, raws in raw_by_file.items():
            source_module = index.module_name(file_path)
            for raw in raws:
                edges.extend(
                    self._edges_for_import(raw, file_path, source_module, index)
                )

        self._mark_cycles(edges)
        edges.sort(
            key=lambda e: (e.source_file, e.line, e.target_module, e.import_type)
        )
        return edges

    def _edges_for_import(
        self,
        raw: _RawImport,
        source_file: str,
        source_module: str,
        index: ModuleIndex,
    ) -> list[DependencyEdge]:
        """Resolve one raw import into one or more :class:`DependencyEdge` objects.

        A plain or wildcard import yields a single edge to the named module.  A
        from-import may fan out: each imported name that resolves to a project
        *submodule* becomes its own edge, while the remaining names collapse into
        one edge to the containing module.
        """
        absolute = index.normalize(raw.module, source_file)
        import_type = self._import_type(raw)

        # Plain `import x` / wildcard `from x import *`: one edge to the module.
        if not raw.is_from or raw.is_wildcard:
            return [
                self._make_edge(
                    raw,
                    source_file,
                    source_module,
                    absolute,
                    raw.imported_names,
                    import_type,
                    index,
                )
            ]

        # from-import: split submodule targets from plain member names.
        submodule_names: list[str] = []
        member_names: list[str] = []
        for name in raw.imported_names:
            candidate = f"{absolute}.{name}" if absolute else name
            if index.resolve_file(candidate, source_file) is not None:
                submodule_names.append(name)
            else:
                member_names.append(name)

        edges: list[DependencyEdge] = []
        for name in submodule_names:
            candidate = f"{absolute}.{name}" if absolute else name
            edges.append(
                self._make_edge(
                    raw,
                    source_file,
                    source_module,
                    candidate,
                    [name],
                    import_type,
                    index,
                )
            )
        # Emit the containing-module edge for the remaining names, or when the
        # from-import had no names at all (defensive) or none were submodules.
        if member_names or not submodule_names:
            edges.append(
                self._make_edge(
                    raw,
                    source_file,
                    source_module,
                    absolute,
                    member_names,
                    import_type,
                    index,
                )
            )
        return edges

    def _make_edge(
        self,
        raw: _RawImport,
        source_file: str,
        source_module: str,
        absolute_module: str,
        imported_names: list[str],
        import_type: ImportType,
        index: ModuleIndex,
    ) -> DependencyEdge:
        """Construct a single edge, resolving *absolute_module* to a project file."""
        target_file = index.resolve_file(absolute_module, source_file)
        target_module = (
            index.module_name(target_file)
            if target_file is not None
            else absolute_module
        )
        return DependencyEdge(
            source_module=source_module,
            target_module=target_module,
            import_type=import_type,
            imported_names=list(imported_names),
            source_file=source_file,
            target_file=target_file,
            line=raw.line,
            is_relative=raw.is_relative,
            is_wildcard=raw.is_wildcard,
        )

    @staticmethod
    def _import_type(raw: _RawImport) -> ImportType:
        """Classify a raw import into its primary :data:`ImportType` label."""
        if raw.is_wildcard:
            return "wildcard"
        if raw.is_relative:
            return "relative"
        if raw.is_from:
            return "from"
        return "import"

    @staticmethod
    def _mark_cycles(edges: list[DependencyEdge]) -> None:
        """Set ``in_cycle`` on every edge inside a strongly connected cycle."""
        scc_of, cyclic_ids, _ = _scc_partition(edges)
        for edge in edges:
            if edge.is_external:
                continue
            comp = scc_of.get(edge.source_module)
            if (
                comp is not None
                and comp in cyclic_ids
                and comp == scc_of.get(edge.target_module)
            ):
                edge.in_cycle = True

    # ------------------------------------------------------------------
    # Cycle reporting
    # ------------------------------------------------------------------

    @staticmethod
    def detect_cycles(edges: Iterable[DependencyEdge]) -> list[list[str]]:
        """Return every circular-import group among *edges*.

        Each group is a sorted list of canonical module names that mutually
        depend on one another (a strongly connected component of size > 1, or a
        single module that imports itself).  This is the complete membership
        view; :meth:`find_cycle_paths` gives an ordered import chain per group.
        The list of groups is sorted for determinism.  An acyclic dependency
        graph returns ``[]``.
        """
        _, cyclic_ids, components = _scc_partition(edges)
        cycles = [sorted(components[i]) for i in cyclic_ids]
        cycles.sort()
        return cycles

    @staticmethod
    def find_cycle_paths(edges: Iterable[DependencyEdge]) -> list[list[str]]:
        """Return one ordered import chain per circular-import group.

        Where :meth:`detect_cycles` reports the *set* of modules in each cycle,
        this returns a representative ordered path that closes on itself -- e.g.
        ``["a", "b", "c", "a"]`` for ``a -> b -> c -> a`` -- which is the more
        readable form for logs and diagnostics.  Paths are sorted for
        determinism.  An acyclic dependency graph returns ``[]``.
        """
        edges = list(edges)
        adjacency = _internal_adjacency(edges)
        _, cyclic_ids, components = _scc_partition(edges)
        paths = [_ordered_cycle(components[i], adjacency) for i in cyclic_ids]
        paths.sort()
        return paths

    # ------------------------------------------------------------------
    # Language inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_language(path: Path) -> str | None:
        """Infer a language from a file extension via ``settings.extension_map``."""
        from src.reporag.config import settings

        return settings.extension_map.get(path.suffix.lower())

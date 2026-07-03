"""Call graph builder.

Walks tree-sitter ASTs to find function-call expressions and resolves each
one to the symbol it targets, producing directed ``caller -> callee`` edges
with call-site metadata (file + line).

The call graph answers questions pure vector search cannot: "how does login
work end-to-end?" and "what breaks if I change ``authenticate_user``?".  It is
one half of the code knowledge graph (the other half being the import
dependency graph from Issue 10).

Resolution is *static* and best-effort -- Python's dynamic dispatch and
monkey-patching mean a perfect call graph is undecidable without running the
code.  The resolver handles the cases that carry the most signal:

* **Direct calls** -- ``factorial(n - 1)`` resolves to a same-file definition
  (this also captures recursion, where ``caller == callee``).
* **Self / class methods** -- ``self.method()`` / ``cls.method()`` resolve
  against the enclosing class.
* **Cross-file calls via imports** -- ``from db import get_user`` then
  ``get_user(...)`` resolves through the import to ``db.py``.
* **Constructor calls** -- ``User(...)`` resolves to a class and is tagged
  ``constructor``.
* **Attribute calls on modules** -- ``import db`` then ``db.get_user()``
  resolves through the module import.
* **Attribute calls on values** -- ``obj.method()`` cannot be typed statically,
  so it is resolved only when exactly one project method shares that name
  (an *inferred* edge, flagged as such so downstream consumers can weigh it).

Everything the resolver cannot pin to a project symbol (builtins, third-party
calls such as ``hashlib.sha256``) is dropped by default and can be surfaced via
``include_unresolved=True``.

Usage::

    from src.reporag.graph.call_graph import CallGraphBuilder

    builder = CallGraphBuilder()

    # From files on disk (parses + extracts symbols internally):
    edges = builder.build_from_files([
        "examples/sample_repo/app.py",
        "examples/sample_repo/auth.py",
        "examples/sample_repo/db.py",
    ])
    for e in edges:
        print(f"{e.caller} -> {e.callee} (line {e.call_site_line})")

    # Or from already-extracted symbols + parsed trees (the lower-level API
    # referenced in the issue):
    edges = builder.build_from_symbols(symbols, file_asts)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

logger = logging.getLogger(__name__)

# Sentinel caller for calls that sit at module scope (not inside any
# function, method, or class body) -- e.g. ``settings = get_settings()``.
MODULE_SCOPE = "<module>"

# Receiver identifiers that refer to the enclosing instance / class.
_SELF_RECEIVERS = frozenset({"self", "cls"})

# Symbol types that define a callable/instantiable entity (a potential caller
# scope or callee target).
_DEFINITION_TYPES = frozenset({"function", "method", "class"})


# ---------------------------------------------------------------------------
# Public discriminators
# ---------------------------------------------------------------------------

CallType = Literal["function", "method", "constructor"]
"""Semantic category of a resolved call.

- ``"function"``    -- a plain function call (``foo()``).
- ``"method"``      -- an attribute-style call (``self.foo()``, ``obj.foo()``).
- ``"constructor"`` -- the callee resolved to a *class* (``User()``); takes
  precedence over ``"method"``/``"function"`` because instantiation is the
  more meaningful fact for the graph.
"""

Resolution = Literal["local", "imported", "inferred", "unresolved"]
"""How the callee symbol was located.

- ``"local"``      -- a definition in the *same file* (includes recursion and
  ``self.method`` resolved within the enclosing class).
- ``"imported"``   -- reached through an import binding into another file.
- ``"inferred"``   -- best-effort ``obj.method`` match (exactly one project
  method with that name); may be a false positive, so it is flagged.
- ``"unresolved"`` -- no project symbol matched (builtin / third-party /
  dynamic).  Emitted only when ``include_unresolved=True``.
"""


# ---------------------------------------------------------------------------
# CallEdge dataclass
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """A directed ``caller -> callee`` edge with call-site metadata.

    Attributes:
        caller:         Fully qualified name of the calling function/method
                        (e.g. ``"handle_login"`` or ``"Calculator.add"``), or
                        :data:`MODULE_SCOPE` for a module-level call.
        callee:         Fully qualified name of the resolved target symbol, or
                        -- for unresolved edges -- the raw callee name as
                        written at the call site.
        caller_file:    Path to the file containing the call site.
        callee_file:    Path to the file defining the callee, or ``None`` when
                        unresolved.
        call_site_line: 1-based line number of the call expression.
        call_type:      Semantic category of the call (see :data:`CallType`).
        resolution:     How the callee was located (see :data:`Resolution`).
        resolved:       ``True`` when ``resolution != "unresolved"``; convenience
                        mirror derived in :meth:`__post_init__`.
        is_recursive:   ``True`` when the call targets its own caller
                        (``callee == caller`` in the same file).
    """

    caller: str
    callee: str
    caller_file: str
    call_site_line: int
    callee_file: str | None = None
    call_type: CallType = "function"
    resolution: Resolution = "unresolved"
    resolved: bool = False
    is_recursive: bool = False

    def __post_init__(self) -> None:
        """Derive ``resolved`` from ``resolution`` so the two never disagree."""
        self.resolved = self.resolution != "unresolved"

    def __repr__(self) -> str:
        rec = " (recursive)" if self.is_recursive else ""
        return (
            f"CallEdge({self.caller} -> {self.callee} "
            f"[{self.call_type}/{self.resolution}] "
            f"@ {self.caller_file}:{self.call_site_line}{rec})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this edge.

        Suitable for a Neo4j ``CALLS`` relationship payload (Issue 12) or for
        writing to JSONL for offline analysis.  Every value is a JSON-primitive
        type (``str``, ``int``, ``bool``, ``None``).
        """
        return {
            "caller": self.caller,
            "callee": self.callee,
            "caller_file": self.caller_file,
            "callee_file": self.callee_file,
            "call_site_line": self.call_site_line,
            "call_type": self.call_type,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "is_recursive": self.is_recursive,
        }


# ---------------------------------------------------------------------------
# Internal: a raw call site before resolution
# ---------------------------------------------------------------------------


@dataclass
class _RawCall:
    """A call expression discovered in an AST, prior to symbol resolution.

    Attributes:
        line:              1-based line of the call's function expression.
        callee_name:       The bare function name or the attribute (method)
                           name being invoked.
        receiver:          For attribute calls with a simple identifier
                           receiver (``obj.foo()`` -> ``"obj"``), the receiver
                           name; ``None`` for bare calls or complex receivers.
        is_attribute:      ``True`` for attribute-style calls (``x.foo()``).
        receiver_is_self:  ``True`` when the receiver is ``self`` or ``cls``.
    """

    line: int
    callee_name: str
    receiver: str | None
    is_attribute: bool
    receiver_is_self: bool


# ---------------------------------------------------------------------------
# Module index: dotted module name <-> file path resolution
# ---------------------------------------------------------------------------


class _ModuleIndex:
    """Resolves import module names to project file paths.

    - **Why it exists**: Cross-file call resolution hinges on mapping an import
      like ``from db import get_user`` (module ``"db"``) to the file that
      defines ``db`` -- here ``examples/sample_repo/db.py``.
    - **Algorithm**: For every file it precomputes all dotted *suffixes* of the
      path (``examples.sample_repo.db``, ``sample_repo.db``, ``db``) and maps
      each to the file.  A bare ``import db`` matches the shortest suffix; a
      fully qualified ``import examples.sample_repo.db`` matches the longest.
    - **Edge cases**: ``__init__.py`` collapses to its package directory so a
      package import resolves to the package.  Relative imports (``.``, ``..``)
      are rebuilt into an absolute dotted path against the importing file.
    - **Correctness choice**: Uses POSIX path semantics regardless of host OS so
      module names are stable across platforms.
    """

    def __init__(self, files: Iterable[str]) -> None:
        """Build the suffix map from the set of project *files*."""
        self._suffix_map: dict[str, list[str]] = {}
        for file_path in files:
            for candidate in self._module_candidates(file_path):
                bucket = self._suffix_map.setdefault(candidate, [])
                if file_path not in bucket:
                    bucket.append(file_path)

    @staticmethod
    def _parts(file_path: str) -> list[str]:
        """Return the dotted-module path components for *file_path*.

        Strips the file extension and folds ``__init__`` into its package
        directory (``pkg/__init__.py`` -> ``["pkg"]``).
        """
        pure = PurePosixPath(file_path.replace("\\", "/"))
        parts = list(pure.parts[:-1]) + [pure.stem]
        if pure.stem == "__init__":
            parts = parts[:-1]
        return [p for p in parts if p not in ("", ".", "/")]

    def _module_candidates(self, file_path: str) -> list[str]:
        """Return every dotted suffix of *file_path*'s module path."""
        parts = self._parts(file_path)
        return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]

    def resolve_absolute(self, module: str) -> list[str]:
        """Return files matching an absolute dotted *module* name."""
        return list(self._suffix_map.get(module, []))

    def resolve_relative(self, module: str, importing_file: str) -> list[str]:
        """Resolve a relative import (``.``, ``..pkg``) to project files.

        - **Algorithm**: Counts leading dots to find how many package levels to
          ascend from *importing_file*'s package, appends the remaining dotted
          remainder, then looks the rebuilt absolute name up in the suffix map.
        - **Edge cases**: A bare ``from . import x`` (empty remainder) resolves
          to the importing file's own package directory.
        """
        level = len(module) - len(module.lstrip("."))
        remainder = module[level:]

        base = self._parts(importing_file)[:-1]  # importing module's package
        ascend = level - 1  # a single dot means "this package"
        if ascend > 0:
            base = base[:-ascend] if ascend <= len(base) else []

        target = base + (remainder.split(".") if remainder else [])
        if not target:
            return []
        return self.resolve_absolute(".".join(target))

    def resolve(self, module: str, importing_file: str) -> list[str]:
        """Resolve *module* (absolute or relative) to candidate files."""
        if module.startswith("."):
            return self.resolve_relative(module, importing_file)
        return self.resolve_absolute(module)


# ---------------------------------------------------------------------------
# Enclosing-symbol index: which definition owns a given line
# ---------------------------------------------------------------------------


@dataclass
class _Enclosure:
    """The caller and enclosing class for a call site at a given line."""

    caller: str
    caller_file: str | None
    enclosing_class: str | None


class _EnclosingIndex:
    """Maps a line number to the innermost definition that contains it.

    - **Why it exists**: Every call site must be attributed to a *caller* (the
      function/method it sits inside) and, for ``self.method`` resolution, the
      enclosing *class*.
    - **Algorithm**: Holds the ``(start_line, end_line)`` span of each
      function/method/class symbol in a file.  For a query line it selects the
      deepest (largest ``start_line``) callable span as the caller and the
      deepest class span as the enclosing class.
    - **Edge cases**: A line inside no definition (module-level code) yields
      :data:`MODULE_SCOPE` and no enclosing class.  A call inside a class body
      but outside any method (rare) is attributed to the class.
    - **Correctness choice**: "Deepest span wins" correctly attributes calls in
      nested functions/classes to the innermost scope.
    """

    def __init__(self, file_path: str, symbols: Iterable[Symbol]) -> None:
        """Index the definition spans for one file."""
        self.file_path = file_path
        self._callables: list[tuple[int, int, str]] = []
        self._classes: list[tuple[int, int, str]] = []
        for sym in symbols:
            if sym.type not in _DEFINITION_TYPES:
                continue
            qname = sym.qualified_name or sym.name
            span = (sym.start_line, sym.end_line, qname)
            if sym.type == "class":
                self._classes.append(span)
            else:
                self._callables.append(span)

    @staticmethod
    def _deepest(spans: list[tuple[int, int, str]], line: int) -> str | None:
        """Return the qualified name of the deepest span containing *line*."""
        best: tuple[int, int, str] | None = None
        for start, end, qname in spans:
            if start <= line <= end and (best is None or start > best[0]):
                best = (start, end, qname)
        return best[2] if best else None

    def enclosure(self, line: int) -> _Enclosure:
        """Return the :class:`_Enclosure` for a call site at *line*."""
        enclosing_class = self._deepest(self._classes, line)
        caller = self._deepest(self._callables, line)
        if caller is None:
            # No function/method owns the line; fall back to the class body,
            # then to module scope.
            caller = enclosing_class or MODULE_SCOPE
            caller_file = self.file_path if enclosing_class else None
            return _Enclosure(caller, caller_file, enclosing_class)
        return _Enclosure(caller, self.file_path, enclosing_class)


# ---------------------------------------------------------------------------
# Resolution context: the global view over all files
# ---------------------------------------------------------------------------


class _ResolutionContext:
    """Holds every index needed to resolve a call to a project symbol.

    Built once per :meth:`CallGraphBuilder.build` invocation from the symbols
    of *all* files, so cross-file resolution has a complete picture.
    """

    def __init__(self, symbols_by_file: Mapping[str, list[Symbol]]) -> None:
        """Index all symbols across the repository."""
        self.module_index = _ModuleIndex(symbols_by_file.keys())

        # file -> {name -> module-level def/class Symbol}
        self._module_level: dict[str, dict[str, Symbol]] = {}
        # file -> {name -> [any-scope def/class Symbols]}
        self._local_defs: dict[str, dict[str, list[Symbol]]] = {}
        # file -> {qualified_name -> Symbol}
        self._by_qualified: dict[str, dict[str, Symbol]] = {}
        # file -> {local binding name -> import Symbol}
        self._imports: dict[str, dict[str, Symbol]] = {}
        # file -> [module dotted names imported via ``from X import *``]
        self._wildcards: dict[str, list[str]] = {}
        # method name -> [Symbol] across the whole project
        self._methods_by_name: dict[str, list[Symbol]] = {}
        # file -> _EnclosingIndex
        self._enclosing: dict[str, _EnclosingIndex] = {}

        for file_path, symbols in symbols_by_file.items():
            self._index_file(file_path, symbols)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _index_file(self, file_path: str, symbols: list[Symbol]) -> None:
        """Populate every per-file index for a single file."""
        module_level: dict[str, Symbol] = {}
        local_defs: dict[str, list[Symbol]] = {}
        by_qualified: dict[str, Symbol] = {}
        imports: dict[str, Symbol] = {}
        wildcards: list[str] = []

        for sym in symbols:
            if sym.type == "import":
                self._index_import(sym, imports, wildcards)
                continue
            if sym.type not in _DEFINITION_TYPES:
                continue

            qname = sym.qualified_name or sym.name
            by_qualified[qname] = sym
            local_defs.setdefault(sym.name, []).append(sym)
            if sym.parent_symbol is None:
                module_level[sym.name] = sym
            if sym.type == "method":
                self._methods_by_name.setdefault(sym.name, []).append(sym)

        self._module_level[file_path] = module_level
        self._local_defs[file_path] = local_defs
        self._by_qualified[file_path] = by_qualified
        self._imports[file_path] = imports
        self._wildcards[file_path] = wildcards
        self._enclosing[file_path] = _EnclosingIndex(file_path, symbols)

    @staticmethod
    def _index_import(
        sym: Symbol, imports: dict[str, Symbol], wildcards: list[str]
    ) -> None:
        """Record one import symbol in the per-file import table."""
        if sym.is_wildcard_import:
            if sym.import_source:
                wildcards.append(sym.import_source)
            return
        # Last binding of a name wins (matches Python rebinding semantics).
        imports[sym.name] = sym

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def enclosure(self, file_path: str, line: int) -> _Enclosure:
        """Return the caller / enclosing class for a call at *line*."""
        index = self._enclosing.get(file_path)
        if index is None:
            return _Enclosure(MODULE_SCOPE, None, None)
        return index.enclosure(line)

    # ------------------------------------------------------------------
    # Bare-call resolution: ``name()``
    # ------------------------------------------------------------------

    def resolve_bare(
        self, name: str, file_path: str
    ) -> tuple[Symbol, Resolution] | None:
        """Resolve a bare ``name()`` call to a project symbol.

        Resolution order: same-file module-level definition (covers recursion
        and same-file constructors) -> a unique same-file nested definition ->
        an import binding into another file -> a wildcard-imported name.
        """
        # 1. Same-file module-level definition (functions, classes).
        local = self._module_level.get(file_path, {}).get(name)
        if local is not None:
            return local, "local"

        # 2. A unique same-file definition at any scope (e.g. a nested helper).
        candidates = self._local_defs.get(file_path, {}).get(name, [])
        if len(candidates) == 1:
            return candidates[0], "local"

        # 3. An import binding: ``from module import name``.
        imp = self._imports.get(file_path, {}).get(name)
        if imp is not None:
            target = self._resolve_member_import(imp, file_path)
            if target is not None:
                return target, "imported"

        # 4. A name pulled in via ``from module import *``.
        wildcard = self._resolve_wildcard(name, file_path)
        if wildcard is not None:
            return wildcard, "imported"

        return None

    def _resolve_member_import(self, imp: Symbol, file_path: str) -> Symbol | None:
        """Resolve ``from module import member`` to the member's definition."""
        module, member = self._member_target(imp)
        for target_file in self.module_index.resolve(module, file_path):
            sym = self._module_level.get(target_file, {}).get(member)
            if sym is not None:
                return sym
        return None

    @staticmethod
    def _member_target(imp: Symbol) -> tuple[str, str]:
        """Split an import symbol into ``(module, member_name)``.

        Handles the extractor's encodings:
        ``from M import Y`` -> ``(M, Y)``; ``from M import Y as B`` stores the
        source as ``"M.Y"`` -> split back to ``(M, Y)``.
        """
        source = imp.import_source or ""
        if imp.import_alias and "." in source:
            module, member = source.rsplit(".", 1)
            return module, member
        return source, imp.name

    def _resolve_wildcard(self, name: str, file_path: str) -> Symbol | None:
        """Resolve *name* against any ``from module import *`` in the file."""
        for module in self._wildcards.get(file_path, []):
            for target_file in self.module_index.resolve(module, file_path):
                sym = self._module_level.get(target_file, {}).get(name)
                if sym is not None:
                    return sym
        return None

    # ------------------------------------------------------------------
    # Attribute-call resolution: ``receiver.name()``
    # ------------------------------------------------------------------

    def resolve_attribute(
        self,
        name: str,
        receiver: str | None,
        receiver_is_self: bool,
        file_path: str,
        enclosing_class: str | None,
    ) -> tuple[Symbol, Resolution] | None:
        """Resolve an attribute call ``receiver.name()`` to a project symbol.

        Resolution order: ``self``/``cls`` against the enclosing class ->
        ``module.name`` through a module import -> a unique project method by
        name (inferred).
        """
        # 1. self.method / cls.method -> the enclosing class's method.
        if receiver_is_self and enclosing_class is not None:
            sym = self._by_qualified.get(file_path, {}).get(f"{enclosing_class}.{name}")
            if sym is not None:
                return sym, "local"

        # 2. module.func -> ``import module`` then ``module.func()``.
        if receiver is not None and not receiver_is_self:
            imp = self._imports.get(file_path, {}).get(receiver)
            if imp is not None:
                sym = self._resolve_module_attr(imp, name, file_path)
                if sym is not None:
                    return sym, "imported"

        # 3. obj.method -> exactly one project method with that name.
        methods = self._methods_by_name.get(name, [])
        if len(methods) == 1:
            return methods[0], "inferred"

        return None

    def _resolve_module_attr(
        self, imp: Symbol, name: str, file_path: str
    ) -> Symbol | None:
        """Resolve ``module.name`` where ``module`` is an imported module."""
        module = self._module_of_import(imp)
        for target_file in self.module_index.resolve(module, file_path):
            sym = self._module_level.get(target_file, {}).get(name)
            if sym is not None:
                return sym
        return None

    @staticmethod
    def _module_of_import(imp: Symbol) -> str:
        """Return the module a name refers to when used as an attribute base.

        ``import x`` / ``import x.y`` -> the source; ``import x as a`` -> the
        source; ``from p import sub`` (used as ``sub.foo()``) -> ``p.sub``.
        """
        source = imp.import_source or ""
        if imp.import_alias:
            return source
        if imp.name == source:
            return source
        # ``from p import sub`` used as a module attribute -> submodule p.sub.
        return f"{source}.{imp.name}" if source else imp.name


# ---------------------------------------------------------------------------
# Python call-site finder
# ---------------------------------------------------------------------------


def _find_python_call_sites(tree: Tree) -> list[_RawCall]:
    """Find every call expression in a Python tree-sitter *tree*.

    - **Why it exists**: Isolates the language-specific step (locating and
      classifying ``call`` nodes) from the language-agnostic resolution step.
    - **Algorithm**: Iterative pre-order DFS over the whole tree (so nested
      calls in arguments, ``f(g())``, are all captured).  Each ``call`` node's
      ``function`` field is classified as a bare ``identifier`` or an
      ``attribute`` access.
    - **Edge cases**: Complex callees -- chained calls ``a().b()`` or subscript
      calls ``handlers[k]()`` -- yield a raw call only when a method name is
      recoverable; otherwise they are skipped (nothing to resolve).
    - **Correctness choice**: Iterative traversal avoids Python recursion limits
      on deeply nested source, mirroring :meth:`ASTParser.walk`.
    """
    calls: list[_RawCall] = []
    stack: list[Node] = [tree.root_node]

    while stack:
        node = stack.pop()
        if node.type == "call":
            raw = _classify_python_call(node)
            if raw is not None:
                calls.append(raw)
        stack.extend(reversed(node.children))

    return calls


def _classify_python_call(node: Node) -> _RawCall | None:
    """Classify a single ``call`` node into a :class:`_RawCall`.

    Returns ``None`` when the callee is too dynamic to name (e.g. the result of
    another call or a subscript expression).
    """
    func = node.child_by_field_name("function")
    if func is None:
        return None

    if func.type == "identifier":
        name = _node_text(func)
        return _RawCall(
            line=func.start_point[0] + 1,
            callee_name=name,
            receiver=None,
            is_attribute=False,
            receiver_is_self=False,
        )

    if func.type == "attribute":
        attr = func.child_by_field_name("attribute")
        obj = func.child_by_field_name("object")
        if attr is None:
            return None
        name = _node_text(attr)
        receiver: str | None = None
        receiver_is_self = False
        if obj is not None and obj.type == "identifier":
            receiver = _node_text(obj)
            receiver_is_self = receiver in _SELF_RECEIVERS
        return _RawCall(
            line=attr.start_point[0] + 1,
            callee_name=name,
            receiver=receiver,
            is_attribute=True,
            receiver_is_self=receiver_is_self,
        )

    return None


def _node_text(node: Node) -> str:
    """Decode a node's source bytes to UTF-8 text."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


# Language-specific call-site finders. Add new languages here alongside a
# tree-sitter grammar in the parser registry.
_CALL_SITE_FINDERS = {
    "python": _find_python_call_sites,
}


# ---------------------------------------------------------------------------
# Public coordinator
# ---------------------------------------------------------------------------


class CallGraphBuilder:
    """Builds a directed call graph from parsed source files.

    A single instance can be reused across repositories; it caches one
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
        include_unresolved: bool = False,
    ) -> list[CallEdge]:
        """Parse *paths* from disk, extract symbols, and build the call graph.

        Files whose language has no registered call-site finder are skipped
        with a debug log rather than raising, so a mixed-language repository is
        processed on a best-effort basis.

        Args:
            paths:              Iterable of source file paths.
            include_unresolved: When ``True``, also emit edges whose callee
                                could not be resolved to a project symbol.

        Returns:
            Deterministically ordered list of :class:`CallEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}
        trees_by_file: dict[str, Tree] = {}

        for path in paths:
            fpath = Path(path)
            language = self._infer_language(fpath)
            if language is None or language not in _CALL_SITE_FINDERS:
                logger.debug("Skipping %s: no call-site finder for language", fpath)
                continue
            try:
                source = fpath.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            tree = self._parser.parse(source, language=language)
            key = str(fpath)
            trees_by_file[key] = tree
            symbols_by_file[key] = self._extractor.extract_from_tree(
                tree, key, source, language=language
            )

        return self._build(
            symbols_by_file, trees_by_file, include_unresolved=include_unresolved
        )

    def build_from_sources(
        self,
        sources: Mapping[str, str | bytes],
        *,
        language: str = "python",
        include_unresolved: bool = False,
    ) -> list[CallEdge]:
        """Build the call graph from in-memory ``{file_path: source}`` mappings.

        Ideal for tests and for ingesting a repository already held in memory.

        Args:
            sources:            Mapping of file path label -> source code.
            language:           Language of every source (single-language batch).
            include_unresolved: See :meth:`build_from_files`.

        Returns:
            Deterministically ordered list of :class:`CallEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}
        trees_by_file: dict[str, Tree] = {}

        for file_path, source in sources.items():
            source_bytes = source.encode("utf-8") if isinstance(source, str) else source
            tree = self._parser.parse(source_bytes, language=language)
            trees_by_file[file_path] = tree
            symbols_by_file[file_path] = self._extractor.extract_from_tree(
                tree, file_path, source_bytes, language=language
            )

        return self._build(
            symbols_by_file, trees_by_file, include_unresolved=include_unresolved
        )

    def build_from_symbols(
        self,
        symbols: Iterable[Symbol],
        file_asts: Mapping[str, Tree],
        *,
        include_unresolved: bool = False,
    ) -> list[CallEdge]:
        """Build the call graph from pre-extracted symbols and parsed trees.

        This is the lower-level API referenced in the issue.  Use it when the
        ingestion pipeline has already parsed the trees and extracted symbols
        (so the trees are not re-parsed here).

        Args:
            symbols:            Flat iterable of :class:`Symbol` objects across
                                all files (each carries its ``file_path``).
            file_asts:          Mapping of ``file_path -> tree_sitter.Tree``.
            include_unresolved: See :meth:`build_from_files`.

        Returns:
            Deterministically ordered list of :class:`CallEdge` objects.
        """
        symbols_by_file: dict[str, list[Symbol]] = {}
        for sym in symbols:
            symbols_by_file.setdefault(sym.file_path, []).append(sym)
        # Files that have a tree but no symbols still need to be walked.
        for file_path in file_asts:
            symbols_by_file.setdefault(file_path, [])

        return self._build(
            symbols_by_file, file_asts, include_unresolved=include_unresolved
        )

    # ------------------------------------------------------------------
    # Core build
    # ------------------------------------------------------------------

    def _build(
        self,
        symbols_by_file: Mapping[str, list[Symbol]],
        trees_by_file: Mapping[str, Tree],
        *,
        include_unresolved: bool,
    ) -> list[CallEdge]:
        """Resolve every call site across all files into :class:`CallEdge` objects.

        - **Why it exists**: Central pass that ties together call-site discovery
          and cross-file resolution.
        - **Algorithm**: Builds one global :class:`_ResolutionContext` from all
          symbols, then for each file finds its call sites and resolves each
          against that shared context.
        - **Edge cases**: Unresolved calls are only emitted when
          ``include_unresolved`` is set; otherwise they are dropped so the
          default graph contains only edges to known project symbols.
        - **Correctness choice**: Edges are sorted by
          ``(caller_file, call_site_line, callee)`` so output is deterministic
          and diff-friendly for tests and snapshots.
        """
        context = _ResolutionContext(symbols_by_file)
        edges: list[CallEdge] = []

        for file_path, tree in trees_by_file.items():
            language = self._tree_language(file_path)
            finder = _CALL_SITE_FINDERS.get(language)
            if finder is None:
                continue
            for raw in finder(tree):
                edge = self._resolve_call(raw, file_path, context)
                if edge is None:
                    continue
                if edge.resolution == "unresolved" and not include_unresolved:
                    continue
                edges.append(edge)

        edges.sort(key=lambda e: (e.caller_file, e.call_site_line, e.callee))
        return edges

    def _resolve_call(
        self,
        raw: _RawCall,
        file_path: str,
        context: _ResolutionContext,
    ) -> CallEdge | None:
        """Resolve one raw call site into a :class:`CallEdge`.

        Returns an unresolved edge (callee = raw name, ``callee_file=None``)
        when no project symbol matches; the caller decides whether to keep it.
        """
        enclosure = context.enclosure(file_path, raw.line)

        if raw.is_attribute:
            resolved = context.resolve_attribute(
                raw.callee_name,
                raw.receiver,
                raw.receiver_is_self,
                file_path,
                enclosure.enclosing_class,
            )
        else:
            resolved = context.resolve_bare(raw.callee_name, file_path)

        if resolved is None:
            return CallEdge(
                caller=enclosure.caller,
                callee=raw.callee_name,
                caller_file=file_path,
                callee_file=None,
                call_site_line=raw.line,
                call_type="method" if raw.is_attribute else "function",
                resolution="unresolved",
            )

        target, resolution = resolved
        callee = target.qualified_name or target.name
        is_recursive = callee == enclosure.caller and target.file_path == file_path
        return CallEdge(
            caller=enclosure.caller,
            callee=callee,
            caller_file=file_path,
            callee_file=target.file_path,
            call_site_line=raw.line,
            call_type=self._call_type(target),
            resolution=resolution,
            is_recursive=is_recursive,
        )

    @staticmethod
    def _call_type(target: Symbol) -> CallType:
        """Classify a resolved call from the *target* symbol's kind.

        A class target is a ``constructor`` (instantiation); a method target is
        a ``method`` regardless of call syntax; everything else (module-level
        functions, including those reached via ``module.func()``) is a
        ``function``.  Deriving the category from the resolved symbol -- rather
        than the call syntax -- keeps ``lib.factorial()`` classified as a
        function and ``obj.greet()`` as a method.
        """
        if target.type == "class":
            return "constructor"
        if target.type == "method":
            return "method"
        return "function"

    # ------------------------------------------------------------------
    # Language inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_language(path: Path) -> str | None:
        """Infer a language from a file extension via ``settings.extension_map``."""
        from src.reporag.config import settings

        return settings.extension_map.get(path.suffix.lower())

    @staticmethod
    def _tree_language(file_path: str) -> str:
        """Infer the language of a tree keyed by *file_path* (defaults to python)."""
        from src.reporag.config import settings

        ext = Path(file_path).suffix.lower()
        return settings.extension_map.get(ext, "python")

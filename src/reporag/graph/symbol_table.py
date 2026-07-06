"""Global symbol table / registry.

Central lookup index. Given a symbol name, returns the defining file,
line range, type, and signature. Supports lookup by exact name, fully
qualified name, regex pattern, and file path.

Dependency on Issue 7
----------------------
This module builds directly on
:mod:`src.reporag.ingestion.symbol_extractor` (Issue 7): every
:class:`~src.reporag.ingestion.symbol_extractor.Symbol` produced by
``SymbolExtractor`` becomes one :class:`SymbolRecord` here.

Issue 7's ``Symbol.qualified_name`` is only *file-local* -- e.g.
``"Class.method"`` -- since it's built without knowledge of the module's
own dotted path. Two different files can legitimately both define
``Class.method`` and get the same ``qualified_name`` from the extractor.
This module's job is exactly to remove that ambiguity: every
:class:`SymbolRecord` gets a **globally** qualified name
(``module.Class.method``) built by prefixing the file's dotted module
name, so ``lookup_qualified`` can tell the two apart.

Known limitation inherited from Issue 7: ``Symbol.is_from_import`` and
``Symbol.from_module`` are declared on the dataclass and documented, but
the current extractor never actually sets them (they're always ``False``/
``None``). They're carried through onto ``SymbolRecord`` as-is so this
table picks them up for free once Issue 7 populates them, but they can't
be relied on today -- use ``import_source`` / ``import_alias``, which
*are* populated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor, SymbolType

# ---------------------------------------------------------------------------
# Module-name helper (kept in sync with the call/dependency graph builders)
# ---------------------------------------------------------------------------


def _module_name(file_path: str) -> str:
    """Return the dotted module name for *file_path*.

    Same convention used by
    :mod:`src.reporag.graph.call_graph` and
    :mod:`src.reporag.graph.dependency_graph`, so a symbol's module prefix
    here lines up with how those graphs would refer to the same file.
    """
    pure = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(pure.parts[:-1]) + [pure.stem]
    if pure.stem == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p not in ("", ".", "/"))


def _global_qualified_name(module: str, local_qualified: str) -> str:
    """Prefix a file-local qualified name with its module, disambiguating it."""
    if not module:
        return local_qualified
    return f"{module}.{local_qualified}"


def _make_symbol_id(file_path: str, qualified_name: str, start_line: int) -> str:
    """Build a deterministic, globally unique identifier for one symbol.

    Format: ``"{file_path}::{qualified_name}:{start_line}"``. Deterministic
    (same input always produces the same id, so re-registering the same
    source twice is idempotent) and human-readable, so it doubles as a
    debugging aid and a stable dict/JSON key.
    """
    return f"{file_path}::{qualified_name}:{start_line}"


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class SymbolRecord:
    """One registered symbol with full lookup metadata.

    Attributes:
        symbol_id: Globally unique, stable, deterministic identifier (see
            :func:`_make_symbol_id`). Safe to use as a dict key or as a
            JSON object's own reference.
        name: The bare symbol name as written (e.g. ``"method"``, not
            ``"Class.method"``) -- expected to collide across files and
            classes by design; use ``qualified_name`` or ``symbol_id`` to
            disambiguate.
        qualified_name: Globally qualified name -- ``module.Class.method``
            for definitions, ``module.bound_name`` for imports. Unique in
            practice (two genuinely different definitions at the same
            dotted path in the same module would be a real Python name
            collision), though :meth:`SymbolTable.lookup_qualified` still
            returns a list to handle legitimate cases like re-imports that
            rebind the same name at different lines.
        type: ``"class" | "function" | "method" | "import"``.
        file_path: File the symbol was defined or bound in.
        module: Dotted module name of ``file_path``.
        start_line: 1-based start line.
        end_line: 1-based end line.
        signature: Function/method signature text (``None`` for
            class/import symbols).
        docstring: Docstring text if present.
        decorators: Decorator expressions, outermost first.
        bases: Base class expressions (classes only).
        is_async: ``True`` for ``async def``.
        parent_symbol: File-local qualified name of the enclosing
            class/function, mirrored from Issue 7's ``Symbol`` (``None``
            for module-level symbols).
        parent_qualified_name: Globally qualified version of
            ``parent_symbol`` (``None`` if there is no parent).
        import_source: For imports, the fully-qualified origin of the
            bound name (see ``Symbol.import_source``).
        import_alias: The ``as`` alias, if any.
        is_wildcard_import: ``True`` for ``from x import *``.
        is_from_import:
        Whether this import originated from a `from ... import ...`
        statement. Uses the value provided by Symbol when available,
        otherwise defaults to False.
        docstring's "Known limitation" note; not currently populated
            by the extractor.
        from_module:
        Source module for `from ... import ...` imports when available.
        Defaults to None if the extractor does not provide this metadata.
        has_parse_error: ``True`` if extracted from a source region with a
            syntax error.
    """

    symbol_id: str
    name: str
    qualified_name: str
    type: SymbolType
    file_path: str
    module: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    is_async: bool = False
    parent_symbol: str | None = None
    parent_qualified_name: str | None = None
    import_source: str | None = None
    import_alias: str | None = None
    is_wildcard_import: bool = False
    is_from_import: bool = False
    from_module: str | None = None
    has_parse_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-primitive dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymbolRecord:
        """Reconstruct a :class:`SymbolRecord` from :meth:`to_dict` output."""
        return cls(**dict(data))


def _record_from_symbol(sym: Symbol, module: str) -> SymbolRecord:
    """Build a :class:`SymbolRecord` from one (already flat) Issue-7 ``Symbol``.

    Issue 7's extractor returns a flat list per file -- methods and nested
    classes appear as their own top-level entries in that list, linked to
    their parent only via ``parent_symbol`` / ``qualified_name`` -- so no
    recursive flattening is needed here.
    """
    if sym.type == "import":
        # An import's "qualified name" is where the bound identifier lives
        # in the *importing* module's own namespace, not the thing it
        # points at (that's what import_source is for).
        local_qualified = sym.name
    else:
        local_qualified = sym.qualified_name or sym.name

    qualified_name = _global_qualified_name(module, local_qualified)
    parent_qualified_name = (
        _global_qualified_name(module, sym.parent_symbol) if sym.parent_symbol else None
    )
    symbol_id = _make_symbol_id(sym.file_path, qualified_name, sym.start_line)

    return SymbolRecord(
        symbol_id=symbol_id,
        name=sym.name,
        qualified_name=qualified_name,
        type=sym.type,
        file_path=sym.file_path,
        module=module,
        start_line=sym.start_line,
        end_line=sym.end_line,
        signature=sym.signature,
        docstring=sym.docstring,
        decorators=list(sym.decorators),
        bases=list(sym.bases),
        is_async=sym.is_async,
        parent_symbol=sym.parent_symbol,
        parent_qualified_name=parent_qualified_name,
        import_source=sym.import_source,
        import_alias=sym.import_alias,
        is_wildcard_import=sym.is_wildcard_import,
        is_from_import=getattr(sym, "is_from_import", False),
        from_module=getattr(sym, "from_module", None),
        has_parse_error=sym.has_parse_error,
    )


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------


class SymbolTable:
    """Central registry mapping ``symbol_id -> SymbolRecord``.

    Maintains secondary indices for name, qualified-name, and file-path
    lookups so those queries stay O(1)/O(matches) rather than scanning
    every record. A single instance can hold the whole repository.

    Args:
        extractor: Optional pre-built
            :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
            (Issue 7), reused by the ``build_from_*`` classmethods (inject
            in tests, or to share grammar-loading cost with a call/
            dependency graph builder on the same repository).
    """

    def __init__(self, extractor: SymbolExtractor | None = None) -> None:
        """Initialise an empty table and its shared extractor."""
        self._extractor = extractor if extractor is not None else SymbolExtractor()
        self._records: dict[str, SymbolRecord] = {}
        self._by_name: dict[str, list[str]] = {}
        self._by_qualified_name: dict[str, list[str]] = {}
        self._by_file: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_files(
        cls,
        paths: Iterable[str | Path],
        *,
        extractor: SymbolExtractor | None = None,
    ) -> SymbolTable:
        """Parse *paths* from disk and register every symbol found.

        Files that fail to read or parse are skipped (matching the
        tolerant, best-effort behaviour of the call/dependency graph
        builders); everything else is registered.
        """
        table = cls(extractor=extractor)
        symbols_by_file: dict[str, list[Symbol]] = {}
        for path in paths:
            fpath = Path(path)
            try:
                symbols_by_file[str(fpath)] = table._extractor.extract_from_file(fpath)
            except Exception:  # noqa: BLE001 - best-effort ingestion
                continue
        table.register_symbols(symbols_by_file)
        return table

    @classmethod
    def build_from_sources(
        cls,
        sources: Mapping[str, str | bytes],
        *,
        language: str = "python",
        extractor: SymbolExtractor | None = None,
    ) -> SymbolTable:
        """Build a table from in-memory ``{file_path: source}`` -- ideal for tests."""
        table = cls(extractor=extractor)
        symbols_by_file = {
            file_path: table._extractor.extract_from_source(
                source, language=language, file_path=file_path
            )
            for file_path, source in sources.items()
        }
        table.register_symbols(symbols_by_file)
        return table

    @classmethod
    def build_from_symbols(
        cls,
        symbols_by_file: Mapping[str, Iterable[Symbol]],
        *,
        extractor: SymbolExtractor | None = None,
    ) -> SymbolTable:
        """Build a table directly from already-extracted Issue-7 ``Symbol`` lists."""
        table = cls(extractor=extractor)
        table.register_symbols(symbols_by_file)
        return table

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, record: SymbolRecord) -> str:
        """Register (or replace) one record, indexing it under every lookup key.

        Re-registering a record with a ``symbol_id`` that's already
        present replaces the old entry in place first, so re-running
        extraction on an edited file and re-registering its symbols never
        accumulates stale duplicates.
        """
        if record.symbol_id in self._records:
            self._deindex(record.symbol_id)
        self._records[record.symbol_id] = record
        self._by_name.setdefault(record.name, []).append(record.symbol_id)
        self._by_qualified_name.setdefault(record.qualified_name, []).append(
            record.symbol_id
        )
        self._by_file.setdefault(record.file_path, []).append(record.symbol_id)
        return record.symbol_id

    def register_symbols(
        self, symbols_by_file: Mapping[str, Iterable[Symbol]]
    ) -> list[str]:
        """Register every ``Symbol`` for a set of files, returning the new ids."""
        ids: list[str] = []
        for file_path, symbols in symbols_by_file.items():
            module = _module_name(file_path)
            for sym in symbols:
                record = _record_from_symbol(sym, module)
                ids.append(self.register(record))
        return ids

    def unregister_file(self, file_path: str) -> None:
        """Remove every record for *file_path* (e.g. before re-registering an edit)."""
        for symbol_id in list(self._by_file.get(file_path, [])):
            self._deindex(symbol_id)
            del self._records[symbol_id]

    def _deindex(self, symbol_id: str) -> None:
        """Remove *symbol_id* from every secondary index (not from ``_records``)."""
        old = self._records.get(symbol_id)
        if old is None:
            return
        for index, key in (
            (self._by_name, old.name),
            (self._by_qualified_name, old.qualified_name),
            (self._by_file, old.file_path),
        ):
            ids = index.get(key)
            if ids and symbol_id in ids:
                ids.remove(symbol_id)
                if not ids:
                    del index[key]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, symbol_id: str) -> SymbolRecord | None:
        """Look up a single record by its exact ``symbol_id``."""
        return self._records.get(symbol_id)

    def lookup_exact(self, name: str) -> list[SymbolRecord]:
        """Look up every record whose bare ``name`` matches exactly.

        This is the collision-prone mode by design: ``"process"`` will
        return one match per file/class that defines a symbol called
        ``process``. Use :meth:`lookup_qualified` to disambiguate.
        """
        return [self._records[i] for i in self._by_name.get(name, [])]

    def lookup_qualified(self, qualified_name: str) -> list[SymbolRecord]:
        """Look up every record whose globally qualified name matches exactly.

        Returns a list rather than a single record because legitimate
        rebinding (e.g. two ``from x import y`` / ``from z import y``
        statements at different lines in the same file) can share a
        qualified name while being genuinely different symbols.
        """
        return [
            self._records[i] for i in self._by_qualified_name.get(qualified_name, [])
        ]

    def lookup_regex(
        self,
        pattern: str | re.Pattern[str],
        *,
        field_name: str = "qualified_name",
    ) -> list[SymbolRecord]:
        """Look up every record whose *field_name* matches *pattern* (``re.search``).

        Args:
            pattern: A regex string or a pre-compiled pattern.
            field_name: Which string field to match against -- ``"name"``
                or ``"qualified_name"`` (default).

        Returns:
            Matching records, in registration order.

        Raises:
            ValueError: If ``field_name`` isn't ``"name"`` or
                ``"qualified_name"``.
        """
        if field_name not in ("name", "qualified_name"):
            raise ValueError(
                f"field_name must be 'name' or 'qualified_name', got {field_name!r}"
            )
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        return [
            record
            for record in self._records.values()
            if compiled.search(getattr(record, field_name))
        ]

    def lookup_by_file(self, file_path: str) -> list[SymbolRecord]:
        """Look up every record defined/bound in *file_path*, in registration order."""
        return [self._records[i] for i in self._by_file.get(file_path, [])]

    def all(self) -> list[SymbolRecord]:
        """Return every registered record, in registration order."""
        return list(self._records.values())

    def __len__(self) -> int:
        """Return the number of registered records."""
        return len(self._records)

    def __contains__(self, symbol_id: str) -> bool:
        """Return ``True`` if *symbol_id* is registered."""
        return symbol_id in self._records

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        """Return every record as a list of JSON-primitive dicts, sorted by id.

        Sorted (rather than registration order) so serialization is
        deterministic across runs -- important for diffing persisted
        snapshots.
        """
        return [
            r.to_dict()
            for r in sorted(self._records.values(), key=lambda r: r.symbol_id)
        ]

    @classmethod
    def from_dict(
        cls,
        payload: Iterable[Mapping[str, Any]],
        *,
        extractor: SymbolExtractor | None = None,
    ) -> SymbolTable:
        """Rebuild a table from :meth:`to_dict` output."""
        table = cls(extractor=extractor)
        for item in payload:
            table.register(SymbolRecord.from_dict(item))
        return table

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the whole table to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls, data: str, *, extractor: SymbolExtractor | None = None
    ) -> SymbolTable:
        """Rebuild a table from :meth:`to_json` output."""
        return cls.from_dict(json.loads(data), extractor=extractor)

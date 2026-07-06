"""Global symbol table / registry.

The symbol table is the code knowledge graph's *central lookup index*. Given a
symbol name it answers "where is this defined, what kind of thing is it, and
what is its signature?" -- returning the defining file, line range, type,
signature and docstring. The call graph (Issue 9) and dependency graph
(Issue 10) reference definitions by their *fully qualified name*; this table is
what resolves those names back to concrete :class:`SymbolRecord` metadata.

Where the extractor stops and the table begins
----------------------------------------------
Issue 7's :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
produces a flat list of :class:`~src.reporag.ingestion.symbol_extractor.Symbol`
objects per file, each already carrying a *file-local* qualified name
(``Calculator.add``, not ``pkg.mod.Calculator.add``). The symbol table's job is
to lift those per-file symbols into a single **repository-global** registry:

* It prefixes every qualified name with the symbol's dotted *module* path
  (``examples.sample_repo.db.User.__init__``) so names are unique across the
  whole project, matching the "module.class.method" fully-qualified form the
  issue asks for. The module-name derivation mirrors
  :func:`src.reporag.graph.dependency_graph._module_name` /
  :class:`src.reporag.graph.call_graph._ModuleIndex` so all three agree on how
  a file path maps to a module.
* It assigns each record a stable, collision-free ``symbol_id`` (the module
  FQN, disambiguated with a ``#n`` suffix only on a genuine same-name
  redefinition) that Neo4j (Issue 12) can use as a node key.
* It builds four secondary indexes so lookups are O(1)/O(matches) rather than a
  linear scan: by bare name, by fully qualified name, by file path, and (for
  regex) the ordered record list.

Lookup surface
--------------
* :meth:`SymbolTable.lookup` -- exact *bare* name, returns **all** matches
  across files (``handle_login`` defined in two modules -> two records).
* :meth:`SymbolTable.lookup_qualified` -- fully qualified name, returns the
  **unique** definition (or ``None``).
* :meth:`SymbolTable.lookup_regex` -- regex over bare names, qualified names or
  file paths (``r"test_.*"`` finds every test function).
* :meth:`SymbolTable.lookup_file` -- every definition in a given file, in
  definition order.

Import symbols (``type="import"``) are *bindings*, not definitions -- they have
no body, signature or docstring to look up -- so they are skipped by default
(pass ``include_imports=True`` to keep them). The dependency graph is the right
tool for reasoning about imports.

Persistence
-----------
:meth:`SymbolTable.to_json` / :meth:`SymbolTable.from_json` (and the ``dict``
variants) round-trip the whole registry losslessly, preserving each
``symbol_id`` so a reloaded table is byte-for-byte equivalent -- useful for
caching an ingested repo and for debugging.

Usage::

    from src.reporag.graph.symbol_table import SymbolTable
    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    extractor = SymbolExtractor()
    all_symbols = extractor.extract_from_file("examples/sample_repo/auth.py")

    table = SymbolTable()
    table.register_symbols(all_symbols)

    for r in table.lookup("authenticate_user"):
        print(f"{r.qualified_name} @ {r.file_path}:{r.start_line}")
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)

# Schema version stamped into serialized output so a future format change can be
# migrated rather than silently mis-read.
_SCHEMA_VERSION = 1

# What a regex lookup matches against.
RegexTarget = Literal["name", "qualified", "file"]


# ---------------------------------------------------------------------------
# SymbolRecord
# ---------------------------------------------------------------------------


@dataclass
class SymbolRecord:
    """A single flat, repository-global entry in the :class:`SymbolTable`.

    This is the registry's own value type -- deliberately decoupled from the
    extractor's hierarchical
    :class:`~src.reporag.ingestion.symbol_extractor.Symbol` (which carries
    tree-sitter-shaped ``methods``/``children`` lists). A record is a flat,
    JSON-friendly description of *one* definition.

    Attributes:
        symbol_id: Stable, globally unique registry key. Equal to
            ``qualified_name`` except when a name is genuinely redefined in one
            scope, where a ``#n`` suffix disambiguates (``mod.f`` / ``mod.f#2``).
        name: The bare, unqualified identifier (``__init__``).
        qualified_name: The module-prefixed fully qualified name
            (``examples.sample_repo.db.User.__init__``).
        type: Symbol kind -- ``"class"``, ``"function"``, ``"method"``,
            ``"variable"``, ... (free-form ``str`` so non-Python extractors can
            contribute their own kinds).
        file_path: Path to the file that defines the symbol.
        module: Dotted module name of ``file_path``.
        start_line: 1-based first line of the definition.
        end_line: 1-based last line of the definition.
        signature: Source signature for callables (``def add(self, x: int)``),
            or ``None``.
        docstring: The symbol's docstring, or ``None``.
        parent: ``qualified_name`` of the enclosing class/function, or ``None``
            for a module-level definition.
        decorators: Decorator expressions applied to the symbol (``@property``).
        bases: Base-class expressions for a class, else empty.
        is_async: ``True`` for ``async def``.
        language: Source language the symbol was extracted from.
    """

    symbol_id: str
    name: str
    qualified_name: str
    type: str
    file_path: str
    module: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    parent: str | None = None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    is_async: bool = False
    language: str = "python"

    def __repr__(self) -> str:
        loc = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return f"SymbolRecord({self.qualified_name} [{self.type}] @ {loc})"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this record.

        Every value is a JSON primitive (``str``/``int``/``bool``/``None``) or a
        list thereof, so the output feeds straight into :func:`json.dumps`, a
        Neo4j node payload (Issue 12), or a JSONL debug dump.
        """
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "type": self.type,
            "file_path": self.file_path,
            "module": self.module,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent": self.parent,
            "decorators": list(self.decorators),
            "bases": list(self.bases),
            "is_async": self.is_async,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SymbolRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Unknown keys are ignored and missing optional keys fall back to their
        defaults, so an older serialized table still loads under a newer schema.
        """
        return cls(
            symbol_id=str(data["symbol_id"]),
            name=str(data["name"]),
            qualified_name=str(data["qualified_name"]),
            type=str(data["type"]),
            file_path=str(data["file_path"]),
            module=str(data.get("module", "")),
            start_line=int(data.get("start_line", 0)),  # type: ignore[arg-type]
            end_line=int(data.get("end_line", 0)),  # type: ignore[arg-type]
            signature=_opt_str(data.get("signature")),
            docstring=_opt_str(data.get("docstring")),
            parent=_opt_str(data.get("parent")),
            decorators=[str(d) for d in data.get("decorators", [])],  # type: ignore[union-attr]
            bases=[str(b) for b in data.get("bases", [])],  # type: ignore[union-attr]
            is_async=bool(data.get("is_async", False)),
            language=str(data.get("language", "python")),
        )


def _opt_str(value: object) -> str | None:
    """Coerce a JSON value to ``str`` while preserving ``None``."""
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Module-name helper (kept in sync with _ModuleIndex._parts)
# ---------------------------------------------------------------------------


def _module_name(file_path: str) -> str:
    """Return the dotted module name for *file_path*.

    Mirrors :meth:`src.reporag.graph.call_graph._ModuleIndex._parts` and
    :func:`src.reporag.graph.dependency_graph._module_name` so the module names
    on :class:`SymbolRecord` line up with how the call and dependency graphs name
    the same file (POSIX semantics on every host; ``__init__`` folds into its
    package directory).
    """
    pure = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(pure.parts[:-1]) + [pure.stem]
    if pure.stem == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p not in ("", ".", "/"))


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------


class SymbolTable:
    """A repository-global registry mapping ``symbol_id`` -> :class:`SymbolRecord`.

    - **Why it exists**: The call and dependency graphs speak in fully qualified
      names; something has to turn those names back into concrete metadata
      (file, line, type, signature, docstring). That something is this table.
    - **Algorithm**: A primary ``dict`` keyed by ``symbol_id`` preserves
      insertion (definition) order, backed by three secondary indexes -- bare
      name -> ids, qualified name -> id, file path -> ids -- so every documented
      lookup is a hash probe, not a scan.
    - **Edge cases**: Same-name symbols across files/classes stay distinct
      because their module-prefixed qualified names differ. A genuine
      redefinition in one scope (``def f`` twice) keeps both records but points
      qualified lookup at the *last* one, mirroring Python rebinding.
    - **Correctness choice**: Import symbols are skipped by default -- they are
      bindings, not definitions, and would otherwise shadow the real target in
      name lookups.

    The table is not thread-safe; build it once during ingestion, then treat it
    as read-only.
    """

    def __init__(self) -> None:
        """Create an empty symbol table."""
        # symbol_id -> record (insertion-ordered == definition order).
        self._records: dict[str, SymbolRecord] = {}
        # bare name -> [symbol_id] in definition order.
        self._by_name: dict[str, list[str]] = {}
        # fully qualified name -> symbol_id (last definition wins).
        self._by_qualified: dict[str, str] = {}
        # file path -> [symbol_id] in definition order.
        self._by_file: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_symbols(
        self,
        symbols: Iterable[Symbol],
        *,
        language: str = "python",
        include_imports: bool = False,
    ) -> list[SymbolRecord]:
        """Register every definition in *symbols*, returning the new records.

        Accepts the flat list a
        :class:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor`
        produces (each :class:`Symbol` already carries its ``file_path`` and a
        file-local ``qualified_name``). Import symbols are dropped unless
        *include_imports* is set.

        Args:
            symbols: Flat iterable of extracted symbols across any number of
                files.
            language: Source language recorded on each resulting record.
            include_imports: When ``True``, ``type="import"`` symbols are also
                registered (off by default -- see the class docstring).

        Returns:
            The list of :class:`SymbolRecord` objects created, in input order.
        """
        created: list[SymbolRecord] = []
        for symbol in symbols:
            record = self.register_symbol(
                symbol, language=language, include_imports=include_imports
            )
            if record is not None:
                created.append(record)
        return created

    def register_symbol(
        self,
        symbol: Symbol,
        *,
        language: str = "python",
        include_imports: bool = False,
    ) -> SymbolRecord | None:
        """Register a single extractor :class:`Symbol`.

        Returns the created :class:`SymbolRecord`, or ``None`` when the symbol
        is skipped (an import while *include_imports* is ``False``, or an
        unnamed symbol).
        """
        if symbol.type == "import" and not include_imports:
            return None
        if not symbol.name:
            return None

        module = _module_name(symbol.file_path)
        local_qualified = symbol.qualified_name or symbol.name
        qualified_name = f"{module}.{local_qualified}" if module else local_qualified
        parent = None
        if symbol.parent_symbol:
            parent = (
                f"{module}.{symbol.parent_symbol}" if module else symbol.parent_symbol
            )

        record = SymbolRecord(
            symbol_id="",  # assigned by add()
            name=symbol.name,
            qualified_name=qualified_name,
            type=symbol.type,
            file_path=symbol.file_path,
            module=module,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            signature=symbol.signature,
            docstring=symbol.docstring,
            parent=parent,
            decorators=list(symbol.decorators),
            bases=list(symbol.bases),
            is_async=symbol.is_async,
            language=language,
        )
        return self.add(record)

    def add(self, record: SymbolRecord) -> SymbolRecord:
        """Insert a pre-built :class:`SymbolRecord`, assigning a unique id.

        - **Why it exists**: The primitive every registration path funnels
          through, and the public seam for source-agnostic use -- persistence
          round-trips, non-Python extractors, or synthetic records (e.g.
          module-level variables an extractor does not yet emit).
        - **Algorithm**: Reuses ``record.symbol_id`` when it is set and still
          free (so :meth:`from_dict` preserves ids exactly); otherwise derives a
          collision-free id from ``qualified_name``. Then wires the record into
          the primary map and all three secondary indexes.
        - **Edge cases**: A repeated qualified name keeps both records (unique
          ids) but repoints qualified lookup at the newcomer -- Python's
          last-definition-wins rebinding semantics.
        - **Correctness choice**: Mutates *record.symbol_id* in place so the
          returned object and the stored object are the same instance with the
          final id.
        """
        symbol_id = record.symbol_id or record.qualified_name
        if symbol_id in self._records:
            symbol_id = self._unique_id(record.qualified_name)
        record.symbol_id = symbol_id

        self._records[symbol_id] = record
        self._by_name.setdefault(record.name, []).append(symbol_id)
        self._by_file.setdefault(record.file_path, []).append(symbol_id)
        # Last definition of a qualified name wins for unique lookup.
        self._by_qualified[record.qualified_name] = symbol_id
        return record

    def _unique_id(self, base: str) -> str:
        """Return ``base`` or the first free ``base#n`` variant (n >= 2)."""
        suffix = 2
        candidate = f"{base}#{suffix}"
        while candidate in self._records:
            suffix += 1
            candidate = f"{base}#{suffix}"
        return candidate

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> list[SymbolRecord]:
        """Return every definition with bare ``name``, in definition order.

        This is the collision-tolerant lookup: ``lookup("handle_login")``
        returns one record per file that defines it. Use
        :meth:`lookup_qualified` when a single, disambiguated result is wanted.
        """
        return [self._records[i] for i in self._by_name.get(name, ())]

    def lookup_qualified(self, qualified_name: str) -> SymbolRecord | None:
        """Return the unique definition for a fully qualified name, or ``None``.

        Accepts either the module-prefixed ``qualified_name``
        (``db.User.__init__``) or a ``symbol_id`` (they coincide unless a name
        was redefined). On a redefinition, the most recently registered
        definition wins.
        """
        symbol_id = self._by_qualified.get(qualified_name)
        if symbol_id is not None:
            return self._records[symbol_id]
        # Fall back to a direct id hit (handles disambiguated ``name#n`` ids).
        return self._records.get(qualified_name)

    def lookup_regex(
        self, pattern: str | re.Pattern[str], *, target: RegexTarget = "name"
    ) -> list[SymbolRecord]:
        """Return every record whose *target* field matches *pattern*.

        - **Why it exists**: Powers pattern queries such as "every test
          function" (``r"test_.*"``) or "everything in a package"
          (``target="qualified"``).
        - **Algorithm**: Compiles *pattern* once and scans records in definition
          order, keeping those where :meth:`re.Pattern.search` hits the selected
          field. ``search`` (not ``match``) is used so callers anchor
          explicitly with ``^``/``$`` when they want to.
        - **Edge cases**: An invalid pattern raises :class:`re.error` from the
          caller's ``pattern`` -- it is not swallowed.

        Args:
            pattern: A regex string or a pre-compiled :class:`re.Pattern`.
            target: Which field to match -- ``"name"`` (bare name, the default),
                ``"qualified"`` (fully qualified name), or ``"file"`` (path).
        """
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        return [
            r for r in self._records.values() if compiled.search(self._field(r, target))
        ]

    @staticmethod
    def _field(record: SymbolRecord, target: RegexTarget) -> str:
        """Return the record field a regex *target* selects."""
        if target == "qualified":
            return record.qualified_name
        if target == "file":
            return record.file_path
        return record.name

    def lookup_file(self, file_path: str) -> list[SymbolRecord]:
        """Return every definition in *file_path*, in definition order."""
        return [self._records[i] for i in self._by_file.get(file_path, ())]

    def get(self, symbol_id: str) -> SymbolRecord | None:
        """Return the record for a ``symbol_id``, or ``None`` if absent."""
        return self._records.get(symbol_id)

    # ------------------------------------------------------------------
    # Container protocol / introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered records."""
        return len(self._records)

    def __iter__(self) -> Iterator[SymbolRecord]:
        """Iterate records in definition (insertion) order."""
        return iter(self._records.values())

    def __contains__(self, symbol_id: object) -> bool:
        """Return ``True`` if a ``symbol_id`` is registered."""
        return symbol_id in self._records

    @property
    def records(self) -> list[SymbolRecord]:
        """All records in definition order (a fresh list, safe to mutate)."""
        return list(self._records.values())

    @property
    def files(self) -> list[str]:
        """Sorted list of distinct file paths that contributed a definition."""
        return sorted(self._by_file)

    def names(self) -> list[str]:
        """Sorted list of distinct bare names in the table."""
        return sorted(self._by_name)

    def counts_by_type(self) -> dict[str, int]:
        """Return ``{type: count}`` across all records (for debug summaries)."""
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.type] = counts.get(record.type, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialise the whole table to a JSON-ready ``dict``.

        Preserves each ``symbol_id`` and the definition ordering so
        :meth:`from_dict` reconstructs an equivalent table.
        """
        return {
            "version": _SCHEMA_VERSION,
            "symbols": [r.to_dict() for r in self._records.values()],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the table to a JSON string (pretty-printed by default)."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SymbolTable:
        """Rebuild a table from :meth:`to_dict` output, preserving ids.

        Records are re-added in their serialized order; because each already
        carries a unique ``symbol_id`` no disambiguation suffixes are re-minted,
        so a load-after-save is a faithful round-trip.
        """
        table = cls()
        raw_symbols = data.get("symbols", [])
        if not isinstance(raw_symbols, list):
            raise ValueError("SymbolTable.from_dict: 'symbols' must be a list")
        for raw in raw_symbols:
            if not isinstance(raw, Mapping):
                raise ValueError("SymbolTable.from_dict: each symbol must be a mapping")
            table.add(SymbolRecord.from_dict(raw))
        return table

    @classmethod
    def from_json(cls, text: str) -> SymbolTable:
        """Rebuild a table from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        """Write the table to *path* as JSON (parent dirs are not created)."""
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SymbolTable:
        """Load a table previously written with :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return f"SymbolTable({len(self._records)} symbols across {len(self._by_file)} files)"

"""Global symbol table / registry.

Central lookup index. Given a symbol name, returns the defining file,
line range, type, and signature. Supports lookup by exact name, fully
qualified name, regex pattern, and file path.
"""

from __future__ import annotations

import bisect
import difflib
import fnmatch
import hashlib
import json
import re
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from src.reporag.ingestion.symbol_extractor import Symbol


@dataclass(slots=True)
class SymbolRecord:
    """A record representing a symbol in the global registry."""

    symbol_id: str
    name: str
    qualified_name: str
    type: str
    file_path: str
    start_line: int
    end_line: int
    signature: str | None
    docstring: str | None
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolRecord:
        """Create a SymbolRecord from a dictionary."""
        return cls(**data)  # type: ignore


class SymbolTable:
    """The central registry for all symbols in the repository."""

    def __init__(self) -> None:
        """Initialise an empty symbol table."""
        self._lock = threading.RLock()
        self._registry: dict[str, SymbolRecord] = {}
        self._name_index: dict[str, set[str]] = {}
        self._qualified_name_index: dict[str, set[str]] = {}
        self._file_index: dict[str, set[str]] = {}
        self._type_index: dict[str, set[str]] = {}
        self._children_index: dict[str, set[str]] = {}
        self._file_intervals: dict[str, list[tuple[int, int, str]]] = {}

        # Caches for expensive queries
        self._regex_cache: dict[str, list[SymbolRecord]] = {}
        self._fuzzy_cache: dict[tuple[str, int], list[SymbolRecord]] = {}

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[SymbolRecord]:
        return iter(self._registry.values())

    def __contains__(self, item: str) -> bool:
        return item in self._name_index or item in self._qualified_name_index

    def clear(self) -> None:
        """Safely clear the entire registry and all indices."""
        with self._lock:
            self._registry.clear()
            self._name_index.clear()
            self._qualified_name_index.clear()
            self._file_index.clear()
            self._type_index.clear()
            self._children_index.clear()
            self._file_intervals.clear()
            self._regex_cache.clear()
            self._fuzzy_cache.clear()

    def _remove_sids_from_index(
        self, index: dict[str, set[str]], keys: Iterable[str], sids_to_remove: set[str]
    ) -> None:
        """Helper to efficiently remove multiple SIDs from a set index."""
        for key in keys:
            if key in index:
                index[key].difference_update(sids_to_remove)
                if not index[key]:
                    del index[key]

    def remove_by_file(self, file_path: str) -> None:
        """Remove all symbols associated with a specific file path."""
        with self._lock:
            sids_to_remove = self._file_index.pop(file_path, None)
            if not sids_to_remove:
                return

        affected_names = set()
        affected_qnames = set()
        affected_types = set()
        affected_parents = set()

        for sid in sids_to_remove:
            record = self._registry.pop(sid, None)
            if not record:
                continue

            affected_names.add(record.name)
            affected_qnames.add(record.qualified_name)
            affected_types.add(record.type)
            if record.parent_id:
                affected_parents.add(record.parent_id)

            self._children_index.pop(sid, None)

        self._remove_sids_from_index(self._name_index, affected_names, sids_to_remove)
        self._remove_sids_from_index(
            self._qualified_name_index, affected_qnames, sids_to_remove
        )
        self._remove_sids_from_index(self._type_index, affected_types, sids_to_remove)
        self._remove_sids_from_index(
            self._children_index, affected_parents, sids_to_remove
        )
        self._file_intervals.pop(file_path, None)
        self._regex_cache.clear()
        self._fuzzy_cache.clear()

    def update_file(self, file_path: str, symbols: Iterable[Symbol]) -> None:
        """Replace all symbols for a specific file with a new set of symbols."""
        self.remove_by_file(file_path)
        self.register_symbols(symbols)

    def _infer_module_name(self, file_path: str) -> str:
        """Convert a file path into a Python module name.

        Handles absolute paths by stripping arbitrary OS prefixes and
        anchoring at standard source roots like 'src', 'app', or 'lib'.
        """
        pure = PurePosixPath(file_path.replace("\\", "/"))
        parts = list(pure.parts[:-1]) + [pure.stem]

        # Anchor to known source directories to avoid absolute path pollution
        for anchor in ("src", "app", "lib", "site-packages"):
            if anchor in parts:
                parts = parts[parts.index(anchor) :]
                break

        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(p for p in parts if p not in ("", ".", "/"))

    def _generate_symbol_id(
        self, file_path: str, qualified_name: str, start_line: int
    ) -> str:
        """Generate a stable, deterministic ID for a symbol to ensure idempotent ingestion."""
        unique_str = f"{file_path}::{qualified_name}"
        base_id = hashlib.sha1(unique_str.encode("utf-8")).hexdigest()[:16]

        final_id = base_id
        counter = 1
        while final_id in self._registry:
            existing = self._registry[final_id]
            # If it's the exact same symbol (re-entrant parsing), keep the ID
            if (
                existing.file_path == file_path
                and existing.qualified_name == qualified_name
                and existing.start_line == start_line
            ):
                break
            final_id = f"{base_id}_{counter}"
            counter += 1

        return final_id

    def _flatten_and_register(
        self, symbol: Symbol, module_name: str, parent_id: str | None = None
    ) -> None:
        """Recursively flatten and register a symbol and its children."""
        # Compute fully qualified name
        qname = symbol.qualified_name or symbol.name
        fully_qualified_name = f"{module_name}.{qname}" if module_name else qname

        symbol_id = self._generate_symbol_id(
            symbol.file_path, fully_qualified_name, symbol.start_line
        )

        record = SymbolRecord(
            symbol_id=symbol_id,
            name=symbol.name,
            qualified_name=fully_qualified_name,
            type=symbol.type,
            file_path=symbol.file_path,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            signature=symbol.signature,
            docstring=symbol.docstring,
            parent_id=parent_id,
        )

        self._registry[symbol_id] = record

        self._name_index.setdefault(record.name, set()).add(symbol_id)
        self._qualified_name_index.setdefault(record.qualified_name, set()).add(
            symbol_id
        )
        self._file_index.setdefault(record.file_path, set()).add(symbol_id)
        self._type_index.setdefault(record.type, set()).add(symbol_id)

        if parent_id:
            self._children_index.setdefault(parent_id, set()).add(symbol_id)

        for child in symbol.methods:
            self._flatten_and_register(child, module_name, parent_id=symbol_id)
        for child in symbol.children:
            self._flatten_and_register(child, module_name, parent_id=symbol_id)

    def register_symbols(self, symbols: Iterable[Symbol]) -> None:
        """Register a list of root symbols and their children recursively."""
        with self._lock:
            modified_files = set()
            for symbol in symbols:
                module_name = self._infer_module_name(symbol.file_path)
                self._flatten_and_register(symbol, module_name)
                modified_files.add(symbol.file_path)

            # Rebuild O(log N) positional intervals for modified files
            for file_path in modified_files:
                intervals = [
                    (r.start_line, r.end_line, r.symbol_id)
                    for r in self.lookup_by_file(file_path)
                ]
                intervals.sort(key=lambda x: x[0])
                self._file_intervals[file_path] = intervals

            self._regex_cache.clear()
            self._fuzzy_cache.clear()

    def get_by_id(self, symbol_id: str) -> SymbolRecord | None:
        """Retrieve a symbol directly by its unique ID."""
        return self._registry.get(symbol_id)

    def lookup(self, name: str) -> list[SymbolRecord]:
        """Lookup symbols by exact name (e.g. 'authenticate')."""
        with self._lock:
            return [self._registry[sid] for sid in self._name_index.get(name, {})]

    def lookup_by_qualified_name(self, qualified_name: str) -> list[SymbolRecord]:
        """Lookup symbols by fully qualified name (returns unique match per file)."""
        with self._lock:
            return [
                self._registry[sid]
                for sid in self._qualified_name_index.get(qualified_name, {})
            ]

    def lookup_by_regex(self, pattern: str) -> list[SymbolRecord]:
        """Lookup symbols using a regex pattern against the fully qualified name or name."""
        with self._lock:
            if pattern in self._regex_cache:
                return list(self._regex_cache[pattern])

            try:
                regex = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: '{pattern}'") from e

            results = [
                record
                for record in self._registry.values()
                if regex.search(record.name) or regex.search(record.qualified_name)
            ]
            self._regex_cache[pattern] = results
            return list(results)

    def lookup_fuzzy(self, query: str, limit: int = 3) -> list[SymbolRecord]:
        """Fuzzy search for typo-tolerant matching against symbol names."""
        with self._lock:
            cache_key = (query, limit)
            if cache_key in self._fuzzy_cache:
                return list(self._fuzzy_cache[cache_key])

            # Fallback to fast substring matching first to save heavy difflib CPU cycles
            query_lower = query.lower()
            exact_substring_matches = [
                name for name in self._name_index if query_lower in name.lower()
            ]

            # Only use difflib on the filtered subset if possible, else full keys
            search_space = (
                exact_substring_matches
                if exact_substring_matches
                else self._name_index.keys()
            )

            matches = difflib.get_close_matches(
                query, search_space, n=limit, cutoff=0.6
            )

            results = []
            for match in matches:
                # Extend matches from exact lookup
                results.extend(self.lookup(match))

            final_results = results[:limit]
            self._fuzzy_cache[cache_key] = final_results
            return list(final_results)

    def lookup_by_file(self, file_path: str) -> list[SymbolRecord]:
        """Lookup all symbols defined in a specific file."""
        with self._lock:
            return [self._registry[sid] for sid in self._file_index.get(file_path, {})]

    def lookup_by_file_pattern(self, glob_pattern: str) -> list[SymbolRecord]:
        """Lookup symbols across files matching a glob pattern."""
        results = []
        for file_path, sids in self._file_index.items():
            if fnmatch.fnmatch(file_path, glob_pattern):
                for sid in sids:
                    results.append(self._registry[sid])
        return results

    def lookup_by_type(self, symbol_type: str) -> list[SymbolRecord]:
        """Lookup all symbols of a specific type (e.g. 'class', 'function')."""
        return [self._registry[sid] for sid in self._type_index.get(symbol_type, {})]

    def lookup_by_position(
        self, file_path: str, line_number: int
    ) -> SymbolRecord | None:
        """Find the innermost symbol that encloses the given line number in a file."""
        with self._lock:
            intervals = self._file_intervals.get(file_path)
            if not intervals:
                return None

            # O(log N) search for the right-most interval whose start_line <= line_number
            idx = bisect.bisect_right(intervals, (line_number, float("inf"), ""))

            best_match = None
            min_length = float("inf")

            # Walk backwards from insertion point to find the tightest wrapping interval
            for i in range(idx - 1, -1, -1):
                start, end, sid = intervals[i]
                if start <= line_number <= end:
                    length = end - start
                    if length < min_length:
                        min_length = length
                        best_match = sid

            if best_match:
                return self._registry.get(best_match)
            return None

    def lookup_hierarchy_by_position(
        self, file_path: str, line_number: int
    ) -> list[SymbolRecord]:
        """Return the stack of symbols enclosing the position, from outermost to innermost."""
        innermost = self.lookup_by_position(file_path, line_number)
        if not innermost:
            return []

        hierarchy = []
        current = innermost
        while current:
            hierarchy.insert(0, current)
            current = self.get_parent(current.symbol_id) if current.parent_id else None

        return hierarchy

    def get_parent(self, symbol_id: str) -> SymbolRecord | None:
        """Retrieve the parent symbol of a given symbol_id."""
        record = self._registry.get(symbol_id)
        if record and record.parent_id:
            return self._registry.get(record.parent_id)
        return None

    def get_children(self, symbol_id: str) -> list[SymbolRecord]:
        """Retrieve all immediate children of a given symbol_id."""
        return [self._registry[sid] for sid in self._children_index.get(symbol_id, {})]

    def to_json(self) -> str:
        """Serialize the symbol table to JSON."""
        data = {sid: record.to_dict() for sid, record in self._registry.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> SymbolTable:
        """Deserialize a symbol table from JSON."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return cls()

        table = cls()
        for sid, record_dict in data.items():
            record = SymbolRecord.from_dict(record_dict)
            table._registry[sid] = record
            table._name_index.setdefault(record.name, set()).add(sid)
            table._qualified_name_index.setdefault(record.qualified_name, set()).add(
                sid
            )
            table._file_index.setdefault(record.file_path, set()).add(sid)
            table._type_index.setdefault(record.type, set()).add(sid)
            if record.parent_id:
                table._children_index.setdefault(record.parent_id, set()).add(sid)
        return table

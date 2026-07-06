"""Global symbol table / registry.

Central lookup index. Given a symbol name, returns the defining file,
line range, type, and signature. Supports lookup by exact name, fully
qualified name, regex pattern, and file path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Import Symbol safely for type annotations
try:
    from ..ingestion.symbol_extractor import Symbol
except ImportError:
    # Fallback/Mock for testing or if environment differs
    Symbol = Any


@dataclass
class SymbolRecord:
    """Metadata record for a registered symbol."""

    symbol_id: str
    file_path: str
    start_line: int
    end_line: int
    type: str
    signature: str | None
    docstring: str | None
    name: str
    qualified_name: str
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    import_source: str | None = None
    import_alias: str | None = None
    is_wildcard_import: bool = False


class SymbolTable:
    """Global registry mapping symbol IDs to metadata records.

    Allows registering extracted symbols and performing lookup by exact name,
    fully qualified name, regex pattern, and file path. Can be serialized to/from JSON.
    """

    def __init__(self) -> None:
        """Initialize an empty symbol table."""
        self.registry: dict[str, SymbolRecord] = {}

    def _get_module_name(self, file_path: str | None) -> str:
        """Construct the dotted module path from a file path."""
        if not file_path or file_path == "<string>":
            return ""

        # Normalize separators
        path_str = file_path.replace("\\", "/")
        path = Path(path_str)

        if path.is_absolute():
            # Try to make it relative to CWD if possible
            try:
                path = path.relative_to(Path.cwd())
            except ValueError:
                # Fall back to finding 'src' or 'reporag' in path parts to strip root path
                parts = list(path.parts)
                if "src" in parts:
                    idx = parts.index("src")
                    path = Path(*parts[idx:])
                elif "reporag-ai_2" in parts:
                    idx = parts.index("reporag-ai_2")
                    path = Path(*parts[idx + 1 :])

        # Split path and append stem
        parts = list(path.parent.parts) + [path.stem]
        if path.stem == "__init__":
            parts = parts[:-1]

        # Clean empty and root characters
        cleaned_parts = [p for p in parts if p not in ("", ".", "/")]
        return ".".join(cleaned_parts)

    def register_symbols(self, all_symbols: Iterable[Symbol]) -> None:
        """Register all symbols with fully qualified names (module.class.method)."""
        visited = set()

        def _register(sym: Any) -> None:
            # Handle dictionary-like inputs or Symbol objects
            is_dict = isinstance(sym, dict)

            # Avoid cyclic/redundant processing by storing id(sym) or a unique identifier
            sym_ref = id(sym) if not is_dict else sym.get("symbol_id", str(sym))
            if sym_ref in visited:
                return
            visited.add(sym_ref)

            # Extract fields
            name = sym.get("name") if is_dict else getattr(sym, "name", None)
            file_path = (
                sym.get("file_path") if is_dict else getattr(sym, "file_path", None)
            )
            start_line = (
                sym.get("start_line") if is_dict else getattr(sym, "start_line", 0)
            )
            end_line = sym.get("end_line") if is_dict else getattr(sym, "end_line", 0)
            sym_type = sym.get("type") if is_dict else getattr(sym, "type", "unknown")
            signature = (
                sym.get("signature") if is_dict else getattr(sym, "signature", None)
            )
            docstring = (
                sym.get("docstring") if is_dict else getattr(sym, "docstring", None)
            )
            local_qname = (
                sym.get("qualified_name")
                if is_dict
                else getattr(sym, "qualified_name", None)
            )

            is_async = (
                sym.get("is_async") if is_dict else getattr(sym, "is_async", False)
            )
            decorators = (
                sym.get("decorators") if is_dict else getattr(sym, "decorators", [])
            )
            bases = sym.get("bases") if is_dict else getattr(sym, "bases", [])
            import_source = (
                sym.get("import_source")
                if is_dict
                else getattr(sym, "import_source", None)
            )
            import_alias = (
                sym.get("import_alias")
                if is_dict
                else getattr(sym, "import_alias", None)
            )
            is_wildcard_import = (
                sym.get("is_wildcard_import")
                if is_dict
                else getattr(sym, "is_wildcard_import", False)
            )

            if not name:
                return

            # Clean/determine module name and qualified name
            module_name = self._get_module_name(file_path)
            qname_suffix = local_qname or name

            if module_name and module_name != "<string>":
                fully_qualified = f"{module_name}.{qname_suffix}"
            else:
                fully_qualified = qname_suffix

            # Ensure symbol_id is unique
            symbol_id = f"{file_path}::{fully_qualified}"

            record = SymbolRecord(
                symbol_id=symbol_id,
                file_path=file_path or "",
                start_line=start_line,
                end_line=end_line,
                type=str(sym_type),
                signature=signature,
                docstring=docstring,
                name=name,
                qualified_name=fully_qualified,
                is_async=bool(is_async),
                decorators=list(decorators),
                bases=list(bases),
                import_source=import_source,
                import_alias=import_alias,
                is_wildcard_import=bool(is_wildcard_import),
            )
            self.registry[symbol_id] = record

            # Recursively register methods / children
            methods = sym.get("methods") if is_dict else getattr(sym, "methods", [])
            children = sym.get("children") if is_dict else getattr(sym, "children", [])

            if methods:
                for method in methods:
                    _register(method)
            if children:
                for child in children:
                    _register(child)

        for sym in all_symbols:
            _register(sym)

    def lookup(self, query: str) -> list[SymbolRecord]:
        """Look up symbols by exact name, fully qualified name, regex pattern, or file path."""
        matched: dict[str, SymbolRecord] = {}

        # 1. Exact match on name
        for record in self.registry.values():
            if record.name == query:
                matched[record.symbol_id] = record

        # 2. Exact match or suffix match on fully qualified name
        for record in self.registry.values():
            if record.qualified_name == query or record.qualified_name.endswith(
                "." + query
            ):
                matched[record.symbol_id] = record

        # 3. Match on file path
        # Match exact file path, or if file path ends with query (e.g. "symbol_table.py")
        for record in self.registry.values():
            if record.file_path == query or record.file_path.replace(
                "\\", "/"
            ).endswith("/" + query.replace("\\", "/")):
                matched[record.symbol_id] = record

        # 4. Regex match
        # Perform regex lookup only if the query contains regex metacharacters
        is_regex = any(c in query for c in "*?+[]()^{}|\\$")
        if is_regex:
            try:
                pattern = re.compile(query)
                for record in self.registry.values():
                    if pattern.match(record.name) or pattern.match(
                        record.qualified_name
                    ):
                        matched[record.symbol_id] = record
            except re.error:
                pass

        return list(matched.values())

    def to_json(self) -> str:
        """Serialize the symbol table to a JSON string."""
        data = {
            symbol_id: asdict(record) for symbol_id, record in self.registry.items()
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> SymbolTable:
        """Deserialize the symbol table from a JSON string."""
        table = cls()
        data = json.loads(json_str)
        for symbol_id, record_dict in data.items():
            record = SymbolRecord(**record_dict)
            table.registry[symbol_id] = record
        return table

"""Symbol extractor.

Walks a tree-sitter AST and extracts the meaningful code entities that become
nodes in the knowledge graph and units for embedding: functions, classes,
methods, and module-level imports.

Raw ASTs are too granular to reason about directly. This module distills a tree
into a flat, source-ordered inventory of :class:`Symbol` records, each carrying
the metadata downstream stages need -- line range, signature, docstring,
decorators, base classes, and (for methods) the enclosing class.

Extraction is currently Python-specific because the node types it inspects
(``function_definition``, ``class_definition``, ``import_from_statement`` ...)
belong to the tree-sitter Python grammar. Parsing itself is delegated to the
language-agnostic :class:`~src.reporag.ingestion.parser.ASTParser` so adding
another language later only means adding its extraction rules here.

Typical usage::

    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    extractor = SymbolExtractor()
    for symbol in extractor.extract_from_file("examples/sample_repo/app.py"):
        print(f"{symbol.type}: {symbol.name} [{symbol.start_line}-{symbol.end_line}]")
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node

from src.reporag.config import settings
from src.reporag.ingestion.parser import ASTParser, UnsupportedLanguageError

logger = logging.getLogger(__name__)

# Symbol.type values. Plain strings (not an enum) so the issue's
# ``print(f"{s.type}: ...")`` renders cleanly and equality checks read naturally.
FUNCTION = "function"
METHOD = "method"
CLASS = "class"
IMPORT = "import"


@dataclass(frozen=True)
class Symbol:
    """A single extracted code entity.

    Line numbers are 1-based and inclusive, matching how editors count lines and
    the ``[start_line-end_line]`` convention used across the ingestion pipeline.
    For a decorated definition the range covers the decorators too. Fields that
    do not apply to a given ``type`` keep their empty defaults (an import has no
    ``bases``; a function has no ``methods``).
    """

    name: str
    type: str
    start_line: int
    end_line: int
    file_path: str | None = None
    signature: str = ""
    docstring: str | None = None
    decorators: tuple[str, ...] = field(default_factory=tuple)
    parent_class: str | None = None
    return_type: str | None = None
    bases: tuple[str, ...] = field(default_factory=tuple)
    methods: tuple[str, ...] = field(default_factory=tuple)
    is_async: bool = False
    module: str | None = None

    @property
    def is_method(self) -> bool:
        """True when this symbol is a method bound to an enclosing class."""
        return self.type == METHOD


class SymbolExtractor:
    """Extract structured :class:`Symbol` records from Python source.

    A single instance is reusable across files; it owns one :class:`ASTParser`
    whose per-language grammars are cached. Extraction tolerates syntax errors:
    it walks whatever nodes the (possibly partial) tree exposes rather than
    raising on malformed input.
    """

    def __init__(self, parser: ASTParser | None = None) -> None:
        """Initialize with an optional shared :class:`ASTParser`."""
        self._parser = parser or ASTParser()

    def extract(
        self,
        source: str | bytes,
        language: str = "python",
        file_path: str | None = None,
    ) -> list[Symbol]:
        """Parse ``source`` and return its symbols in source order.

        Args:
            source: Python source as ``str`` or ``bytes``.
            language: Source language. Only ``"python"`` is supported today.
            file_path: Optional path recorded on every returned symbol.

        Returns:
            A flat, source-ordered list of :class:`Symbol` records, including
            nested functions and methods.

        Raises:
            UnsupportedLanguageError: If ``language`` is not Python.
        """
        if language.lower().strip() != "python":
            raise UnsupportedLanguageError(
                f"Symbol extraction currently supports only Python, not "
                f"{language!r}."
            )

        tree = self._parser.parse(source, language="python")
        if self._parser.has_errors(tree):
            logger.debug(
                "Extracting symbols from %s with syntax errors; results are partial.",
                file_path or "<string>",
            )
        return self._collect(tree.root_node, file_path, parent_class=None)

    def extract_from_file(
        self, file_path: str | Path, language: str | None = None
    ) -> list[Symbol]:
        """Read a file from disk and extract its symbols.

        Args:
            file_path: Path to the source file.
            language: Optional language override. When omitted it is inferred
                from the file extension via ``settings.extension_map`` (the same
                mapping the cloner and parser use).

        Returns:
            A flat, source-ordered list of :class:`Symbol` records.

        Raises:
            UnsupportedLanguageError: If the language cannot be inferred or is
                not Python.
        """
        path = Path(file_path)
        if language is None:
            language = settings.extension_map.get(path.suffix.lower())
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for {path.name!r} from its extension."
                )
        return self.extract(
            path.read_bytes(), language=language, file_path=str(file_path)
        )

    # ----- Traversal -----

    def _collect(
        self, parent: Node, file_path: str | None, parent_class: str | None
    ) -> list[Symbol]:
        """Dispatch over the direct named children of ``parent``.

        ``parent`` is a ``module`` node at the top level or a ``block`` node when
        recursing into a function or class body. Only the four symbol-bearing
        node kinds produce records; everything else (assignments, calls, control
        flow) is ignored here.
        """
        symbols: list[Symbol] = []
        for child in parent.named_children:
            symbols.extend(self._dispatch(child, file_path, parent_class))
        return symbols

    def _dispatch(
        self, node: Node, file_path: str | None, parent_class: str | None
    ) -> list[Symbol]:
        """Turn a single statement node into zero or more symbols."""
        node_type = node.type
        if node_type in ("import_statement", "import_from_statement"):
            return self._extract_imports(node, file_path)
        if node_type == "decorated_definition":
            decorators = self._decorators(node)
            definition = node.child_by_field_name("definition")
            if definition is None:
                return []
            return self._extract_definition(
                definition, file_path, parent_class, decorators, outer=node
            )
        if node_type == "function_definition":
            return self._extract_definition(
                node, file_path, parent_class, (), outer=node
            )
        if node_type == "class_definition":
            return self._extract_definition(
                node, file_path, parent_class, (), outer=node
            )
        return []

    def _extract_definition(
        self,
        node: Node,
        file_path: str | None,
        parent_class: str | None,
        decorators: tuple[str, ...],
        outer: Node,
    ) -> list[Symbol]:
        """Route a (possibly decorated) definition to its specific extractor.

        ``outer`` is the decorated_definition when decorators are present, so the
        reported line range spans the decorators; otherwise it is ``node`` itself.
        """
        if node.type == "function_definition":
            return self._extract_function(
                node, file_path, parent_class, decorators, outer
            )
        if node.type == "class_definition":
            return self._extract_class(node, file_path, parent_class, decorators, outer)
        return []

    def _extract_function(
        self,
        node: Node,
        file_path: str | None,
        parent_class: str | None,
        decorators: tuple[str, ...],
        outer: Node,
    ) -> list[Symbol]:
        """Build a function/method symbol, then recurse into its body."""
        name = self._field_text(node, "name") or "<anonymous>"
        params = self._normalize(self._field_text(node, "parameters") or "()")
        return_type = self._field_text(node, "return_type")
        is_async = any(child.type == "async" for child in node.children)
        body = node.child_by_field_name("body")

        prefix = "async def " if is_async else "def "
        signature = f"{prefix}{name}{params}"
        if return_type:
            signature += f" -> {return_type}"
        signature += ":"

        symbol = Symbol(
            name=name,
            type=METHOD if parent_class else FUNCTION,
            start_line=outer.start_point[0] + 1,
            end_line=outer.end_point[0] + 1,
            file_path=file_path,
            signature=signature,
            docstring=self._docstring(body),
            decorators=decorators,
            parent_class=parent_class,
            return_type=return_type,
            is_async=is_async,
        )

        symbols = [symbol]
        if body is not None:
            # A nested definition is a local function, not a method, so the
            # enclosing-class context is cleared as we descend into the body.
            symbols.extend(self._collect(body, file_path, parent_class=None))
        return symbols

    def _extract_class(
        self,
        node: Node,
        file_path: str | None,
        parent_class: str | None,
        decorators: tuple[str, ...],
        outer: Node,
    ) -> list[Symbol]:
        """Build a class symbol, then recurse into its body for members."""
        name = self._field_text(node, "name") or "<anonymous>"
        bases = self._base_classes(node)
        body = node.child_by_field_name("body")

        signature = f"class {name}"
        if bases:
            signature += "(" + ", ".join(bases) + ")"
        signature += ":"

        symbol = Symbol(
            name=name,
            type=CLASS,
            start_line=outer.start_point[0] + 1,
            end_line=outer.end_point[0] + 1,
            file_path=file_path,
            signature=signature,
            docstring=self._docstring(body),
            decorators=decorators,
            parent_class=parent_class,
            bases=bases,
            methods=self._method_names(body),
        )

        symbols = [symbol]
        if body is not None:
            symbols.extend(self._collect(body, file_path, parent_class=name))
        return symbols

    def _extract_imports(self, node: Node, file_path: str | None) -> list[Symbol]:
        """Expand an import statement into one symbol per imported name.

        Handles ``import x``, ``import x as y``, ``from m import a, b``,
        ``from m import *``, and relative imports such as ``from . import x``.
        """
        is_from = node.type == "import_from_statement"
        module = self._field_text(node, "module_name") if is_from else None
        signature = self._normalize(self._text(node))
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        symbols: list[Symbol] = []
        for name_node in self._imported_name_nodes(node):
            bound, source_module = self._resolve_import(name_node, module, is_from)
            symbols.append(
                Symbol(
                    name=bound,
                    type=IMPORT,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=file_path,
                    signature=signature,
                    module=source_module,
                )
            )
        return symbols

    # ----- Node helpers -----

    @staticmethod
    def _imported_name_nodes(node: Node) -> list[Node]:
        """Return the nodes naming each thing an import statement brings in.

        These are the children in the ``name`` field plus any ``wildcard_import``
        (``*``); the ``from`` module itself lives in ``module_name`` and is
        deliberately excluded.
        """
        names: list[Node] = []
        for index, child in enumerate(node.children):
            is_named_field = node.field_name_for_child(index) == "name"
            if is_named_field or child.type == "wildcard_import":
                names.append(child)
        return names

    def _resolve_import(
        self, name_node: Node, module: str | None, is_from: bool
    ) -> tuple[str, str | None]:
        """Resolve one imported name node to ``(bound_name, source_module)``."""
        if name_node.type == "wildcard_import":
            return "*", module
        if name_node.type == "aliased_import":
            original = self._field_text(name_node, "name") or ""
            alias = self._field_text(name_node, "alias") or original
            return alias, (module if is_from else original)
        # Plain dotted_name, e.g. ``os`` or ``a.b.c``.
        bound = self._text(name_node)
        return bound, (module if is_from else bound)

    def _base_classes(self, class_node: Node) -> tuple[str, ...]:
        """Return positional base-class expressions, skipping keyword args."""
        superclasses = class_node.child_by_field_name("superclasses")
        if superclasses is None:
            return ()
        return tuple(
            self._normalize(self._text(child))
            for child in superclasses.named_children
            if child.type != "keyword_argument"
        )

    def _method_names(self, body: Node | None) -> tuple[str, ...]:
        """Return the names of methods defined directly in a class body."""
        if body is None:
            return ()
        names: list[str] = []
        for child in body.named_children:
            func = child
            if child.type == "decorated_definition":
                func = child.child_by_field_name("definition")
            if func is not None and func.type == "function_definition":
                name = self._field_text(func, "name")
                if name:
                    names.append(name)
        return tuple(names)

    def _decorators(self, decorated_node: Node) -> tuple[str, ...]:
        """Return each decorator expression text with its leading ``@`` removed."""
        decorators: list[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                decorators.append(self._text(child).lstrip("@").strip())
        return tuple(decorators)

    def _docstring(self, body: Node | None) -> str | None:
        """Return the cleaned docstring if the body opens with a string literal."""
        if body is None or not body.named_children:
            return None
        first = body.named_children[0]
        if first.type != "expression_statement" or not first.named_children:
            return None
        string_node = first.named_children[0]
        if string_node.type != "string":
            return None
        contents = "".join(
            self._text(part)
            for part in string_node.children
            if part.type == "string_content"
        )
        cleaned = inspect.cleandoc(contents).strip()
        return cleaned or None

    @staticmethod
    def _field_text(node: Node, field_name: str) -> str | None:
        """Return the decoded text of a named field, or ``None`` if absent."""
        child = node.child_by_field_name(field_name)
        return SymbolExtractor._text(child) if child is not None else None

    @staticmethod
    def _text(node: Node) -> str:
        """Decode a node's source bytes to text, replacing invalid sequences."""
        raw = node.text if node.text is not None else b""
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse internal whitespace so multi-line headers read on one line."""
        return " ".join(text.split())


def extract_symbols(
    source: str | bytes, language: str = "python", file_path: str | None = None
) -> list[Symbol]:
    """Convenience wrapper that extracts symbols with a throwaway extractor."""
    return SymbolExtractor().extract(source, language=language, file_path=file_path)

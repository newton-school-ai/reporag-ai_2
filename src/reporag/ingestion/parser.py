"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python (extensible
to JS/TS). Handles parse errors gracefully with partial ASTs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree

from src.reporag.config import settings


class UnsupportedLanguageError(Exception):
    """Raised when no parser is registered for a language."""


class ParseError(Exception):
    """Raised when a source file cannot be parsed at all."""


@dataclass
class NodeData:
    """Structured information extracted from an AST node."""

    type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_named: bool
    is_error: bool


def _load_python_grammar() -> object:
    import tree_sitter_python as ts_python

    return ts_python.language()


def _load_javascript_grammar() -> object:
    import tree_sitter_javascript as ts_javascript

    return ts_javascript.language()


_GRAMMAR_LOADERS: dict[str, Callable[[], object]] = {
    "python": _load_python_grammar,
    "javascript": _load_javascript_grammar,
}


class ASTParser:

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, language: str) -> Parser:
        lang = language.lower().strip()
        if lang in self._parsers:
            return self._parsers[lang]

        loader = _GRAMMAR_LOADERS.get(lang)
        if loader is None:
            raise UnsupportedLanguageError(
                f"No parser registered for language: {language}"
            )

        try:
            ts_language = Language(loader())
        except ImportError as e:
            raise UnsupportedLanguageError(
                f"Grammar for {language!r} is not installed: {e}"
            ) from e

        parser = Parser()
        parser.language = ts_language
        self._parsers[lang] = parser
        return parser

    def parse(
        self,
        source_code: str | bytes,
        language: str = "python",
    ) -> Tree:
        parser = self._get_parser(language)

        source_bytes = (
            source_code.encode("utf-8") if isinstance(source_code, str) else source_code
        )

        return parser.parse(source_bytes)

    def walk(self, tree: Tree, named_only: bool = False) -> Iterator[NodeData]:
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if not named_only or node.is_named:
                yield self._to_node_data(node)
            stack.extend(reversed(node.children))

    def has_errors(self, tree: Tree) -> bool:
        return tree.root_node.has_error

    def find_errors(self, tree: Tree) -> list[NodeData]:
        return [
            self._to_node_data(n)
            for n in self._iter_native(tree.root_node)
            if n.is_error or n.is_missing
        ]

    @staticmethod
    def _to_node_data(node: Node) -> NodeData:
        """Convert a native tree-sitter Node into structured NodeData."""
        raw = node.text if node.text is not None else b""
        return NodeData(
            type=node.type,
            text=raw.decode("utf-8", errors="replace"),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            is_named=node.is_named,
            is_error=node.is_error or node.is_missing,
        )

    @staticmethod
    def _iter_native(node: Node) -> Iterator[Node]:
        """Iteratively yield node and its descendants in pre-order.

        Uses an explicit stack rather than recursion so deeply nested
        trees cannot hit Python's recursion limit.
        """
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    def parse_file(self, file_path: str | Path, language: str | None = None) -> Tree:
        path = Path(file_path)
        if language is None:
            language = settings.extension_map.get(path.suffix.lower())
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for {path.name!r}"
                )
        return self.parse(path.read_text(encoding="utf-8"), language)

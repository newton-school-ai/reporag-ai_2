"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python (extensible
to JS/TS). Handles parse errors gracefully with partial ASTs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
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


class ASTParser:
    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._register("python", tspython.language())

    def _register(
        self,
        language: str,
        language_obj: Any,
    ) -> None:
        """Register a tree-sitter parser for a language."""
        parser = Parser()
        parser.language = Language(language_obj)
        self._parsers[language] = parser

    def parse(
        self,
        source_code: str | bytes,
        language: str = "python",
    ) -> Tree:
        if language not in self._parsers:
            raise UnsupportedLanguageError(
                f"No parser registered for language: {language}"
            )

        parser = self._parsers[language]

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

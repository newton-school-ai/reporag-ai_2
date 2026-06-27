"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python (extensible
to JS/TS). Handles parse errors gracefully with partial ASTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

from src.reporag.config import settings


class UnsupportedLanguageError(Exception):
    """Raised when no parser is registered for a language."""


@dataclass
class NodeData:
    """Structured information extracted from an AST node."""

    type: str
    text: str
    start_line: int
    end_line: int


@dataclass
class ParseResult:
    tree: Tree
    language: str
    has_errors: bool
    error_count: int
    nodes: list[NodeData]


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
        source_code: str,
        language: str = "python",
    ) -> ParseResult:
        """Parse source code into a tree-sitter AST."""

        if language not in self._parsers:
            raise UnsupportedLanguageError(
                f"No parser registered for language: {language}"
            )

        parser = self._parsers[language]

        tree = parser.parse(source_code.encode("utf-8"))
        errors = self._count_errors(tree.root_node)

        nodes = self._extract_nodes(
            tree.root_node,
            source_code,
        )

        return ParseResult(
            tree=tree,
            language=language,
            has_errors=errors > 0,
            error_count=errors,
            nodes=nodes,
        )

    def _count_errors(self, node: Node) -> int:
        """Recursively count ERROR nodes in the AST."""

        count = 1 if node.type == "ERROR" else 0

        for child in node.children:
            count += self._count_errors(child)

        return count

    def _extract_nodes(
        self,
        node: Node,
        source_code: str,
    ) -> list[NodeData]:
        """Recursively extract structured node data from the AST."""

        nodes = []

        if node.is_named:
            nodes.append(
                NodeData(
                    type=node.type,
                    text=source_code[node.start_byte : node.end_byte],
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                )
            )

        for child in node.children:
            nodes.extend(self._extract_nodes(child, source_code))

        return nodes

    def parse_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> ParseResult:
        """Parse a source file into a tree-sitter AST."""

        path = Path(file_path)

        source_code = path.read_text(encoding="utf-8")

        if language is None:
            extension = path.suffix.lower()

            try:
                language = settings.extension_map[extension]
            except KeyError as exc:
                raise UnsupportedLanguageError(
                    f"Unsupported file extension: {extension}"
                ) from exc

        return self.parse(source_code, language)

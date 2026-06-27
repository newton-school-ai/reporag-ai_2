"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python (extensible
to JS/TS). Handles parse errors gracefully with partial ASTs.
"""

# TODO: Implement in Issue 6
# - Load tree-sitter grammar for target language
# - Parse source string -> tree-sitter Tree
# - Walk tree, return structured node data (type, text, start/end lines)
# - Handle syntax errors (return partial AST, flag errors)
# - Language-agnostic interface: Parser.parse(source, language)


from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree


@dataclass
class ParseResult:
    tree: Tree
    language: str
    has_errors: bool
    error_count: int


class ASTParser:
    def __init__(self):
        self._parsers: dict[str, Parser] = {}
        self._register("python", tspython.language())

    def _register(self, language: str, language_obj) -> None:
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
            raise ValueError(f"No parser registered for language: {language}")

        parser = self._parsers[language]

        tree = parser.parse(source_code.encode("utf-8"))

        return ParseResult(
            tree=tree,
            language=language,
            has_errors=False,
            error_count=0,
        )

"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python and JavaScript out
of the box and is extensible to any language with a tree-sitter grammar.

tree-sitter is used (instead of Python's built-in ``ast`` module) because it is
fast, incremental, error-tolerant, and language-agnostic: a single interface
parses every supported language, broken files yield a *partial* tree with
explicit ERROR / MISSING nodes rather than raising, and byte/point offsets are
preserved so comments and whitespace can be located precisely.

Typical usage::

    from src.reporag.ingestion.parser import ASTParser

    parser = ASTParser()
    tree = parser.parse("def hello():\\n    return 42\\n", language="python")
    for node in parser.walk(tree, named_only=True):
        print(node.type, node.start_line, node.end_line)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree

from src.reporag.config import settings

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when a source file cannot be parsed at all."""


class UnsupportedLanguageError(ParseError):
    """Raised when parsing is requested for a language with no loaded grammar."""


def _load_python_grammar() -> object:
    """Return the tree-sitter Language capsule for Python (imported lazily)."""
    import tree_sitter_python as ts_python

    return ts_python.language()


def _load_javascript_grammar() -> object:
    """Return the tree-sitter Language capsule for JavaScript (imported lazily)."""
    import tree_sitter_javascript as ts_javascript

    return ts_javascript.language()


# Language name -> zero-arg loader returning a tree-sitter Language capsule.
# Grammars are imported lazily so a missing optional grammar only fails when its
# language is actually requested. Register additional languages here.
_GRAMMAR_LOADERS: dict[str, Callable[[], object]] = {
    "python": _load_python_grammar,
    "javascript": _load_javascript_grammar,
}


@dataclass(frozen=True)
class ASTNode:
    """Structured, framework-independent view of a single tree-sitter node.

    Line numbers are 1-based (matching how editors and humans count lines);
    column numbers are 0-based byte offsets within their line (matching
    tree-sitter points). ``text`` is the exact source slice the node spans.
    """

    type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_named: bool = True
    is_error: bool = False


class ASTParser:
    """A language-agnostic tree-sitter parser.

    Lazily loads and caches one tree-sitter ``Parser`` per language, parses
    source into a tree-sitter ``Tree``, and offers helpers to walk that tree as
    structured, framework-independent :class:`ASTNode` data. A single instance
    is safe to reuse across many files of mixed languages.
    """

    def __init__(self) -> None:
        """Initialize the parser with an empty per-language cache."""
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        # File extension -> language name, sourced from central settings so the
        # mapping stays consistent with the cloner's file discovery.
        self.extension_map: dict[str, str] = dict(settings.extension_map)

    @property
    def supported_languages(self) -> list[str]:
        """Sorted list of language names this parser can load a grammar for."""
        return sorted(_GRAMMAR_LOADERS)

    def parse(self, source: str | bytes, language: str = "python") -> Tree:
        """Parse ``source`` into a tree-sitter ``Tree``.

        Args:
            source: Source code as ``str`` or ``bytes``. ``str`` is encoded as
                UTF-8 so byte/point offsets stay consistent.
            language: Target language name (e.g. ``"python"``). Case-insensitive.

        Returns:
            A tree-sitter ``Tree``. For syntactically broken input this is a
            *partial* tree whose root reports ``has_error`` -- it is never an
            exception.

        Raises:
            UnsupportedLanguageError: If no grammar is registered for ``language``.
            ParseError: If the underlying parser fails unexpectedly.
        """
        parser = self._get_parser(language)
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source

        try:
            tree = parser.parse(source_bytes)
        except Exception as e:  # pragma: no cover - tree-sitter rarely raises here
            raise ParseError(f"Failed to parse {language} source: {e}") from e

        if tree.root_node.has_error:
            logger.debug(
                "Parsed %s source with syntax errors; returning partial AST.",
                language,
            )
        return tree

    def parse_file(self, file_path: str | Path, language: str | None = None) -> Tree:
        """Read a file from disk and parse it.

        Args:
            file_path: Path to the source file.
            language: Optional language override. When omitted, the language is
                inferred from the file extension via the configured
                ``extension_map``.

        Returns:
            A tree-sitter ``Tree`` (partial if the file has syntax errors).

        Raises:
            UnsupportedLanguageError: If the language cannot be inferred or has
                no registered grammar.
        """
        path = Path(file_path)
        if language is None:
            language = self.extension_map.get(path.suffix.lower())
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for {path.name!r} from its extension."
                )
        return self.parse(path.read_bytes(), language=language)

    def walk(
        self, tree_or_node: Tree | Node, named_only: bool = False
    ) -> Iterator[ASTNode]:
        """Yield every node under ``tree_or_node`` in pre-order as ``ASTNode``.

        Args:
            tree_or_node: A parsed ``Tree`` or any ``Node`` to start from.
            named_only: When True, skip anonymous nodes (punctuation, keywords)
                and yield only named grammar nodes.

        Yields:
            One :class:`ASTNode` per visited node, parents before children.
        """
        for node in self._iter_native(self._root_of(tree_or_node)):
            if named_only and not node.is_named:
                continue
            yield self.to_ast_node(node)

    def find_errors(self, tree_or_node: Tree | Node) -> list[ASTNode]:
        """Return the ERROR and MISSING nodes in a (possibly partial) tree."""
        return [
            self.to_ast_node(node)
            for node in self._iter_native(self._root_of(tree_or_node))
            if node.is_error or node.is_missing
        ]

    def has_errors(self, tree_or_node: Tree | Node) -> bool:
        """Return True if the tree contains any syntax errors."""
        return self._root_of(tree_or_node).has_error

    @staticmethod
    def to_ast_node(node: Node) -> ASTNode:
        """Convert a native tree-sitter ``Node`` into a structured ``ASTNode``."""
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        raw_text = node.text if node.text is not None else b""
        return ASTNode(
            type=node.type,
            text=raw_text.decode("utf-8", errors="replace"),
            start_line=start_row + 1,
            end_line=end_row + 1,
            start_col=start_col,
            end_col=end_col,
            is_named=node.is_named,
            is_error=node.is_error or node.is_missing,
        )

    def _get_parser(self, language: str) -> Parser:
        """Return a cached ``Parser`` for ``language``, building it on first use."""
        lang = language.lower().strip()
        if lang in self._parsers:
            return self._parsers[lang]

        loader = _GRAMMAR_LOADERS.get(lang)
        if loader is None:
            raise UnsupportedLanguageError(
                f"Unsupported language {language!r}. Supported languages: "
                f"{', '.join(self.supported_languages)}."
            )

        try:
            ts_language = Language(loader())
        except ImportError as e:
            raise UnsupportedLanguageError(
                f"Grammar for {language!r} is not installed: {e}"
            ) from e

        parser = Parser(ts_language)
        self._languages[lang] = ts_language
        self._parsers[lang] = parser
        return parser

    @staticmethod
    def _root_of(tree_or_node: Tree | Node) -> Node:
        """Return the root ``Node`` for a ``Tree`` or pass a ``Node`` through."""
        return (
            tree_or_node.root_node if isinstance(tree_or_node, Tree) else tree_or_node
        )

    @staticmethod
    def _iter_native(node: Node) -> Iterator[Node]:
        """Iteratively yield ``node`` and its descendants in pre-order.

        Uses an explicit stack rather than recursion so deeply nested trees
        cannot hit Python's recursion limit.
        """
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            yield current
            # Push children reversed so they pop left-to-right (source order).
            stack.extend(reversed(current.children))

"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python and JavaScript
out of the box (extensible to any language via the grammar registry).
Handles parse errors gracefully: broken source returns a partial AST and
never raises; ERROR/MISSING nodes can be surfaced via has_errors() and
find_errors().

tree-sitter is preferred over Python's built-in ast module because it is:
- Fast and incremental
- Error-tolerant (partial ASTs for broken files)
- Language-agnostic (one interface for every supported language)
- Byte/point-accurate (preserves start/end offsets for every node)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree

from src.reporag.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grammar registry
# ---------------------------------------------------------------------------


def _load_python() -> Language:
    import tree_sitter_python as _tspy  # noqa: PLC0415

    return Language(_tspy.language())


def _load_javascript() -> Language:
    import tree_sitter_javascript as _tsjs  # noqa: PLC0415

    return Language(_tsjs.language())


# Maps canonical language name -> loader callable.
# Add new languages here as a one-line entry.
_GRAMMAR_LOADERS: dict[str, object] = {
    "python": _load_python,
    "javascript": _load_javascript,
    "typescript": _load_javascript,  # fallback to JS grammar for TS
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when a parse operation fails for a non-recoverable reason."""

    pass


class UnsupportedLanguageError(ParseError):
    """Raised when no grammar is registered for the requested language."""

    pass


# ---------------------------------------------------------------------------
# ASTNode dataclass
# ---------------------------------------------------------------------------


@dataclass
class ASTNode:
    """Structured metadata for a single tree-sitter AST node.

    Attributes:
        type: The grammar node type (e.g. ``"function_definition"``).
        text: The source text captured by this node (decoded from bytes).
        start_line: 1-based line number of the node's first character.
        end_line: 1-based line number of the node's last character.
        start_col: 0-based column of the node's first character.
        end_col: 0-based column of the node's last character.
        is_named: Whether this is a named node in the grammar.
        children: Direct child ASTNodes (empty for leaf nodes).
    """

    type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_named: bool
    children: list[ASTNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ASTParser
# ---------------------------------------------------------------------------


class ASTParser:
    """Language-agnostic tree-sitter AST parser.

    A single instance can be reused across an entire mixed-language
    repository.  Parsers are created lazily on first use and cached.
    """

    def __init__(self) -> None:
        """Initialise the parser cache."""
        self._parsers: dict[str, Parser] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser(self, language: str) -> Parser:
        """Return a cached Parser for *language*, creating one if needed.

        Args:
            language: Canonical lower-case language name.

        Returns:
            A ready-to-use ``tree_sitter.Parser``.

        Raises:
            UnsupportedLanguageError: If no grammar is registered.
        """
        if language not in self._parsers:
            loader = _GRAMMAR_LOADERS.get(language)
            if loader is None:
                raise UnsupportedLanguageError(
                    f"No tree-sitter grammar registered for language '{language}'. "
                    f"Supported: {sorted(_GRAMMAR_LOADERS)}"
                )
            try:
                lang_obj = loader()  # type: ignore[operator]
                self._parsers[language] = Parser(lang_obj)
            except Exception as exc:
                raise ParseError(
                    f"Failed to load tree-sitter grammar for '{language}': {exc}"
                ) from exc
        return self._parsers[language]

    @staticmethod
    def _to_bytes(source: str | bytes, encoding: str = "utf-8") -> bytes:
        """Ensure *source* is bytes."""
        return source.encode(encoding) if isinstance(source, str) else source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        source: str | bytes,
        language: str = "python",
    ) -> Tree:
        """Parse *source* and return the native tree-sitter ``Tree``.

        The returned ``Tree`` object is exactly what tree-sitter produces,
        so ``tree.root_node.children`` works as shown in the issue snippet.
        Broken source is silently tolerated: the tree may contain ERROR or
        MISSING nodes which can be detected via :meth:`has_errors` and
        :meth:`find_errors`.

        Args:
            source: Source code as ``str`` or ``bytes``.
            language: Language name (case-insensitive). Defaults to
                ``"python"``.

        Returns:
            A ``tree_sitter.Tree`` rooted at a ``module`` / ``program``
            node (depending on the grammar).

        Raises:
            UnsupportedLanguageError: If the grammar is not registered.
            ParseError: If the grammar itself fails to load.
        """
        lang = language.lower().strip()
        parser = self._get_parser(lang)
        src_bytes = self._to_bytes(source)
        return parser.parse(src_bytes)

    def parse_file(
        self,
        path: str | Path,
        language: str | None = None,
    ) -> Tree:
        """Parse a source file on disk.

        The language is inferred from the file extension via
        ``settings.extension_map`` (the same mapping used by the Issue-5
        cloner), so behaviour is consistent across the ingestion pipeline.

        Args:
            path: Absolute or relative path to the source file.
            language: Override the inferred language.

        Returns:
            A ``tree_sitter.Tree``.

        Raises:
            ParseError: If the file cannot be read or the language is
                unsupported.
        """
        fpath = Path(path)
        if language is None:
            ext = fpath.suffix.lower()
            language = settings.extension_map.get(ext)
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for extension '{ext}'. "
                    f"Pass language= explicitly or add it to settings.extension_map."
                )
        try:
            source = fpath.read_bytes()
        except OSError as exc:
            raise ParseError(f"Cannot read file '{fpath}': {exc}") from exc
        return self.parse(source, language=language)

    def walk(
        self,
        tree: Tree,
        named_only: bool = False,
    ) -> Generator[ASTNode, None, None]:
        """Yield :class:`ASTNode` for every node in *tree* (iterative DFS).

        Iterative depth-first traversal avoids hitting Python's recursion
        limit on deeply nested source files.

        Args:
            tree: A ``tree_sitter.Tree`` returned by :meth:`parse`.
            named_only: If ``True``, skip anonymous (punctuation/keyword)
                nodes and yield only named grammar nodes.

        Yields:
            :class:`ASTNode` instances in DFS pre-order.
        """
        stack: list[Node] = [tree.root_node]
        while stack:
            node: Node = stack.pop()
            if named_only and not node.is_named:
                # Still recurse into unnamed nodes so we don't skip children
                stack.extend(reversed(node.children))
                continue

            raw_text = node.text
            text = raw_text.decode("utf-8", errors="replace") if raw_text else ""

            yield ASTNode(
                type=node.type,
                text=text,
                start_line=node.start_point[0] + 1,  # 1-based
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1],
                is_named=node.is_named,
            )
            # Push children in reverse order so left-most is processed first
            stack.extend(reversed(node.children))

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def has_errors(self, tree: Tree) -> bool:
        """Return ``True`` if the tree contains any ERROR or MISSING nodes.

        Args:
            tree: A ``tree_sitter.Tree``.

        Returns:
            ``True`` if at least one error node exists.
        """
        return tree.root_node.has_error

    def find_errors(self, tree: Tree) -> list[ASTNode]:
        """Return all ERROR and MISSING nodes in *tree*.

        Args:
            tree: A ``tree_sitter.Tree``.

        Returns:
            A list of :class:`ASTNode` with ``type`` equal to ``"ERROR"``
            or ``"MISSING"``.
        """
        return [node for node in self.walk(tree) if node.type in ("ERROR", "MISSING")]

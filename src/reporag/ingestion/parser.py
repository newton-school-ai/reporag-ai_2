"""Tree-sitter AST parser for Python (and JavaScript) source files.

Parses source files into tree-sitter ASTs using a language-agnostic interface.
Handles parse errors gracefully by returning partial ASTs with an ``has_error``
flag rather than raising exceptions. Supports Python and JavaScript out of the
box; additional grammars can be registered at runtime via
:meth:`ASTParser.register_language`.

Usage::

    from reporag.ingestion.parser import ASTParser, ParseResult

    parser = ASTParser()

    # Parse a Python snippet
    result = parser.parse('def hello():\\n    return 42\\n', language='python')
    print(result.root_node.type)           # 'module'
    print(result.has_error)               # False

    # Walk the tree
    for node_info in parser.walk(result):
        print(node_info)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported language registry
# ---------------------------------------------------------------------------

#: Registry mapping canonical language name -> tree_sitter Language object.
#: Populated lazily on first use so import time stays fast.
_LANGUAGE_REGISTRY: dict[str, Language] = {}


def _load_builtin_languages() -> None:
    """Populate _LANGUAGE_REGISTRY with built-in grammars."""
    # Python
    try:
        import tree_sitter_python as _tspy  # type: ignore[import]

        _LANGUAGE_REGISTRY["python"] = Language(_tspy.language())
    except ImportError:  # pragma: no cover
        logger.warning(
            "tree-sitter-python grammar not installed; Python parsing unavailable."
        )

    # JavaScript
    try:
        import tree_sitter_javascript as _tsjs  # type: ignore[import]

        _LANGUAGE_REGISTRY["javascript"] = Language(_tsjs.language())
    except ImportError:  # pragma: no cover
        logger.warning(
            "tree-sitter-javascript grammar not installed; JavaScript parsing unavailable."
        )

    # TypeScript (optional -- not in default requirements)
    try:
        import tree_sitter_typescript as _tsts  # type: ignore[import]

        _LANGUAGE_REGISTRY["typescript"] = Language(_tsts.language_typescript())
        _LANGUAGE_REGISTRY["tsx"] = Language(_tsts.language_tsx())
    except ImportError:
        pass  # TypeScript is optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeInfo:
    """Structured representation of a single tree-sitter node.

    Attributes:
        node_type:   Grammar node type string (e.g. ``"function_definition"``).
        text:        UTF-8 decoded source text covered by this node.
        start_line:  0-indexed start row.
        end_line:    0-indexed end row (inclusive).
        start_col:   0-indexed start column.
        end_col:     0-indexed end column (exclusive, on ``end_line``).
        is_error:    ``True`` when this node is an ERROR node.
        is_named:    ``True`` when the node type is named (vs. anonymous).
    """

    node_type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_error: bool
    is_named: bool

    @classmethod
    def from_node(cls, node: Node) -> NodeInfo:
        """Build a :class:`NodeInfo` from a raw tree-sitter :class:`Node`."""
        raw_text: bytes = node.text if node.text is not None else b""
        return cls(
            node_type=node.type,
            text=raw_text.decode("utf-8", errors="replace"),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            is_error=node.is_error,
            is_named=node.is_named,
        )


@dataclass
class ParseResult:
    """The output of a single :meth:`ASTParser.parse` call.

    Attributes:
        tree:      Raw tree-sitter :class:`Tree` (access ``root_node`` etc.).
        language:  Canonical language name used for parsing.
        has_error: ``True`` when the source contained syntax errors.  The
                   ``tree`` is still valid (partial AST); callers should
                   decide whether to proceed or discard.
        source:    Original source bytes used for parsing.
    """

    tree: Tree
    language: str
    has_error: bool
    source: bytes

    @property
    def root_node(self) -> Node:
        """Convenience alias for ``tree.root_node``."""
        return self.tree.root_node


# ---------------------------------------------------------------------------
# ASTParser
# ---------------------------------------------------------------------------


class ParserError(ValueError):
    """Raised for unrecoverable parser configuration errors.

    This is intentionally *not* raised for source syntax errors -- those are
    captured in :attr:`ParseResult.has_error` instead.
    """


class ASTParser:
    """Language-agnostic tree-sitter AST parser.

    The parser lazily initialises one :class:`tree_sitter.Parser` per
    language the first time it is requested, so multi-language workloads
    incur no extra startup cost.

    Args:
        languages: Optional override mapping of ``language_name -> Language``.
            When supplied, *replaces* the built-in registry for this instance.
            Useful for testing custom grammars.

    Examples:
        >>> parser = ASTParser()
        >>> result = parser.parse('def hello():\\n    return 42\\n', language='python')
        >>> result.root_node.type
        'module'
        >>> result.has_error
        False
    """

    def __init__(self, languages: dict[str, Language] | None = None) -> None:
        if not _LANGUAGE_REGISTRY:
            _load_builtin_languages()

        # Per-instance language map (allows injection for testing)
        self._languages: dict[str, Language] = (
            dict(languages) if languages is not None else _LANGUAGE_REGISTRY
        )
        # Cache of tree_sitter.Parser objects keyed by language name
        self._parsers: dict[str, Parser] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """Sorted list of language names this instance can parse."""
        return sorted(self._languages)

    def register_language(self, name: str, language: Language) -> None:
        """Register an additional grammar at runtime.

        Args:
            name:     Canonical language name (e.g. ``"rust"``).
            language: Initialised :class:`tree_sitter.Language` object.
        """
        self._languages[name] = language
        # Invalidate any cached parser for this language
        self._parsers.pop(name, None)
        logger.debug("Registered language '%s'", name)

    def parse(self, source: str | bytes, *, language: str) -> ParseResult:
        """Parse *source* and return a :class:`ParseResult`.

        The parser *never* raises for syntax errors; instead it sets
        :attr:`ParseResult.has_error` to ``True`` and returns the partial
        (best-effort) tree produced by tree-sitter's error-recovery algorithm.

        Args:
            source:   Source code as a ``str`` (UTF-8 encoded internally) or
                      ``bytes``.
            language: Canonical language name, e.g. ``"python"``.

        Returns:
            A :class:`ParseResult` containing the AST tree, language,
            ``has_error`` flag, and original source bytes.

        Raises:
            ParserError: If *language* is not registered.
        """
        if language not in self._languages:
            available = ", ".join(sorted(self._languages)) or "(none)"
            raise ParserError(
                f"Unsupported language '{language}'. "
                f"Available languages: {available}."
            )

        source_bytes: bytes = (
            source.encode("utf-8") if isinstance(source, str) else source
        )

        ts_parser = self._get_parser(language)
        tree: Tree = ts_parser.parse(source_bytes)
        has_error: bool = tree.root_node.has_error

        if has_error:
            logger.debug(
                "Syntax error(s) detected while parsing '%s' source (%d bytes); "
                "returning partial AST.",
                language,
                len(source_bytes),
            )

        return ParseResult(
            tree=tree,
            language=language,
            has_error=has_error,
            source=source_bytes,
        )

    def parse_file(self, file_path: str, *, language: str) -> ParseResult:
        """Parse the file at *file_path*.

        Convenience wrapper around :meth:`parse` that reads the file for you.

        Args:
            file_path: Absolute or relative path to the source file.
            language:  Canonical language name (e.g. ``"python"``).

        Returns:
            A :class:`ParseResult`.

        Raises:
            ParserError:  If *language* is not registered.
            OSError:      If *file_path* cannot be read.
        """
        with open(file_path, "rb") as fh:
            source_bytes = fh.read()
        return self.parse(source_bytes, language=language)

    def walk(
        self,
        result: ParseResult,
        *,
        named_only: bool = False,
    ) -> Iterator[NodeInfo]:
        """Depth-first walk of all nodes in *result*.

        Args:
            result:     A :class:`ParseResult` from :meth:`parse`.
            named_only: When ``True``, yield only named nodes (skipping
                        anonymous punctuation/keyword nodes).

        Yields:
            :class:`NodeInfo` for each visited node.
        """
        yield from self._walk_node(result.root_node, named_only=named_only)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser(self, language: str) -> Parser:
        """Return (or create and cache) a :class:`Parser` for *language*."""
        if language not in self._parsers:
            lang_obj = self._languages[language]
            self._parsers[language] = Parser(lang_obj)
        return self._parsers[language]

    def _walk_node(self, node: Node, *, named_only: bool) -> Iterator[NodeInfo]:
        """Recursive depth-first traversal starting from *node*."""
        if named_only and not node.is_named:
            return
        yield NodeInfo.from_node(node)
        for child in node.children:
            yield from self._walk_node(child, named_only=named_only)

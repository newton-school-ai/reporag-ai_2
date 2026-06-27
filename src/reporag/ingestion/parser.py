"""Tree-sitter AST parser for Python (and JavaScript) source files.

Parses source files into tree-sitter ASTs using a language-agnostic interface.
Handles parse errors gracefully by returning partial ASTs with a ``has_error``
flag rather than raising exceptions.  Supports Python and JavaScript out of the
box; additional grammars can be registered at runtime via
:meth:`ASTParser.register_language`.

Usage::

    from reporag.ingestion.parser import ASTParser, ParseResult

    parser = ASTParser()

    # Parse a Python snippet
    result = parser.parse('def hello():\\n    return 42\\n', language='python')
    print(result.root_node.type)    # 'module'
    print(result.has_error)         # False
    print(result.node_count)        # total nodes in the tree

    # Walk every node depth-first
    for node_info in parser.walk(result):
        print(node_info.node_type, node_info.start_line, node_info.text[:40])

    # Walk only named (non-anonymous) nodes
    for node_info in parser.walk(result, named_only=True):
        print(node_info)
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported language registry
# ---------------------------------------------------------------------------

#: Module-level registry: canonical language name -> tree_sitter Language.
#: Populated lazily on first ASTParser instantiation so import time is fast.
_LANGUAGE_REGISTRY: dict[str, Language] = {}


def _load_builtin_languages() -> None:
    """Populate _LANGUAGE_REGISTRY with grammars that ship with requirements.txt."""
    # Python (tree-sitter-python is a hard requirement)
    try:
        import tree_sitter_python as _tspy  # type: ignore[import]

        _LANGUAGE_REGISTRY["python"] = Language(_tspy.language())
    except ImportError:  # pragma: no cover
        logger.warning(
            "tree-sitter-python grammar not installed; Python parsing unavailable."
        )

    # JavaScript (tree-sitter-javascript is a hard requirement)
    try:
        import tree_sitter_javascript as _tsjs  # type: ignore[import]

        _LANGUAGE_REGISTRY["javascript"] = Language(_tsjs.language())
    except ImportError:  # pragma: no cover
        logger.warning(
            "tree-sitter-javascript grammar not installed; JavaScript parsing unavailable."
        )

    # TypeScript -- optional, not in default requirements.txt
    try:
        import tree_sitter_typescript as _tsts  # type: ignore[import]

        _LANGUAGE_REGISTRY["typescript"] = Language(_tsts.language_typescript())
        _LANGUAGE_REGISTRY["tsx"] = Language(_tsts.language_tsx())
    except ImportError:
        pass  # TypeScript is gracefully optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeInfo:
    """Structured, serialisable representation of a single tree-sitter node.

    All position fields are **0-indexed** and match tree-sitter conventions
    (``start_point`` / ``end_point`` tuples).

    Attributes:
        node_type:   Grammar node type string, e.g. ``"function_definition"``.
        text:        UTF-8 decoded source text spanned by this node.
        start_line:  0-indexed line where the node begins.
        end_line:    0-indexed line where the node ends (inclusive).
        start_col:   0-indexed column where the node begins on ``start_line``.
        end_col:     0-indexed column where the node ends on ``end_line``
                     (exclusive, consistent with Python slice notation).
        is_error:    ``True`` when this node is an ``ERROR`` or ``MISSING`` node
                     produced by tree-sitter's error-recovery algorithm.
        is_named:    ``True`` when the node type is a named grammar rule (as
                     opposed to an anonymous literal such as ``"def"`` or ``":"``).
        child_count: Number of direct children of this node in the AST.
    """

    node_type: str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_error: bool
    is_named: bool
    child_count: int

    @classmethod
    def from_node(cls, node: Node) -> NodeInfo:
        """Construct a :class:`NodeInfo` from a raw tree-sitter :class:`Node`.

        Args:
            node: A tree-sitter node obtained from a parsed tree.

        Returns:
            An immutable :class:`NodeInfo` populated from the node's fields.
        """
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
            child_count=node.child_count,
        )


@dataclass
class ParseResult:
    """The output of a single :meth:`ASTParser.parse` call.

    Attributes:
        tree:      Raw tree-sitter :class:`Tree`.  Access ``root_node`` for the
                   AST root, or pass the whole result to :meth:`ASTParser.walk`.
        language:  Canonical language name used for parsing (e.g. ``"python"``).
        has_error: ``True`` when the source contained syntax errors.  The
                   ``tree`` is still valid (partial AST produced by tree-sitter's
                   error-recovery); callers decide whether to proceed or discard.
        source:    Original source as ``bytes`` (UTF-8).  Stored so callers can
                   slice out node text by byte offset if needed.
    """

    tree: Tree
    language: str
    has_error: bool
    source: bytes

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def root_node(self) -> Node:
        """Shortcut for ``tree.root_node``."""
        return self.tree.root_node

    @property
    def node_count(self) -> int:
        """Total number of nodes in the AST (named + anonymous).

        Computed by a single BFS pass; cached implicitly via Python's property
        semantics on the frozen :class:`Tree` object.
        """
        count = 0
        queue: deque[Node] = deque([self.tree.root_node])
        while queue:
            node = queue.popleft()
            count += 1
            queue.extend(node.children)
        return count


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParserError(ValueError):
    """Raised for unrecoverable *configuration* errors (e.g. unknown language).

    This is intentionally NOT raised for source-level syntax errors -- those
    are captured in :attr:`ParseResult.has_error` and the partial AST is
    always returned so that downstream consumers can still extract whatever
    symbols are parseable.
    """


# ---------------------------------------------------------------------------
# ASTParser
# ---------------------------------------------------------------------------


class ASTParser:
    """Language-agnostic tree-sitter AST parser.

    One :class:`tree_sitter.Parser` instance is created per language and
    cached, so repeated parses in the same language are cheap.

    Args:
        languages: Optional mapping of ``language_name -> Language`` that
            *replaces* the built-in module-level registry for this instance.
            Pass an empty dict to start with no grammars (useful in tests that
            inject a specific grammar via :meth:`register_language`).

    Examples:
        >>> parser = ASTParser()
        >>> result = parser.parse('def hello():\\n    return 42\\n', language='python')
        >>> result.root_node.type
        'module'
        >>> result.has_error
        False
        >>> result.node_count > 0
        True
    """

    def __init__(self, languages: dict[str, Language] | None = None) -> None:
        # Ensure the module-level registry is populated the first time any
        # ASTParser is constructed (lazy so import of this module is fast).
        if languages is None and not _LANGUAGE_REGISTRY:
            _load_builtin_languages()

        self._languages: dict[str, Language] = (
            dict(languages) if languages is not None else _LANGUAGE_REGISTRY
        )
        # Cache of tree_sitter.Parser objects, keyed by language name.
        self._parsers: dict[str, Parser] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """Alphabetically sorted list of language names available on this instance."""
        return sorted(self._languages)

    def register_language(self, name: str, language: Language) -> None:
        """Register (or replace) a grammar on this instance.

        Args:
            name:     Canonical language name, e.g. ``"rust"``.
            language: An initialised :class:`tree_sitter.Language` object.

        Note:
            Registering a language that is already registered invalidates the
            cached :class:`Parser` so the new grammar takes effect immediately.
        """
        self._languages[name] = language
        self._parsers.pop(name, None)  # drop stale cached parser
        logger.debug("Registered language '%s'", name)

    def parse(self, source: str | bytes, *, language: str) -> ParseResult:
        """Parse *source* using the named *language* grammar.

        The method **never raises** for source-level syntax errors.  Instead it
        returns a :class:`ParseResult` with ``has_error=True`` and the partial
        AST produced by tree-sitter's built-in error-recovery algorithm.  This
        ensures that broken files are still partially indexed rather than
        silently dropped.

        Args:
            source:   Source code as a ``str`` (UTF-8 encoded internally before
                      parsing) or as raw ``bytes``.
            language: Canonical language name registered on this parser instance,
                      e.g. ``"python"`` or ``"javascript"``.

        Returns:
            A :class:`ParseResult` containing the tree-sitter ``Tree``, the
            ``language`` name, a ``has_error`` flag, and the original source
            ``bytes``.

        Raises:
            ParserError: If *language* is not registered on this instance.
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
        """Read *file_path* from disk and parse it.

        Convenience wrapper around :meth:`parse` for the common case where the
        caller has a path rather than an in-memory string.

        Args:
            file_path: Path to the source file (absolute or relative).
            language:  Canonical language name (e.g. ``"python"``).

        Returns:
            A :class:`ParseResult`.

        Raises:
            ParserError: If *language* is not registered.
            OSError:     If *file_path* cannot be opened or read.
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
        """Yield every node in the AST in depth-first (pre-order) order.

        Uses an explicit stack instead of recursion so that deeply nested
        files (e.g. generated code with thousands of levels) never hit
        Python's default recursion limit.

        Args:
            result:     A :class:`ParseResult` returned by :meth:`parse` or
                        :meth:`parse_file`.
            named_only: When ``True``, anonymous nodes (punctuation, keywords
                        stored as literals in the grammar) are skipped.  Only
                        named grammar rules are yielded.

        Yields:
            :class:`NodeInfo` for each visited node, in pre-order.
        """
        # Explicit stack-based DFS to avoid Python recursion limits on deep trees.
        stack: list[Node] = [result.root_node]
        while stack:
            node = stack.pop()
            if named_only and not node.is_named:
                # Still descend into children -- a named child can be under an
                # anonymous parent (e.g. a "block" inside a ":" token's sibling).
                stack.extend(reversed(node.children))
                continue
            yield NodeInfo.from_node(node)
            # Push children in reverse so leftmost child is processed first.
            stack.extend(reversed(node.children))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser(self, language: str) -> Parser:
        """Return a cached (or newly created) :class:`Parser` for *language*."""
        if language not in self._parsers:
            self._parsers[language] = Parser(self._languages[language])
        return self._parsers[language]

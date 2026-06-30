"""AST-aware semantic code chunking.

The chunker consumes the tree-sitter parser and Python symbol extractor from
the ingestion pipeline and emits chunks suitable for code embedding. It keeps
functions and classes intact whenever possible, and only splits oversized
symbols between complete body statements. Continuation chunks repeat the
smallest useful syntactic context -- the original signature -- without
duplicating the preceding body.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from tree_sitter import Node, Tree

from src.reporag.config import settings
from src.reporag.ingestion.parser import (
    ASTParser,
    ParseError,
    UnsupportedLanguageError,
)
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

_DEFINITION_NODE_TYPES = {"class_definition", "function_definition"}
_COMMENT_NODE_TYPE = "comment"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeChunk:
    """A syntactically coherent source chunk plus retrieval metadata.

    Attributes:
        text: Source text for the chunk. Oversized continuation chunks may be
            synthetic snippets made from a signature plus a later body segment.
        file_path: Repository-relative or absolute source path.
        start_line: 1-based first source line represented by the chunk text.
        end_line: 1-based final source line represented by the chunk body.
        parent_symbol: Qualified name of the lexical parent symbol, if known.
        language: Canonical language name used by the parser.
        token_count: Number of tokens in ``text`` according to tiktoken.
    """

    text: str
    file_path: str
    start_line: int
    end_line: int
    parent_symbol: str | None
    language: str
    token_count: int


@dataclass(frozen=True)
class _ChunkingState:
    """Immutable per-file state shared by helper methods."""

    source_bytes: bytes
    file_path: str
    language: str
    symbol_index: dict[tuple[int, int, str], Symbol]


@dataclass(frozen=True)
class _BodyUnit:
    """A complete body statement and any comments attached to it."""

    nodes: tuple[Node, ...]

    @property
    def first(self) -> Node:
        """Return the first AST node in the unit."""
        return self.nodes[0]

    @property
    def last(self) -> Node:
        """Return the final AST node in the unit."""
        return self.nodes[-1]


@dataclass(frozen=True)
class _DefinitionContext:
    """Prefix text needed to keep nested definition chunks syntactically valid."""

    first_prefix: str = ""
    continuation_prefix: str = ""
    first_start_line: int | None = None
    continuation_start_line: int | None = None


class SemanticChunker:
    """Chunk source files at AST and symbol boundaries.

    Args:
        max_tokens: Maximum target chunk size. Defaults to
            ``settings.chunk_max_tokens``.
        parser: Optional parser instance to reuse across files.
        symbol_extractor: Optional extractor instance to reuse across files.
        encoding_name: tiktoken encoding used for token counting.
        encoding: Optional preloaded tiktoken encoding. Primarily useful for
            tests or deployments that manage tokenizer assets explicitly.

    The chunker is intentionally stateless between calls. Parser and extractor
    instances are held only so their internal caches can be reused safely.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        parser: ASTParser | None = None,
        symbol_extractor: SymbolExtractor | None = None,
        encoding_name: str = "cl100k_base",
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        """Initialise a semantic chunker."""
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        self.parser = parser or ASTParser()
        self.symbol_extractor = symbol_extractor or SymbolExtractor(parser=self.parser)
        self._encoding = encoding or self._load_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the tiktoken token count for ``text``."""
        return len(self._encoding.encode(text))

    def chunk_source(
        self,
        source: str | bytes,
        language: str = "python",
        file_path: str = "<string>",
    ) -> list[CodeChunk]:
        """Parse and chunk an in-memory source string.

        Args:
            source: Source code as ``str`` or UTF-8 ``bytes``.
            language: Parser language name.
            file_path: Metadata path to attach to emitted chunks.

        Returns:
            A list of semantic chunks in source order.
        """
        lang = language.lower().strip()
        source_bytes = self._to_bytes(source)
        tree = self.parser.parse(source_bytes, language=lang)
        return self.chunk_tree(tree, file_path, source_bytes, language=lang)

    def chunk_tree(
        self,
        tree: Tree,
        file_path: str,
        source: str | bytes,
        language: str = "python",
    ) -> list[CodeChunk]:
        """Chunk an existing tree-sitter tree without reparsing.

        Args:
            tree: Tree returned by :class:`ASTParser`.
            file_path: Metadata path to attach to emitted chunks.
            source: Original source used to build ``tree``.
            language: Parser language name.

        Returns:
            A list of semantic chunks in source order.
        """
        lang = language.lower().strip()
        source_bytes = self._to_bytes(source)
        symbols = self.symbol_extractor.extract_from_tree(
            tree,
            file_path=file_path,
            source=source_bytes,
            language=lang,
        )
        state = _ChunkingState(
            source_bytes=source_bytes,
            file_path=file_path,
            language=lang,
            symbol_index=self._build_symbol_index(symbols),
        )
        return self._chunk_module(tree.root_node, state)

    def chunk_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> list[CodeChunk]:
        """Read, parse, and chunk a source file.

        The language is inferred from ``settings.extension_map`` when it is not
        supplied, matching the parser and symbol extractor APIs.

        Raises:
            ParseError: If the file cannot be read.
            UnsupportedLanguageError: If the language cannot be inferred or is
                not supported.
        """
        fpath = Path(file_path)
        lang = self._infer_language(fpath, language)
        try:
            source_bytes = fpath.read_bytes()
        except OSError as exc:
            raise ParseError(f"Cannot read file '{fpath}': {exc}") from exc

        tree = self.parser.parse(source_bytes, language=lang)
        return self.chunk_tree(tree, str(fpath), source_bytes, language=lang)

    # ------------------------------------------------------------------
    # Module-level chunking
    # ------------------------------------------------------------------

    def _chunk_module(self, root: Node, state: _ChunkingState) -> list[CodeChunk]:
        """Chunk the module root into definitions and top-level code groups."""
        chunks: list[CodeChunk] = []
        pending_top_level: list[Node] = []

        for child in root.named_children:
            definition = self._definition_nodes(child)
            if definition:
                chunks.extend(self._flush_top_level(pending_top_level, state))
                pending_top_level = []
                outer_node, def_node = definition
                chunks.extend(
                    self._chunk_definition(
                        outer_node,
                        def_node,
                        state,
                        context=_DefinitionContext(),
                    )
                )
            else:
                pending_top_level.append(child)

        chunks.extend(self._flush_top_level(pending_top_level, state))
        return chunks

    def _flush_top_level(
        self,
        nodes: list[Node],
        state: _ChunkingState,
    ) -> list[CodeChunk]:
        """Pack adjacent non-definition module nodes into token-sized chunks."""
        chunks: list[CodeChunk] = []
        current: list[Node] = []

        for node in nodes:
            candidate = [*current, node]
            if (
                current
                and self.count_tokens(self._nodes_text(candidate, state))
                > self.max_tokens
            ):
                chunks.append(self._make_top_level_chunk(current, state))
                current = [node]
            else:
                current = candidate

            if (
                len(current) == 1
                and self.count_tokens(self._nodes_text(current, state))
                > self.max_tokens
            ):
                chunks.append(self._make_top_level_chunk(current, state))
                current = []

        if current:
            chunks.append(self._make_top_level_chunk(current, state))
        return chunks

    def _make_top_level_chunk(
        self,
        nodes: list[Node],
        state: _ChunkingState,
    ) -> CodeChunk:
        """Create a chunk from contiguous top-level nodes."""
        text = self._nodes_text(nodes, state)
        return self._make_chunk(
            text=text,
            state=state,
            start_line=self._line(nodes[0]),
            end_line=self._line(nodes[-1], end=True),
            parent_symbol=None,
        )

    # ------------------------------------------------------------------
    # Definition chunking
    # ------------------------------------------------------------------

    def _chunk_definition(
        self,
        outer_node: Node,
        def_node: Node,
        state: _ChunkingState,
        context: _DefinitionContext,
    ) -> list[CodeChunk]:
        """Chunk one function or class definition.

        Definitions are emitted whole when they fit. Oversized definitions are
        split only across direct body units, where each unit contains one
        complete statement plus nearby comments.
        """
        symbol = self._symbol_for_definition(def_node, state)
        whole_text = context.first_prefix + self._slice(
            state,
            outer_node.start_byte,
            outer_node.end_byte,
        )
        whole_start_line = context.first_start_line or self._line(outer_node)

        if self.count_tokens(whole_text) <= self.max_tokens:
            return [
                self._make_chunk(
                    text=whole_text,
                    state=state,
                    start_line=whole_start_line,
                    end_line=self._line(outer_node, end=True),
                    parent_symbol=self._parent_symbol(symbol),
                )
            ]

        body_node = def_node.child_by_field_name("body")
        if body_node is None or not body_node.named_children:
            return [
                self._make_chunk(
                    text=whole_text,
                    state=state,
                    start_line=whole_start_line,
                    end_line=self._line(outer_node, end=True),
                    parent_symbol=self._parent_symbol(symbol),
                )
            ]

        units = self._body_units(body_node)
        if not units:
            return [
                self._make_chunk(
                    text=whole_text,
                    state=state,
                    start_line=whole_start_line,
                    end_line=self._line(outer_node, end=True),
                    parent_symbol=self._parent_symbol(symbol),
                )
            ]

        first_prefix = context.first_prefix + self._slice(
            state,
            outer_node.start_byte,
            body_node.start_byte,
        )
        continuation_prefix = context.continuation_prefix + self._signature_prefix(
            def_node,
            body_node,
            state,
        )
        first_start_line = context.first_start_line or self._line(outer_node)
        continuation_start_line = context.continuation_start_line or self._line(
            def_node
        )

        return self._pack_body_units(
            units=units,
            state=state,
            symbol=symbol,
            first_prefix=first_prefix,
            continuation_prefix=continuation_prefix,
            first_start_line=first_start_line,
            continuation_start_line=continuation_start_line,
        )

    def _pack_body_units(
        self,
        units: list[_BodyUnit],
        state: _ChunkingState,
        symbol: Symbol | None,
        first_prefix: str,
        continuation_prefix: str,
        first_start_line: int,
        continuation_start_line: int,
    ) -> list[CodeChunk]:
        """Pack body units into chunks while preserving statement boundaries."""
        chunks: list[CodeChunk] = []
        current: list[_BodyUnit] = []
        current_prefix = first_prefix
        current_start_line = first_start_line

        for unit in units:
            candidate = [*current, unit]
            candidate_text = current_prefix + self._units_text(candidate, state)
            if current and self.count_tokens(candidate_text) > self.max_tokens:
                chunks.append(
                    self._make_body_chunk(
                        current,
                        current_prefix,
                        current_start_line,
                        state,
                        symbol,
                    )
                )
                current = []
                current_prefix = continuation_prefix
                current_start_line = continuation_start_line

            if not current and self._unit_exceeds_limit(unit, current_prefix, state):
                nested = self._chunk_nested_definition(
                    unit,
                    state,
                    first_prefix=current_prefix,
                    continuation_prefix=continuation_prefix,
                    first_start_line=current_start_line,
                    continuation_start_line=continuation_start_line,
                )
                if nested:
                    chunks.extend(nested)
                else:
                    chunks.append(
                        self._make_body_chunk(
                            [unit],
                            current_prefix,
                            current_start_line,
                            state,
                            symbol,
                        )
                    )
                current_prefix = continuation_prefix
                current_start_line = continuation_start_line
                continue

            current.append(unit)

        if current:
            chunks.append(
                self._make_body_chunk(
                    current,
                    current_prefix,
                    current_start_line,
                    state,
                    symbol,
                )
            )
        return chunks

    def _make_body_chunk(
        self,
        units: list[_BodyUnit],
        prefix: str,
        start_line: int,
        state: _ChunkingState,
        symbol: Symbol | None,
    ) -> CodeChunk:
        """Create a chunk from body units and a definition signature prefix."""
        text = prefix + self._units_text(units, state)
        return self._make_chunk(
            text=text,
            state=state,
            start_line=start_line,
            end_line=self._line(units[-1].last, end=True),
            parent_symbol=self._parent_symbol(symbol),
        )

    def _chunk_nested_definition(
        self,
        unit: _BodyUnit,
        state: _ChunkingState,
        first_prefix: str,
        continuation_prefix: str,
        first_start_line: int,
        continuation_start_line: int,
    ) -> list[CodeChunk]:
        """Recursively split an oversized nested function or class unit."""
        definition = self._definition_nodes(self._first_non_comment(unit))
        if definition is None:
            return []

        outer_node, def_node = definition
        context = _DefinitionContext(
            first_prefix=first_prefix
            + self._leading_comment_text(unit, outer_node, state),
            continuation_prefix=continuation_prefix,
            first_start_line=first_start_line,
            continuation_start_line=continuation_start_line,
        )
        return self._chunk_definition(outer_node, def_node, state, context=context)

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    def _definition_nodes(self, node: Node | None) -> tuple[Node, Node] | None:
        """Return ``(outer_node, definition_node)`` for supported definitions."""
        if node is None:
            return None
        if node.type in _DEFINITION_NODE_TYPES:
            return node, node
        if node.type != "decorated_definition":
            return None

        for child in node.named_children:
            if child.type in _DEFINITION_NODE_TYPES:
                return node, child
        return None

    def _body_units(self, body_node: Node) -> list[_BodyUnit]:
        """Group body statements with adjacent comments.

        Comments do not make a valid Python suite by themselves, so leading
        comments are attached to the following statement and trailing comments
        are attached to the previous unit.
        """
        units: list[_BodyUnit] = []
        pending_comments: list[Node] = []

        for child in body_node.named_children:
            if child.type == _COMMENT_NODE_TYPE:
                pending_comments.append(child)
                continue

            units.append(_BodyUnit(tuple([*pending_comments, child])))
            pending_comments = []

        if pending_comments:
            if units:
                last_unit = units[-1]
                units[-1] = _BodyUnit(tuple([*last_unit.nodes, *pending_comments]))
            else:
                units.append(_BodyUnit(tuple(pending_comments)))

        return units

    def _unit_exceeds_limit(
        self,
        unit: _BodyUnit,
        prefix: str,
        state: _ChunkingState,
    ) -> bool:
        """Return True when a single unit cannot fit with the given prefix."""
        return (
            self.count_tokens(prefix + self._units_text([unit], state))
            > self.max_tokens
        )

    def _first_non_comment(self, unit: _BodyUnit) -> Node | None:
        """Return the first non-comment node in a body unit."""
        for node in unit.nodes:
            if node.type != _COMMENT_NODE_TYPE:
                return node
        return None

    def _leading_comment_text(
        self,
        unit: _BodyUnit,
        outer_node: Node,
        state: _ChunkingState,
    ) -> str:
        """Return comments that were grouped immediately before a definition."""
        if unit.first == outer_node:
            return ""
        return self._slice(state, unit.first.start_byte, outer_node.start_byte)

    def _signature_prefix(
        self,
        def_node: Node,
        body_node: Node,
        state: _ChunkingState,
    ) -> str:
        """Build ``signature + newline + body indentation`` for continuations."""
        header_end = self._header_end_byte(def_node, body_node)
        header = self._slice(state, def_node.start_byte, header_end).rstrip()
        return f"{header}\n{self._indent_before(body_node.start_byte, state)}"

    def _header_end_byte(self, def_node: Node, body_node: Node) -> int:
        """Return the byte offset immediately after a definition header colon."""
        for child in def_node.children:
            if child.type == ":":
                return child.end_byte
        return body_node.start_byte

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _build_symbol_index(
        self,
        symbols: list[Symbol],
    ) -> dict[tuple[int, int, str], Symbol]:
        """Index symbols by range and name for AST node lookup."""
        return {
            (symbol.start_line, symbol.end_line, symbol.name): symbol
            for symbol in symbols
            if symbol.type in {"class", "function", "method"}
        }

    def _symbol_for_definition(
        self,
        def_node: Node,
        state: _ChunkingState,
    ) -> Symbol | None:
        """Return the extracted symbol that corresponds to ``def_node``."""
        name_node = def_node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self._slice(state, name_node.start_byte, name_node.end_byte)
        return state.symbol_index.get(
            (self._line(def_node), self._line(def_node, end=True), name)
        )

    @staticmethod
    def _parent_symbol(symbol: Symbol | None) -> str | None:
        """Extract parent metadata from a symbol."""
        return symbol.parent_symbol if symbol else None

    # ------------------------------------------------------------------
    # Source slicing and chunk construction
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        text: str,
        state: _ChunkingState,
        start_line: int,
        end_line: int,
        parent_symbol: str | None,
    ) -> CodeChunk:
        """Create a :class:`CodeChunk` and compute its token count."""
        return CodeChunk(
            text=text,
            file_path=state.file_path,
            start_line=start_line,
            end_line=end_line,
            parent_symbol=parent_symbol,
            language=state.language,
            token_count=self.count_tokens(text),
        )

    def _nodes_text(self, nodes: list[Node], state: _ChunkingState) -> str:
        """Return source text spanning a list of contiguous nodes."""
        return self._slice(state, nodes[0].start_byte, nodes[-1].end_byte)

    def _units_text(self, units: list[_BodyUnit], state: _ChunkingState) -> str:
        """Return source text spanning a list of contiguous body units."""
        return self._slice(state, units[0].first.start_byte, units[-1].last.end_byte)

    @staticmethod
    def _slice(state: _ChunkingState, start_byte: int, end_byte: int) -> str:
        """Decode a source byte range using replacement for malformed bytes."""
        return state.source_bytes[start_byte:end_byte].decode(
            "utf-8",
            errors="replace",
        )

    @staticmethod
    def _indent_before(byte_offset: int, state: _ChunkingState) -> str:
        """Return the whitespace between a line start and ``byte_offset``."""
        line_start = state.source_bytes.rfind(b"\n", 0, byte_offset) + 1
        indent = state.source_bytes[line_start:byte_offset]
        return indent.decode("utf-8", errors="replace")

    @staticmethod
    def _line(node: Node, end: bool = False) -> int:
        """Return a 1-based start or end line for a tree-sitter node."""
        point = node.end_point if end else node.start_point
        return point[0] + 1

    @staticmethod
    def _to_bytes(source: str | bytes) -> bytes:
        """Encode string source as UTF-8 bytes for tree-sitter offsets."""
        return source.encode("utf-8") if isinstance(source, str) else source

    @staticmethod
    def _infer_language(path: Path, language: str | None) -> str:
        """Infer a canonical language name from a path or explicit override."""
        if language is not None:
            return language.lower().strip()

        ext = path.suffix.lower()
        inferred = settings.extension_map.get(ext)
        if inferred is None:
            raise UnsupportedLanguageError(
                f"Cannot infer language for extension '{ext}'. "
                "Pass language= explicitly or add it to settings.extension_map."
            )
        return inferred

    @staticmethod
    def _load_encoding(encoding_name: str) -> tiktoken.Encoding:
        """Load a tiktoken encoding, falling back to an offline byte encoding.

        Some tiktoken wheels lazily fetch OpenAI BPE files on first use. The
        fallback keeps ingestion deterministic in restricted deployments while
        still using tiktoken's tokenizer implementation rather than ad-hoc word
        or character counts.
        """
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # pragma: no cover - depends on host cache/network
            logger.warning(
                "Falling back to byte-level tiktoken encoding because '%s' "
                "could not be loaded: %s",
                encoding_name,
                exc,
            )
            return tiktoken.Encoding(
                name="reporag_byte_fallback",
                pat_str=r"(?s).",
                mergeable_ranks={bytes([idx]): idx for idx in range(256)},
                special_tokens={},
            )

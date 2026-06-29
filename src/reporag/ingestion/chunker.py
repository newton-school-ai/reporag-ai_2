"""Semantic code chunker.

AST-aware chunking that respects function/class boundaries. Never splits
a function mid-body. Large functions are split at logical points with
function signature overlap.

Each chunk carries rich metadata: file_path, start_line, end_line,
parent_symbol, language, token_count, is_continuation, overlap_header.

Usage::

    from src.reporag.ingestion.chunker import SemanticChunker

    chunker = SemanticChunker(max_tokens=512)
    chunks = chunker.chunk_file('src/reporag/ingestion/cloner.py')
    for c in chunks:
        print(f'[{c.start_line}-{c.end_line}] {c.parent_symbol} ({c.token_count} tokens)')

    # Or chunk in-memory source:
    chunks = chunker.chunk_source(source_code, language='python', file_path='<string>')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser, UnsupportedLanguageError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# cl100k_base is the encoding used by GPT-3.5-turbo / GPT-4; it is the
# de-facto standard for RAG token budgets.
_ENCODING_NAME = "cl100k_base"
_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return the shared tiktoken encoder, loading it lazily once."""
    global _ENCODER  # noqa: PLW0603
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(_ENCODING_NAME)
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the shared encoder.

    - **Why it exists**: Centralises token counting so the encoder is loaded
      exactly once and all callers use the same budget unit.
    - **Algorithm**: Delegates to tiktoken's BPE encoder. The result is the
      number of BPE tokens, which closely matches OpenAI's embedding models.
    - **Edge cases**: Empty strings return 0 without invoking the encoder.
    - **Correctness choice**: Uses ``cl100k_base`` (GPT-4 encoding) so token
      counts match what the downstream embedding step will consume.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A contiguous slice of source code with full metadata.

    Attributes:
        content:        The raw source text of this chunk (UTF-8 string).
        file_path:      Path to the originating source file, or ``"<string>"``
                        for in-memory sources.
        language:       Language name (e.g. ``"python"``).
        start_line:     1-based line number where this chunk begins.
        end_line:       1-based line number where this chunk ends (inclusive).
        parent_symbol:  Qualified name of the enclosing function or class
                        (e.g. ``"MyClass.my_method"``), or ``None`` for
                        module-level code.
        token_count:    Number of BPE tokens in *content* (pre-computed).
        chunk_index:    0-based position of this chunk within its parent
                        symbol.  Always 0 for single-chunk symbols.
        is_continuation: ``True`` when this is not the first chunk of a
                         symbol that was split across multiple chunks.
        overlap_header: The function/class signature line repeated at the
                        start of continuation chunks for context.  ``None``
                        for first chunks and module-level chunks.
        has_parse_error: ``True`` when the source contained a syntax error
                         that caused partial AST extraction.
    """

    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    parent_symbol: str | None = None
    token_count: int = 0
    chunk_index: int = 0
    is_continuation: bool = False
    overlap_header: str | None = None
    has_parse_error: bool = False

    def __post_init__(self) -> None:
        """Pre-compute token_count if caller left it at the default 0."""
        if self.token_count == 0 and self.content:
            self.token_count = count_tokens(self.content)

    def __repr__(self) -> str:
        sym = self.parent_symbol or "<module>"
        cont = " [cont]" if self.is_continuation else ""
        return (
            f"Chunk({sym}{cont} [{self.start_line}-{self.end_line}]"
            f" {self.token_count} tokens)"
        )


# ---------------------------------------------------------------------------
# Python chunker implementation
# ---------------------------------------------------------------------------


class PythonChunker:
    """Produces :class:`Chunk` objects from a Python tree-sitter AST.

    Strategy
    --------
    1. Walk the module's top-level children.
    2. Each function/class definition becomes one or more chunks.
    3. A definition whose token count exceeds *max_tokens* is split at
       statement boundaries inside its body with signature overlap.
    4. Contiguous module-level statements (imports, assignments, etc.) that
       fall between definitions are grouped into a single module-level chunk.
    """

    def __init__(
        self,
        source_bytes: bytes,
        file_path: str,
        language: str,
        max_tokens: int,
    ) -> None:
        """Initialise the Python chunker.

        Args:
            source_bytes: Raw source bytes (UTF-8) of the file being chunked.
            file_path:    Path label stored in each :class:`Chunk`.
            language:     Language name stored in each :class:`Chunk`.
            max_tokens:   Maximum token budget per chunk.
        """
        self.source_bytes = source_bytes
        self.file_path = file_path
        self.language = language
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chunk(self, tree: Tree) -> list[Chunk]:
        """Walk *tree* and produce the ordered list of :class:`Chunk` objects.

        - **Why it exists**: Drives the top-level chunking pass over the
          module root, dispatching to definition or module-level handlers.
        - **Algorithm**: Iterates over top-level named children in source
          order. Definitions go to :meth:`_chunk_definition`; all other nodes
          accumulate into a pending module-level buffer that is flushed when a
          definition boundary is encountered.
        - **Edge cases**: Trailing module-level statements after the last
          definition are flushed after the loop.
        - **Correctness choice**: Uses ``named_children`` to skip anonymous
          punctuation nodes that tree-sitter inserts between statements.
        """
        chunks: list[Chunk] = []
        pending_module_nodes: list[Node] = []

        for node in tree.root_node.named_children:
            if node.type in ("function_definition", "class_definition"):
                # Flush any accumulated module-level code before this definition
                if pending_module_nodes:
                    chunks.extend(self._flush_module_level(pending_module_nodes))
                    pending_module_nodes = []
                chunks.extend(self._chunk_definition(node, parent_symbol=None))

            elif node.type == "decorated_definition":
                if pending_module_nodes:
                    chunks.extend(self._flush_module_level(pending_module_nodes))
                    pending_module_nodes = []
                inner = self._inner_definition(node)
                if inner is not None:
                    chunks.extend(
                        self._chunk_definition(
                            inner,
                            parent_symbol=None,
                            decorator_node=node,
                        )
                    )
                else:
                    pending_module_nodes.append(node)

            else:
                pending_module_nodes.append(node)

        # Flush any remaining module-level nodes
        if pending_module_nodes:
            chunks.extend(self._flush_module_level(pending_module_nodes))

        return chunks

    # ------------------------------------------------------------------
    # Definition chunking
    # ------------------------------------------------------------------

    def _chunk_definition(
        self,
        node: Node,
        *,
        parent_symbol: str | None,
        decorator_node: Node | None = None,
    ) -> list[Chunk]:
        """Produce chunks for a single ``function_definition`` or ``class_definition``.

        - **Why it exists**: Handles both small definitions (one chunk) and
          large ones that exceed the token budget (split into continuation
          chunks).
        - **Algorithm**: First attempts to emit the whole definition as one
          chunk. If it exceeds *max_tokens*, it extracts the signature header
          and splits the body at statement boundaries. Each split produces a
          new chunk prefixed by the overlap header.
        - **Edge cases**: If even the signature alone exceeds the budget the
          chunk is still emitted (hard lower bound: one chunk per definition,
          never zero). If a single statement inside the body exceeds the budget
          it is also emitted as-is.
        - **Correctness choice**: Token budget is enforced on the final text
          (with overlap header prepended) so continuation chunks are never
          silently oversized.
        """
        qualified_name = self._qualified_name(node, parent_symbol)
        has_error = node.has_error or node.is_missing

        # Use the decorated definition span if present (so decorators are included)
        span_node = decorator_node if decorator_node is not None else node
        full_text = self._node_text(span_node)
        full_tokens = count_tokens(full_text)

        if full_tokens <= self.max_tokens:
            # Happy path: entire definition fits in one chunk
            chunk = Chunk(
                content=full_text,
                file_path=self.file_path,
                language=self.language,
                start_line=span_node.start_point[0] + 1,
                end_line=span_node.end_point[0] + 1,
                parent_symbol=parent_symbol,
                token_count=full_tokens,
                chunk_index=0,
                is_continuation=False,
                has_parse_error=has_error,
            )
            # For classes: recurse into methods so they appear as their own chunks
            # (in addition to the full class chunk for context)
            if node.type == "class_definition":
                return [chunk] + self._chunk_class_methods(
                    node, parent_symbol=qualified_name
                )
            return [chunk]

        # Oversized definition: split at statement boundaries
        return self._split_definition(
            node,
            span_node=span_node,
            qualified_name=qualified_name,
            parent_symbol=parent_symbol,
            has_error=has_error,
        )

    def _split_definition(
        self,
        node: Node,
        *,
        span_node: Node,
        qualified_name: str,
        parent_symbol: str | None,
        has_error: bool,
    ) -> list[Chunk]:
        """Split an oversized definition at statement boundaries.

        - **Why it exists**: Prevents a very large function from being emitted
          as a single chunk that exceeds the token budget.
        - **Algorithm**: Extracts the signature header (everything up to the
          first statement in the body). Then iterates over body statements,
          accumulating them into a window. When adding the next statement would
          exceed the budget, the window is flushed as a chunk and a new window
          starts with the overlap header prepended.
        - **Edge cases**: If the body node is missing or empty, the full text
          is emitted as a single (possibly oversized) chunk. If a single
          statement exceeds the budget, it is still emitted alone.
        - **Correctness choice**: Overlap header is prepended to continuation
          chunks *before* token counting, so the budget includes the header
          cost.
        """
        body_node = node.child_by_field_name("body")
        if body_node is None or not body_node.named_children:
            # Cannot split without a body: emit as-is
            full_text = self._node_text(span_node)
            return [
                Chunk(
                    content=full_text,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=span_node.start_point[0] + 1,
                    end_line=span_node.end_point[0] + 1,
                    parent_symbol=parent_symbol,
                    token_count=count_tokens(full_text),
                    chunk_index=0,
                    is_continuation=False,
                    has_parse_error=has_error,
                )
            ]

        # Build the overlap header: signature line(s) up to the body block
        overlap_header = self._extract_header(node, span_node)

        chunks: list[Chunk] = []
        chunk_index = 0
        window_lines: list[str] = []
        window_start: int | None = None
        window_end: int | None = None
        is_first = True

        def _flush(end_line: int) -> None:
            nonlocal chunk_index, is_first, window_start, window_lines, window_end
            if not window_lines:
                return
            body_text = "\n".join(window_lines)
            if is_first:
                content = f"{overlap_header}\n{body_text}"
                is_cont = False
            else:
                content = f"{overlap_header}  # ... continued\n{body_text}"
                is_cont = True

            chunks.append(
                Chunk(
                    content=content,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=window_start or (span_node.start_point[0] + 1),
                    end_line=end_line,
                    parent_symbol=parent_symbol,
                    token_count=count_tokens(content),
                    chunk_index=chunk_index,
                    is_continuation=is_cont,
                    overlap_header=overlap_header,
                    has_parse_error=has_error,
                )
            )
            chunk_index += 1
            is_first = False
            window_lines = []
            window_start = None
            window_end = None

        header_tokens = count_tokens(overlap_header)
        budget = self.max_tokens - header_tokens  # budget for body lines

        for stmt in body_node.named_children:
            stmt_text = self._node_text(stmt)
            stmt_tokens = count_tokens(stmt_text)
            stmt_start = stmt.start_point[0] + 1
            stmt_end = stmt.end_point[0] + 1

            current_tokens = (
                count_tokens("\n".join(window_lines)) if window_lines else 0
            )

            if window_lines and current_tokens + stmt_tokens > budget:
                _flush(window_end or stmt_end)

            if window_start is None:
                window_start = stmt_start
            window_lines.append(stmt_text)
            window_end = stmt_end

        _flush(window_end or (span_node.end_point[0] + 1))
        return chunks

    def _chunk_class_methods(
        self, class_node: Node, *, parent_symbol: str
    ) -> list[Chunk]:
        """Recursively chunk methods inside a class body.

        - **Why it exists**: Classes emit a full-class chunk (for context) and
          then individual method chunks so each method gets its own embedding.
        - **Algorithm**: Iterates over the class body's direct children,
          forwarding each function or decorated definition to
          :meth:`_chunk_definition`.
        - **Edge cases**: Nested classes recurse through :meth:`_chunk_definition`
          which calls this method again, achieving arbitrary nesting.
        - **Correctness choice**: Methods are emitted *after* the class chunk,
          not instead of it, so the class docstring and inheritance are always
          available for retrieval.
        """
        chunks: list[Chunk] = []
        body = class_node.child_by_field_name("body")
        if body is None:
            return chunks

        for child in body.named_children:
            if child.type == "function_definition":
                chunks.extend(
                    self._chunk_definition(child, parent_symbol=parent_symbol)
                )
            elif child.type == "decorated_definition":
                inner = self._inner_definition(child)
                if inner is not None:
                    chunks.extend(
                        self._chunk_definition(
                            inner,
                            parent_symbol=parent_symbol,
                            decorator_node=child,
                        )
                    )
            elif child.type == "class_definition":
                nested_qualified = self._qualified_name(child, parent_symbol)
                chunks.extend(
                    self._chunk_definition(child, parent_symbol=parent_symbol)
                )
                _ = nested_qualified  # already handled inside _chunk_definition

        return chunks

    # ------------------------------------------------------------------
    # Module-level flushing
    # ------------------------------------------------------------------

    def _flush_module_level(self, nodes: list[Node]) -> list[Chunk]:
        """Group a run of module-level nodes into one or more chunks.

        - **Why it exists**: Preserves module-level code (imports, assignments,
          ``__all__``, top-level constants) as retrievable chunks.
        - **Algorithm**: Concatenates node texts with newlines. If the total
          exceeds *max_tokens* the group is split at node boundaries using the
          same sliding-window approach as :meth:`_split_definition`.
        - **Edge cases**: Empty node list returns an empty list immediately.
        - **Correctness choice**: Module-level chunks have ``parent_symbol=None``
          to distinguish them from definition chunks in downstream filtering.
        """
        if not nodes:
            return []

        lines: list[str] = [self._node_text(n) for n in nodes]
        full_text = "\n".join(lines)
        full_tokens = count_tokens(full_text)

        if full_tokens <= self.max_tokens:
            return [
                Chunk(
                    content=full_text,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=nodes[0].start_point[0] + 1,
                    end_line=nodes[-1].end_point[0] + 1,
                    parent_symbol=None,
                    token_count=full_tokens,
                    chunk_index=0,
                    is_continuation=False,
                )
            ]

        # Split at node boundaries
        chunks: list[Chunk] = []
        chunk_index = 0
        window_texts: list[str] = []
        window_start: int | None = None
        window_end: int | None = None

        def _flush(end_line: int) -> None:
            nonlocal chunk_index, window_texts, window_start, window_end
            if not window_texts:
                return
            content = "\n".join(window_texts)
            chunks.append(
                Chunk(
                    content=content,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=window_start or (nodes[0].start_point[0] + 1),
                    end_line=end_line,
                    parent_symbol=None,
                    token_count=count_tokens(content),
                    chunk_index=chunk_index,
                    is_continuation=chunk_index > 0,
                )
            )
            chunk_index += 1
            window_texts = []
            window_start = None
            window_end = None

        for node, text in zip(nodes, lines, strict=False):
            node_tokens = count_tokens(text)
            current = count_tokens("\n".join(window_texts)) if window_texts else 0

            if window_texts and current + node_tokens > self.max_tokens:
                _flush(window_end or node.end_point[0] + 1)

            if window_start is None:
                window_start = node.start_point[0] + 1
            window_texts.append(text)
            window_end = node.end_point[0] + 1

        _flush(window_end or (nodes[-1].end_point[0] + 1))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node_text(self, node: Node) -> str:
        """Decode a node's exact source bytes to a UTF-8 string.

        - **Why it exists**: Slices the original source buffer by byte offsets
          so the returned text is exactly what the author wrote, including
          comments and spacing.
        - **Correctness choice**: Uses ``start_byte`` / ``end_byte`` instead of
          ``node.text`` so multi-line nodes with embedded newlines are handled
          faithfully on all platforms.
        """
        raw = self.source_bytes[node.start_byte : node.end_byte]
        return raw.decode("utf-8", errors="replace")

    def _extract_header(self, node: Node, span_node: Node) -> str:
        """Extract the signature header for use as an overlap prefix.

        - **Why it exists**: Continuation chunks need context to be useful
          standalone. Repeating the signature (and decorator, if present) gives
          the embedding model enough context to understand the surrounding scope.
        - **Algorithm**: Slices source bytes from the span_node start up to
          (but not including) the body block start byte.
        - **Edge cases**: If the body node is missing, returns the entire
          first line of the definition.
        - **Correctness choice**: Byte-level slicing preserves the exact
          formatting of the original source.
        """
        body = node.child_by_field_name("body")
        if body is None:
            # Fallback: use first line only
            first_line = self._node_text(span_node).splitlines()[0]
            return first_line

        start_byte = span_node.start_byte
        end_byte = body.start_byte
        header_bytes = self.source_bytes[start_byte:end_byte]
        return header_bytes.decode("utf-8", errors="replace").rstrip()

    @staticmethod
    def _qualified_name(node: Node, parent_symbol: str | None) -> str:
        """Build a dot-qualified name for a definition node.

        - **Why it exists**: Downstream consumers (knowledge graph, embedding
          index) need globally unique symbol names to avoid collisions between
          ``MyClass.run`` and ``OtherClass.run``.
        - **Algorithm**: Reads the ``name`` field child from the node using
          tree-sitter's field API, then prepends the parent prefix.
        - **Edge cases**: Unnamed nodes (malformed ASTs) fall back to
          ``<unknown>``.
        - **Correctness choice**: Uses ``child_by_field_name("name")`` rather
          than positional children so the result is grammar-stable.
        """
        name_node = node.child_by_field_name("name")
        name = (
            name_node.text.decode("utf-8", errors="replace")
            if name_node and name_node.text
            else "<unknown>"
        )
        return f"{parent_symbol}.{name}" if parent_symbol else name

    @staticmethod
    def _inner_definition(decorated_node: Node) -> Node | None:
        """Return the function or class definition inside a decorated_definition.

        - **Why it exists**: Decorated definitions wrap the real definition in
          an outer node. We need the inner ``function_definition`` or
          ``class_definition`` to extract name and body.
        - **Algorithm**: Scans named children for the target types.
        - **Edge cases**: Returns ``None`` if none found (malformed AST).
        """
        for child in decorated_node.named_children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None


# ---------------------------------------------------------------------------
# Language Registry
# ---------------------------------------------------------------------------

_CHUNKER_REGISTRY: dict[str, type[PythonChunker]] = {
    "python": PythonChunker,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SemanticChunker:
    """Language-agnostic coordinator for AST-aware code chunking.

    A single instance can be reused across many files; it caches one
    :class:`~src.reporag.ingestion.parser.ASTParser` so grammars are loaded
    at most once per language.

    Args:
        max_tokens: Maximum token budget per chunk (default 512).  Chunks
                    may slightly exceed this only when a single statement is
                    larger than the budget.
        parser:     Optional pre-built :class:`ASTParser`.  Inject in tests
                    to avoid disk I/O.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        parser: ASTParser | None = None,
    ) -> None:
        """Initialise the chunker."""
        self.max_tokens = max_tokens
        self._parser: ASTParser = parser if parser is not None else ASTParser()

    def chunk_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> list[Chunk]:
        """Parse a source file from disk and produce semantic chunks.

        Args:
            file_path: Path to the source file.
            language:  Language override.  When ``None``, inferred from the
                       file extension via ``settings.extension_map``.

        Returns:
            Ordered list of :class:`Chunk` objects covering the entire file.

        Raises:
            UnsupportedLanguageError: If the language cannot be determined or
                has no registered chunker.
            ParseError: If the file cannot be read.
        """
        from src.reporag.ingestion.parser import ParseError

        fpath = Path(file_path)

        if language is None:
            from src.reporag.config import settings

            ext = fpath.suffix.lower()
            language = settings.extension_map.get(ext)
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for extension '{ext}'. "
                    "Pass language= explicitly or add it to settings.extension_map."
                )

        try:
            source_bytes = fpath.read_bytes()
        except OSError as exc:
            raise ParseError(f"Cannot read file '{fpath}': {exc}") from exc

        tree: Tree = self._parser.parse(source_bytes, language=language)
        return self._chunk(tree, source_bytes, file_path=str(fpath), language=language)

    def chunk_source(
        self,
        source: str | bytes,
        *,
        language: str = "python",
        file_path: str = "<string>",
    ) -> list[Chunk]:
        """Parse in-memory *source* and produce semantic chunks.

        Args:
            source:    Source code as ``str`` or ``bytes``.
            language:  Language name (default ``"python"``).
            file_path: Optional label stored in each :class:`Chunk`.

        Returns:
            Ordered list of :class:`Chunk` objects.

        Raises:
            UnsupportedLanguageError: If no chunker is registered for
                *language*.
        """
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        tree: Tree = self._parser.parse(source_bytes, language=language)
        return self._chunk(tree, source_bytes, file_path=file_path, language=language)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _chunk(
        self,
        tree: Tree,
        source_bytes: bytes,
        *,
        file_path: str,
        language: str,
    ) -> list[Chunk]:
        """Dispatch to the language-specific chunker implementation.

        - **Why it exists**: Decouples the public API from the implementation
          so new languages can be added to the registry without changing the
          coordinator.
        - **Algorithm**: Looks up the registered chunker class by language name,
          instantiates it with the source bytes, and calls ``chunk(tree)``.
        - **Edge cases**: Unsupported languages raise
          :exc:`~src.reporag.ingestion.parser.UnsupportedLanguageError`
          rather than silently returning an empty list, because a missing chunker
          is a configuration error, not expected input.
        - **Correctness choice**: Language name is lower-cased before registry
          lookup to match the parser's normalisation.
        """
        lang = language.lower().strip()
        chunker_cls = _CHUNKER_REGISTRY.get(lang)
        if chunker_cls is None:
            raise UnsupportedLanguageError(
                f"No SemanticChunker registered for language '{language}'. "
                f"Supported: {sorted(_CHUNKER_REGISTRY)}"
            )
        impl = chunker_cls(source_bytes, file_path, language, self.max_tokens)
        return impl.chunk(tree)

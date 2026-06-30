"""Semantic code chunker.

AST-aware chunking that respects function/class boundaries. Never splits
a function mid-body. Large functions are split at logical points with
function signature overlap.

Each chunk carries rich metadata: file_path, start_line, end_line,
parent_symbol, qualified_name, chunk_kind, language, token_count,
is_continuation, overlap_header.

Usage::

    from src.reporag.ingestion.chunker import SemanticChunker

    chunker = SemanticChunker(max_tokens=512)
    chunks = chunker.chunk_file('examples/sample_repo/app.py')
    for c in chunks:
        print(f'[{c.start_line}-{c.end_line}] {c.qualified_name} ({c.token_count} tokens)')

    # Or chunk in-memory source:
    chunks = chunker.chunk_source(source_code, language='python', file_path='<string>')
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken
from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser, UnsupportedLanguageError

logger = logging.getLogger(__name__)

# cl100k_base is the encoding used by GPT-3.5-turbo / GPT-4; it is the
# de-facto standard for RAG token budgets.
_ENCODING_NAME = "cl100k_base"
_ENCODER: tiktoken.Encoding | None = None
_TIKTOKEN_FAILED = False

# Words and standalone punctuation each count as a rough token unit. This
# correlates well with BPE token counts for source code and serves as an
# offline fallback.
_HEURISTIC_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*.

    - **Why it exists**: Centralises token counting. Crucially, provides a
      regex-based heuristic fallback if ``tiktoken`` is unavailable (e.g., in
      offline CI environments without cached encodings).
    - **Algorithm**: Attempts to use tiktoken's BPE encoder. If that fails,
      falls back to a regex that counts words and punctuation marks.
    - **Edge cases**: Empty strings return 0 without invoking the encoder.
    - **Correctness choice**: Uses ``cl100k_base`` (GPT-4 encoding) by default
      so token counts match what downstream embedding steps consume.
    """
    if not text:
        return 0

    global _ENCODER, _TIKTOKEN_FAILED  # noqa: PLW0603
    if not _TIKTOKEN_FAILED:
        try:
            if _ENCODER is None:
                _ENCODER = tiktoken.get_encoding(_ENCODING_NAME)
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception as exc:  # noqa: BLE001
            logger.debug("tiktoken unavailable (%s); falling back to heuristic", exc)
            _TIKTOKEN_FAILED = True

    return len(_HEURISTIC_RE.findall(text))


# ---------------------------------------------------------------------------
# ChunkKind type alias
# ---------------------------------------------------------------------------

ChunkKind = Literal["definition", "module", "continuation"]
"""Discriminator for the structural role of a :class:`Chunk`.

- ``"definition"`` -- the first (or only) chunk of a function or class.
- ``"continuation"`` -- a subsequent split window of an oversized definition,
  distinguished by ``is_continuation=True`` and an ``overlap_header``.
- ``"module"`` -- module-level code (imports, assignments, constants, comments)
  that sits between top-level definitions.
"""


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A contiguous slice of source code with full metadata.

    Attributes:
        content:         The raw source text of this chunk (UTF-8 string).
        file_path:       Path to the originating source file, or ``"<string>"``
                         for in-memory sources.
        language:        Language name (e.g. ``"python"``).
        start_line:      1-based line number where this chunk begins.
        end_line:        1-based line number where this chunk ends (inclusive).
        parent_symbol:   Qualified name of the enclosing function or class
                         (e.g. ``"MyClass"`` for a method chunk), or ``None``
                         for module-level code.
        qualified_name:  Fully qualified name of the symbol this chunk belongs
                         to (e.g. ``"MyClass.my_method"``), or ``None`` for
                         module-level chunks.  Mirrors :attr:`Symbol.qualified_name`
                         so chunks and symbols can be joined by this key.
        chunk_kind:      Structural role of this chunk: ``"definition"``,
                         ``"continuation"``, or ``"module"``.  Derived
                         automatically from *is_continuation* and *parent_symbol*
                         in :meth:`__post_init__`.
        token_count:     Number of BPE tokens in *content* (pre-computed).
        chunk_index:     0-based position of this chunk within its parent
                         symbol.  Always 0 for single-chunk symbols.
        part:            1-based part number when a symbol is split.
        total_parts:     Total number of parts the owning symbol was split into.
        is_continuation: ``True`` when this is not the first chunk of a
                         symbol that was split across multiple chunks.
        overlap_header:  The function/class signature repeated at the start
                         of continuation chunks for embedding context.
                         ``None`` for first chunks and module-level chunks.
        has_parse_error: ``True`` when the source contained a syntax error
                         that caused partial AST extraction.
    """

    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    parent_symbol: str | None = None
    qualified_name: str | None = None
    chunk_kind: ChunkKind = "definition"
    token_count: int = 0
    chunk_index: int = 0
    part: int = 1
    total_parts: int = 1
    is_continuation: bool = False
    overlap_header: str | None = None
    has_parse_error: bool = False

    def __post_init__(self) -> None:
        """Derive computed fields after construction.

        - ``token_count`` is pre-computed via :func:`count_tokens` when the
          caller leaves it at the default ``0``.
        - ``chunk_kind`` is always derived from *is_continuation* and
          *parent_symbol* so callers never need to set it manually.
        """
        if self.token_count == 0 and self.content:
            self.token_count = count_tokens(self.content)
        # Derive chunk_kind from structural fields; callers must not set it.
        if self.is_continuation:
            self.chunk_kind = "continuation"
        elif self.parent_symbol is None and self.qualified_name is None:
            self.chunk_kind = "module"
        else:
            self.chunk_kind = "definition"

    def __repr__(self) -> str:
        label = self.qualified_name or self.parent_symbol or "<module>"
        cont = f" [part {self.part}/{self.total_parts}]" if self.total_parts > 1 else ""
        return (
            f"Chunk({label}{cont} [{self.start_line}-{self.end_line}]"
            f" {self.token_count} tokens)"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this chunk.

        Suitable for storing as a vector-store payload (e.g. Qdrant) or
        writing to JSONL for offline analysis.  All values are JSON-primitive
        types (``str``, ``int``, ``bool``, ``None``).

        Returns:
            A flat dictionary containing every metadata field.
        """
        return {
            "content": self.content,
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_symbol": self.parent_symbol,
            "qualified_name": self.qualified_name,
            "chunk_kind": self.chunk_kind,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "part": self.part,
            "total_parts": self.total_parts,
            "is_continuation": self.is_continuation,
            "overlap_header": self.overlap_header,
            "has_parse_error": self.has_parse_error,
        }


# ---------------------------------------------------------------------------
# Sliding-window accumulator
# ---------------------------------------------------------------------------


class _Accumulator:
    """A stateful sliding window used during body and module-level splits.

    Accumulates statement texts one at a time and signals when the running
    token total would overflow the budget.

    - **Why it exists**: Replaces inner ``_flush`` closures that mutate
      ``nonlocal`` state.  As a proper class the logic is easy to follow and
      straightforward to unit-test in isolation.
    - **Algorithm**: Maintains a *running token count* updated incrementally
      (O(n) amortised) rather than re-encoding the entire accumulated text
      each iteration (O(n^2)).
    - **Correctness choice**: The ``\\n`` joining separator adds negligible
      BPE tokens in cl100k_base, so the running sum matches
      ``count_tokens("\\n".join(texts))`` within +/-1 token -- precise enough
      for a chunking heuristic.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.token_total: int = 0
        self.start_line: int | None = None
        self.end_line: int | None = None

    def is_empty(self) -> bool:
        """Return True when no statements have been added yet."""
        return not self.texts

    def would_overflow(self, extra_tokens: int, budget: int) -> bool:
        """Return True when adding *extra_tokens* would exceed *budget*."""
        return bool(self.texts) and self.token_total + extra_tokens > budget

    def add(self, text: str, tokens: int, start: int, end: int) -> None:
        """Append *text* to the window, updating running counters."""
        if self.start_line is None:
            self.start_line = start
        self.texts.append(text)
        self.token_total += tokens
        self.end_line = end

    def flush(self) -> tuple[str, int, int, int]:
        """Drain the window and return ``(joined_text, start, end, tokens)``.

        Resets internal state so the accumulator is ready for the next window.
        """
        joined = "\n".join(self.texts)
        start = self.start_line
        end = self.end_line
        tokens = self.token_total
        self.texts = []
        self.token_total = 0
        self.start_line = None
        self.end_line = None
        return joined, start, end, tokens  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Python chunker implementation
# ---------------------------------------------------------------------------


class PythonChunker:
    """Produces :class:`Chunk` objects from a Python tree-sitter AST.

    Strategy
    --------
    1. Walk the module's top-level named children in source order.
    2. Each ``function_definition`` / ``class_definition`` (and
       ``decorated_definition``) becomes one or more chunks.
    3. A definition whose token count exceeds *max_tokens* is split at
       statement boundaries inside its body with signature overlap.
    4. Contiguous module-level statements (imports, assignments, comments,
       docstrings, etc.) that fall between definitions are grouped into a
       single ``"module"`` chunk.
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
          order.  Definitions (``function_definition``, ``class_definition``,
          ``decorated_definition``) flush any pending module-level buffer and
          then become their own chunk(s).  Everything else -- imports,
          assignments, comments, docstrings -- accumulates in the pending
          buffer and is flushed as a single ``"module"`` chunk when a
          definition boundary is hit.
        - **Edge cases**: Trailing module-level statements after the last
          definition are flushed after the loop.
        - **Correctness choice**: Uses ``named_children`` to skip anonymous
          punctuation nodes that tree-sitter inserts between statements.
        """
        chunks: list[Chunk] = []
        pending: list[Node] = []

        for node in tree.root_node.named_children:
            if node.type in ("function_definition", "class_definition"):
                if pending:
                    chunks.extend(self._flush_module_level(pending))
                    pending = []
                chunks.extend(self._chunk_definition(node, parent_symbol=None))

            elif node.type == "decorated_definition":
                if pending:
                    chunks.extend(self._flush_module_level(pending))
                    pending = []
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
                    # Malformed decorated_definition: treat as module-level text
                    pending.append(node)

            else:
                # Imports, assignments, expression statements, comments --
                # group all of these together in the module-level buffer.
                pending.append(node)

        if pending:
            chunks.extend(self._flush_module_level(pending))

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
        - **Algorithm**: Computes the full text token count first.  If it fits
          in *max_tokens*, the definition is emitted as a single chunk.  If it
          exceeds the budget, :meth:`_split_definition` is called to split at
          statement boundaries.  For class definitions, individual method
          chunks are also emitted in either case so each method gets its own
          embedding.
        - **Edge cases**: If even the signature alone exceeds the budget, the
          chunk is still emitted -- one chunk per definition is the hard lower
          bound.
        - **Correctness choice**: Token budget is enforced on the final text
          (with overlap header prepended) so continuation chunks are never
          silently oversized.
        """
        qualified_name = self._qualified_name(node, parent_symbol)
        has_error = node.has_error or node.is_missing

        # Use the decorated definition span so decorators are included
        span_node = decorator_node if decorator_node is not None else node
        full_text = self._node_text(span_node)
        full_tokens = count_tokens(full_text)

        if full_tokens <= self.max_tokens:
            # Happy path: entire definition fits in one chunk
            first_chunk = Chunk(
                content=full_text,
                file_path=self.file_path,
                language=self.language,
                start_line=span_node.start_point[0] + 1,
                end_line=span_node.end_point[0] + 1,
                parent_symbol=parent_symbol,
                qualified_name=qualified_name,
                token_count=full_tokens,
                chunk_index=0,
                is_continuation=False,
                has_parse_error=has_error,
            )
            if node.type == "class_definition":
                # Also emit individual method chunks for fine-grained retrieval
                return [first_chunk] + self._chunk_class_methods(
                    node, parent_symbol=qualified_name
                )
            return [first_chunk]

        # Oversized: split at statement boundaries
        logger.debug(
            "Splitting '%s' (%d tokens > budget %d) in %s",
            qualified_name,
            full_tokens,
            self.max_tokens,
            self.file_path,
        )
        split_chunks = self._split_definition(
            node,
            span_node=span_node,
            qualified_name=qualified_name,
            parent_symbol=parent_symbol,
            has_error=has_error,
        )
        if node.type == "class_definition":
            # Even when the class itself is too large to fit in one chunk,
            # emit individual method chunks so each is independently retrievable
            split_chunks = split_chunks + self._chunk_class_methods(
                node, parent_symbol=qualified_name
            )
        return split_chunks

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
          as a single chunk that blows the token budget.
        - **Algorithm**: Extracts the signature header (source bytes from the
          span_node start up to the body block start).  Then iterates over
          body statements using :class:`_Accumulator` to maintain an O(n)
          running token count.  When adding the next statement would exceed
          ``max_tokens - header_tokens``, the accumulator is flushed as a
          chunk and a new window begins with the overlap header prepended.
        - **Edge cases**: If the body node is missing or empty the full text
          is emitted as a single (possibly oversized) chunk.  If a single
          statement exceeds the budget it is still emitted alone.
        - **Correctness choice**: The overlap header is prepended before token
          counting so the budget includes the header cost, preventing
          continuation chunks from silently exceeding the limit.
        """
        body_node = node.child_by_field_name("body")
        if body_node is None or not body_node.named_children:
            full_text = self._node_text(span_node)
            return [
                Chunk(
                    content=full_text,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=span_node.start_point[0] + 1,
                    end_line=span_node.end_point[0] + 1,
                    parent_symbol=parent_symbol,
                    qualified_name=qualified_name,
                    token_count=count_tokens(full_text),
                    chunk_index=0,
                    is_continuation=False,
                    has_parse_error=has_error,
                )
            ]

        overlap_header = self._extract_header(node, span_node)
        header_tokens = count_tokens(overlap_header)
        budget = self.max_tokens - header_tokens

        chunks: list[Chunk] = []
        chunk_index = 0
        acc = _Accumulator()

        def _emit() -> None:
            nonlocal chunk_index
            body_text, w_start, w_end, _ = acc.flush()
            is_cont = chunk_index > 0
            sep = "  # ... continued" if is_cont else ""
            content = f"{overlap_header}{sep}\n{body_text}"
            chunks.append(
                Chunk(
                    content=content,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=w_start or (span_node.start_point[0] + 1),
                    end_line=w_end or (span_node.end_point[0] + 1),
                    parent_symbol=parent_symbol,
                    qualified_name=qualified_name,
                    token_count=count_tokens(content),
                    chunk_index=chunk_index,
                    is_continuation=is_cont,
                    overlap_header=overlap_header,
                    has_parse_error=has_error,
                )
            )
            chunk_index += 1

        for stmt in body_node.named_children:
            stmt_text = self._node_text(stmt)
            stmt_tokens = count_tokens(stmt_text)

            if acc.would_overflow(stmt_tokens, budget):
                _emit()

            acc.add(
                stmt_text,
                stmt_tokens,
                stmt.start_point[0] + 1,
                stmt.end_point[0] + 1,
            )

        if not acc.is_empty():
            _emit()

        # Update total_parts and part for all chunks in this split
        total = len(chunks)
        for c in chunks:
            c.total_parts = total
            c.part = c.chunk_index + 1

        return chunks

    def _chunk_class_methods(
        self, class_node: Node, *, parent_symbol: str
    ) -> list[Chunk]:
        """Recursively chunk methods inside a class body.

        - **Why it exists**: Classes emit a full-class chunk (for context) and
          individual method chunks so each method gets its own embedding.
        - **Algorithm**: Iterates over the class body's direct children,
          forwarding each function or decorated definition to
          :meth:`_chunk_definition`.  Nested classes recurse through
          :meth:`_chunk_definition` which calls this method again.
        - **Edge cases**: Missing body returns an empty list immediately.
        - **Correctness choice**: Methods are emitted *after* the class chunk,
          not instead of it, so the class docstring and inheritance context
          remain available for retrieval.
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
                # Nested class: emit its own chunk + its methods recursively
                chunks.extend(
                    self._chunk_definition(child, parent_symbol=parent_symbol)
                )

        return chunks

    # ------------------------------------------------------------------
    # Module-level flushing
    # ------------------------------------------------------------------

    def _flush_module_level(self, nodes: list[Node]) -> list[Chunk]:
        """Group a run of module-level nodes into one or more ``"module"`` chunks.

        - **Why it exists**: Preserves module-level code (imports, assignments,
          ``__all__``, constants, comments) as retrievable chunks without
          fragmenting each node into its own tiny chunk.
        - **Algorithm**: Concatenates node texts with newlines.  If the total
          fits within *max_tokens*, a single chunk is produced.  If not, nodes
          are accumulated with :class:`_Accumulator` (O(n) token tracking) and
          flushed at node boundaries into multiple continuation chunks.
        - **Edge cases**: Empty node list returns an empty list immediately.
        - **Correctness choice**: Module-level chunks have ``parent_symbol=None``
          and ``qualified_name=None`` to distinguish them from definition chunks.
          ``chunk_kind`` is automatically set to ``"module"`` by
          :meth:`Chunk.__post_init__`.
        """
        if not nodes:
            return []

        node_texts = [self._node_text(n) for n in nodes]
        full_text = "\n".join(node_texts)
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
                    qualified_name=None,
                    token_count=full_tokens,
                    chunk_index=0,
                    is_continuation=False,
                )
            ]

        # Split at node boundaries using O(n) accumulator
        chunks: list[Chunk] = []
        chunk_index = 0
        acc = _Accumulator()

        def _emit() -> None:
            nonlocal chunk_index
            content, w_start, w_end, tokens = acc.flush()
            chunks.append(
                Chunk(
                    content=content,
                    file_path=self.file_path,
                    language=self.language,
                    start_line=w_start or (nodes[0].start_point[0] + 1),
                    end_line=w_end or (nodes[-1].end_point[0] + 1),
                    parent_symbol=None,
                    qualified_name=None,
                    token_count=tokens,
                    chunk_index=chunk_index,
                    is_continuation=chunk_index > 0,
                )
            )
            chunk_index += 1

        for node, text in zip(nodes, node_texts, strict=False):
            node_tokens = count_tokens(text)

            if acc.would_overflow(node_tokens, self.max_tokens):
                _emit()

            acc.add(
                text,
                node_tokens,
                node.start_point[0] + 1,
                node.end_point[0] + 1,
            )

        if not acc.is_empty():
            _emit()

        # Update total_parts and part for all chunks in this split
        total = len(chunks)
        for c in chunks:
            c.total_parts = total
            c.part = c.chunk_index + 1

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node_text(self, node: Node) -> str:
        """Decode a node's exact source bytes to a UTF-8 string.

        - **Why it exists**: Slices the original source buffer by byte offsets
          so the returned text is exactly what the author wrote, including
          comments and spacing.
        - **Correctness choice**: Uses ``start_byte`` / ``end_byte`` rather
          than ``node.text`` so multi-line nodes with embedded newlines are
          handled faithfully on all platforms.
        """
        return self.source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def _extract_header(self, node: Node, span_node: Node) -> str:
        """Extract the signature header for use as an overlap prefix.

        - **Why it exists**: Continuation chunks need scope context to be
          useful standalone.  Repeating the signature (and decorator, if
          present) gives the embedding model enough context to understand the
          enclosing scope.
        - **Algorithm**: Slices source bytes from the span_node start up to
          (but not including) the body block start byte.
        - **Edge cases**: If the body node is missing, falls back to the first
          line of the definition.
        - **Correctness choice**: Byte-level slicing preserves the exact
          formatting of the original source, including type annotations and
          multi-line parameter lists.
        """
        body = node.child_by_field_name("body")
        if body is None:
            return self._node_text(span_node).splitlines()[0]

        header_bytes = self.source_bytes[span_node.start_byte : body.start_byte]
        return header_bytes.decode("utf-8", errors="replace").rstrip()

    @staticmethod
    def _qualified_name(node: Node, parent_symbol: str | None) -> str:
        """Build a dot-qualified name for a definition node.

        - **Why it exists**: Downstream consumers (knowledge graph, embedding
          index) need globally unique symbol names to avoid collisions between
          ``MyClass.run`` and ``OtherClass.run``.
        - **Algorithm**: Reads the ``name`` field child using tree-sitter's
          field API, then prepends the parent prefix.
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
          an outer node.  We need the inner ``function_definition`` or
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
# Public coordinator
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

        logger.debug(
            "Chunking %s (%d bytes, language=%s)", fpath, len(source_bytes), language
        )
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

    def chunk_from_tree(
        self,
        tree: Tree,
        file_path: str,
        source: str | bytes,
        language: str = "python",
    ) -> list[Chunk]:
        """Produce semantic chunks from an already-parsed tree-sitter Tree.

        Use this when the caller already has a parsed tree (e.g. the symbol
        extraction and chunking pass share one parse call for efficiency).
        The signature intentionally mirrors
        :meth:`~src.reporag.ingestion.symbol_extractor.SymbolExtractor.extract_from_tree`
        so both can be called with the same arguments.

        Args:
            tree:      A :class:`~tree_sitter.Tree` from
                       :meth:`~src.reporag.ingestion.parser.ASTParser.parse`.
            file_path: Path label stored in each :class:`Chunk`.
            source:    Original source code as ``str`` or ``bytes``.
            language:  Language name stored in each :class:`Chunk`.

        Returns:
            Ordered list of :class:`Chunk` objects.

        Raises:
            UnsupportedLanguageError: If no chunker is registered for
                *language*.
        """
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
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
          so new languages can be added to ``_CHUNKER_REGISTRY`` without
          touching the coordinator.
        - **Algorithm**: Lower-cases *language*, looks up the registered
          chunker class, instantiates it with the source bytes, and calls
          ``chunk(tree)``.
        - **Edge cases**: Unsupported languages raise
          :exc:`~src.reporag.ingestion.parser.UnsupportedLanguageError`
          rather than silently returning an empty list, because a missing
          chunker is a configuration error, not expected input.
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

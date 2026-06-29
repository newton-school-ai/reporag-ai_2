"""Semantic, AST-aware code chunker.

Naive text chunkers split functions and classes mid-body, which destroys the
semantic coherence that downstream embeddings rely on. This module chunks
source code at *syntax-tree boundaries* instead: every function, class, and
top-level block is kept whole whenever it fits inside the configured token
budget. Only constructs that are genuinely larger than ``max_tokens`` are
split, and they are split at logical points (statement boundaries) with the
enclosing signature repeated as overlap so each continuation chunk still
carries the context an embedding model needs.

Pipeline position
-----------------
``cloner -> parser -> symbol_extractor -> chunker -> embedder``

The chunker reuses the Issue-6 :class:`~src.reporag.ingestion.parser.ASTParser`
for parsing and the same ``settings.extension_map`` used across the ingestion
pipeline for language inference, so its behaviour is consistent with the rest
of the system.

Guarantees
----------
- A function/class/method is never cut mid-body unless it alone exceeds
  ``max_tokens``.
- Oversized constructs are split at top-level statement boundaries, with the
  signature re-emitted as overlap on every continuation chunk.
- No chunk exceeds ``max_tokens * (1 + size_tolerance)`` (default +10%) for any
  input whose individual source lines fit the budget.
- Every chunk carries: ``file_path``, ``language``, ``start_line``,
  ``end_line``, ``parent_symbol``, ``symbol_type``, and ``token_count``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from math import floor
from pathlib import Path

from tree_sitter import Node, Tree

from src.reporag.config import settings
from src.reporag.ingestion.parser import ASTParser, ParseError, UnsupportedLanguageError

logger = logging.getLogger(__name__)

# Tree-sitter Python node types that introduce a new lexical scope and should
# be treated as indivisible semantic units (kept whole when they fit).
_DEF_TYPES = frozenset(
    {"function_definition", "class_definition", "decorated_definition"}
)
_FUNC_TYPES = frozenset({"function_definition"})
_CLASS_TYPES = frozenset({"class_definition"})

# Symbol-type tags carried on each chunk's ``symbol_type`` field.
FUNCTION = "function"
METHOD = "method"
CLASS = "class"
MODULE = "module"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TokenCounter:
    """Count tokens in a string, using ``tiktoken`` when it is available.

    ``tiktoken`` gives counts that match how OpenAI-family models tokenise,
    which is the right budget for embedding/context windows. Loading an
    encoding can require a one-time download, so any failure (offline CI,
    missing cache) falls back to a deterministic regex heuristic that
    approximates sub-word tokenisation closely enough for chunk sizing.
    """

    # Words and standalone punctuation each count as a rough token unit. This
    # correlates well with BPE token counts for source code.
    _HEURISTIC_RE = re.compile(r"\w+|[^\w\s]")

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialise the counter, attempting to load a tiktoken encoding."""
        self._encoder = None
        try:
            import tiktoken  # noqa: PLC0415

            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # noqa: BLE001 - any failure -> heuristic
            logger.debug(
                "tiktoken unavailable (%s); using heuristic token counter", exc
            )

    def count(self, text: str) -> int:
        """Return the token count of *text*."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return len(self._HEURISTIC_RE.findall(text))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single embedding-ready unit of code with provenance metadata.

    Attributes:
        text: The chunk's source text (may include a synthetic overlap header
            on continuation chunks; see ``is_continuation``).
        file_path: Path of the originating source file.
        language: Canonical language name (e.g. ``"python"``).
        start_line: 1-based first source line covered by this chunk's code.
        end_line: 1-based last source line covered by this chunk's code.
        token_count: Token count of ``text`` (per the active TokenCounter).
        parent_symbol: Fully-qualified name of the symbol this chunk belongs to
            (e.g. ``"User.save"``), or ``None`` for module-level code.
        symbol_type: One of ``"function"``, ``"method"``, ``"class"``,
            ``"module"``.
        chunk_index: 0-based position of this chunk within the file.
        part: 1-based part number when a symbol is split across chunks.
        total_parts: Total number of parts the owning symbol was split into.
        is_continuation: ``True`` if this chunk is a non-first part and its
            ``text`` is prefixed with a repeated signature for overlap.
    """

    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    token_count: int
    parent_symbol: str | None = None
    symbol_type: str = MODULE
    chunk_index: int = 0
    part: int = 1
    total_parts: int = 1
    is_continuation: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class SemanticChunker:
    """Split source code into semantically coherent, token-bounded chunks.

    A single instance is reusable across an entire repository. Parsing is
    delegated to a shared :class:`ASTParser`; token counting is pluggable.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        *,
        size_tolerance: float = 0.1,
        overlap_signature: bool = True,
        parser: ASTParser | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Configure the chunker.

        Args:
            max_tokens: Target maximum tokens per chunk.
            size_tolerance: Fractional slack above ``max_tokens`` a chunk may
                reach before it must be split (default 0.1 -> +10%).
            overlap_signature: If ``True``, continuation chunks of a split
                symbol are prefixed with the symbol's signature for context.
            parser: Optional shared :class:`ASTParser`.
            token_counter: Optional ``callable(str) -> int`` overriding the
                default :class:`TokenCounter`.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.size_tolerance = max(0.0, size_tolerance)
        self.overlap_signature = overlap_signature
        self._parser = parser or ASTParser()
        self._count: Callable[[str], int] = token_counter or TokenCounter().count
        # Hard ceiling: a chunk may grow up to this many tokens before it is
        # forced to split. Keeps chunks within max_tokens +/- tolerance.
        self._max_allowed = floor(max_tokens * (1 + self.size_tolerance))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> list[Chunk]:
        """Read a file from disk and chunk it.

        Language is inferred from the file extension via
        ``settings.extension_map`` unless *language* is given explicitly.

        Raises:
            UnsupportedLanguageError: If the language cannot be inferred or is
                not supported.
            ParseError: If the file cannot be read.
        """
        path = Path(file_path)
        if language is None:
            language = settings.extension_map.get(path.suffix.lower())
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for extension '{path.suffix}'. "
                    f"Pass language= explicitly or extend settings.extension_map."
                )
        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            raise ParseError(f"Cannot read file '{path}': {exc}") from exc
        return self.chunk_source(
            source_bytes, language=language, file_path=str(file_path)
        )

    def chunk_source(
        self,
        source: str | bytes,
        language: str = "python",
        file_path: str = "<string>",
    ) -> list[Chunk]:
        """Chunk raw source code.

        Raises:
            UnsupportedLanguageError: If *language* is not Python (the only
                language whose grammar this chunker currently understands).
        """
        lang = language.lower().strip()
        if lang != "python":
            raise UnsupportedLanguageError(
                f"SemanticChunker currently supports only Python, not '{language}'."
            )
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        tree = self._parser.parse(source_bytes, language="python")
        if tree.root_node.has_error:
            logger.debug("Chunking '%s' with parse errors present", file_path)
        return self.chunk_tree(
            tree, source_bytes, file_path=file_path, language="python"
        )

    def chunk_tree(
        self,
        tree: Tree,
        source_bytes: bytes,
        file_path: str = "<string>",
        language: str = "python",
    ) -> list[Chunk]:
        """Chunk an already-parsed tree-sitter ``Tree``.

        Top-level definitions become semantic chunks; runs of loose
        module-level statements (imports, assignments, ``if __name__`` blocks)
        are greedily packed into module chunks. Final chunks are returned in
        source order with ``chunk_index`` assigned sequentially.
        """
        ctx = _FileCtx(source_bytes, file_path, language)
        chunks: list[Chunk] = []
        module_buffer: list[Node] = []

        for node in tree.root_node.named_children:
            if node.type in _DEF_TYPES:
                chunks.extend(self._flush_module(module_buffer, ctx))
                module_buffer = []
                chunks.extend(self._chunk_definition(node, ctx, owner=None))
            else:
                module_buffer.append(node)

        chunks.extend(self._flush_module(module_buffer, ctx))

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    # ------------------------------------------------------------------
    # Definition chunking (functions / classes / methods)
    # ------------------------------------------------------------------

    def _chunk_definition(
        self, node: Node, ctx: _FileCtx, owner: str | None
    ) -> list[Chunk]:
        """Chunk a single function/class/method definition node.

        *owner* is the qualified name of the enclosing class when this is a
        method (``None`` at module level). The whole construct is emitted as
        one chunk when it fits; otherwise it is split.
        """
        full, inner = self._unwrap(node)
        if inner is None:
            # Decorated non-definition (rare) - treat as opaque source.
            return self._emit_span(full.start_byte, full.end_byte, ctx, owner, MODULE)

        name = self._def_name(inner)
        qualified = f"{owner}.{name}" if owner else name
        is_class = inner.type in _CLASS_TYPES
        if is_class:
            symbol_type = CLASS
        elif owner:
            symbol_type = METHOD
        else:
            symbol_type = FUNCTION

        text = ctx.text(full)
        tokens = self._count(text)
        if tokens <= self._max_allowed:
            return [
                Chunk(
                    text=text,
                    file_path=ctx.file_path,
                    language=ctx.language,
                    start_line=full.start_point[0] + 1,
                    end_line=full.end_point[0] + 1,
                    token_count=tokens,
                    parent_symbol=qualified,
                    symbol_type=symbol_type,
                )
            ]

        if is_class:
            return self._split_class(full, inner, ctx, qualified)
        return self._split_function(full, inner, ctx, qualified, symbol_type)

    def _split_class(
        self, full: Node, inner: Node, ctx: _FileCtx, qualified: str
    ) -> list[Chunk]:
        """Split an oversized class into a header chunk plus per-method chunks.

        The class header (decorators, ``class`` line, docstring, and any
        class-level statements that precede a method) forms its own chunk(s);
        each method is then chunked independently and may itself be split,
        preserving method boundaries.
        """
        body = inner.child_by_field_name("body")
        if body is None:
            return self._emit_span(
                full.start_byte, full.end_byte, ctx, qualified, CLASS
            )

        chunks: list[Chunk] = []
        # Header spans from the start of the (possibly decorated) definition up
        # to the first method, so the class signature/docstring lead the output.
        header_start = full.start_byte
        header_end = body.start_byte

        for child in body.named_children:
            if child.type in _DEF_TYPES:
                if header_end > header_start:
                    chunks.extend(
                        self._emit_span(header_start, header_end, ctx, qualified, CLASS)
                    )
                header_start = child.end_byte
                header_end = child.end_byte
                chunks.extend(self._chunk_definition(child, ctx, owner=qualified))
            else:
                header_end = child.end_byte

        if header_end > header_start:
            chunks.extend(
                self._emit_span(header_start, header_end, ctx, qualified, CLASS)
            )
        # The class header and each method are distinct symbols, so they keep
        # their own part numbering rather than being renumbered as one split.
        return chunks

    def _split_function(
        self, full: Node, inner: Node, ctx: _FileCtx, qualified: str, symbol_type: str
    ) -> list[Chunk]:
        """Split an oversized function/method at top-level statement points.

        Part 1 keeps the real signature/decorators; continuation parts repeat
        a compact signature as overlap so each chunk is self-describing.
        """
        body = inner.child_by_field_name("body")
        statements = list(body.named_children) if body is not None else []
        if body is None or not statements:
            return self._emit_span(
                full.start_byte, full.end_byte, ctx, qualified, symbol_type
            )

        signature = self._signature(inner, ctx)
        header_text = ctx.slice(full.start_byte, body.start_byte).rstrip()
        # Worst-case-width overlap header used only for sizing, so the packing
        # estimate is never smaller than what is actually emitted.
        sizing_cont = self._overlap(signature, 9999, 9999)
        groups = self._pack(statements, ctx, header_text + "\n", sizing_cont)

        chunks: list[Chunk] = []
        total = len(groups)
        for part_no, (gi, gj) in enumerate(groups, start=1):
            is_cont = part_no > 1
            body_text = self._group_slice(statements, gi, gj, ctx)
            if is_cont:
                text = f"{self._overlap(signature, part_no, total)}{body_text}"
            else:
                text = f"{header_text}\n{body_text}"
            if gi == gj and self._count(text) > self._max_allowed:
                # A single statement larger than the ceiling: hard line-split.
                chunks.extend(
                    self._line_split(
                        text,
                        statements[gi].start_point[0] + 1,
                        signature,
                        ctx,
                        qualified,
                        symbol_type,
                    )
                )
                continue
            chunks.append(
                Chunk(
                    text=text,
                    file_path=ctx.file_path,
                    language=ctx.language,
                    start_line=statements[gi].start_point[0] + 1,
                    end_line=statements[gj].end_point[0] + 1,
                    token_count=self._count(text),
                    parent_symbol=qualified,
                    symbol_type=symbol_type,
                    part=part_no,
                    total_parts=total,
                    is_continuation=is_cont,
                )
            )
        return self._renumber(chunks)

    def _overlap(self, signature: str, part_no: int, total: int) -> str:
        """Build the synthetic overlap header for a continuation chunk."""
        if not self.overlap_signature:
            return ""
        return f"{signature}\n    # ... continued (part {part_no}/{total})\n"

    # ------------------------------------------------------------------
    # Module-level packing
    # ------------------------------------------------------------------

    def _flush_module(self, nodes: list[Node], ctx: _FileCtx) -> list[Chunk]:
        """Greedily pack consecutive module-level statements into chunks."""
        if not nodes:
            return []
        chunks: list[Chunk] = []
        for gi, gj in self._pack(nodes, ctx, "", ""):
            chunks.extend(
                self._emit_span(
                    nodes[gi].start_byte, nodes[gj].end_byte, ctx, None, MODULE
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Packing and splitting primitives
    # ------------------------------------------------------------------

    def _pack(
        self,
        nodes: list[Node],
        ctx: _FileCtx,
        prefix_first: str,
        prefix_cont: str,
    ) -> list[tuple[int, int]]:
        """Greedily group contiguous *nodes* into index ranges under the ceiling.

        Each group's rendered text (``prefix`` + source slice) is measured
        directly, so the emitted chunk is guaranteed to stay within the token
        ceiling. Every group contains at least one node, so a single oversized
        node forms its own group (to be line-split downstream).
        """
        groups: list[tuple[int, int]] = []
        i = 0
        n = len(nodes)
        while i < n:
            prefix = prefix_first if not groups else prefix_cont
            j = i
            while j + 1 < n:
                body = self._group_slice(nodes, i, j + 1, ctx)
                if self._count(prefix + body) > self._max_allowed:
                    break
                j += 1
            groups.append((i, j))
            i = j + 1
        return groups

    def _group_slice(
        self, nodes: list[Node], start_idx: int, end_idx: int, ctx: _FileCtx
    ) -> str:
        """Return the source slice covering ``nodes[start_idx..end_idx]``.

        The slice begins at column 0 of the first node's line so leading
        indentation is preserved in the rendered chunk.
        """
        line_start = ctx.line_start_byte(nodes[start_idx].start_byte)
        return ctx.slice(line_start, nodes[end_idx].end_byte)

    def _emit_span(
        self,
        start_byte: int,
        end_byte: int,
        ctx: _FileCtx,
        qualified: str | None,
        symbol_type: str,
    ) -> list[Chunk]:
        """Emit a byte span as one chunk, line-splitting it if oversized."""
        if end_byte <= start_byte:
            return []
        text = ctx.slice(start_byte, end_byte)
        if not text.strip():
            return []
        start_line = ctx.line_of(start_byte)
        tokens = self._count(text)
        if tokens <= self._max_allowed:
            end_line = start_line + text.rstrip("\n").count("\n")
            return [
                Chunk(
                    text=text,
                    file_path=ctx.file_path,
                    language=ctx.language,
                    start_line=start_line,
                    end_line=end_line,
                    token_count=tokens,
                    parent_symbol=qualified,
                    symbol_type=symbol_type,
                )
            ]
        return self._line_split(text, start_line, "", ctx, qualified, symbol_type)

    def _line_split(
        self,
        text: str,
        abs_start_line: int,
        overlap: str,
        ctx: _FileCtx,
        qualified: str | None,
        symbol_type: str,
    ) -> list[Chunk]:
        """Hard-split *text* into windows of whole lines under the ceiling.

        Last-resort splitter for a single statement larger than the token
        ceiling. Whole lines are never broken, so the only way to exceed the
        ceiling is a single line that does so on its own.
        """
        lines = text.split("\n")
        windows: list[tuple[int, list[str]]] = []
        current: list[str] = []
        current_tokens = 0
        window_start = 0
        for offset, line in enumerate(lines):
            line_tokens = self._count(line) + 1
            if current and current_tokens + line_tokens > self._max_allowed:
                windows.append((window_start, current))
                current = []
                current_tokens = 0
                window_start = offset
            current.append(line)
            current_tokens += line_tokens
        if current:
            windows.append((window_start, current))

        chunks: list[Chunk] = []
        total = len(windows)
        for part_no, (offset, window_lines) in enumerate(windows, start=1):
            is_cont = part_no > 1
            prefix = (
                self._overlap(overlap, part_no, total) if is_cont and overlap else ""
            )
            chunk_text = f"{prefix}{chr(10).join(window_lines)}"
            chunks.append(
                Chunk(
                    text=chunk_text,
                    file_path=ctx.file_path,
                    language=ctx.language,
                    start_line=abs_start_line + offset,
                    end_line=abs_start_line + offset + len(window_lines) - 1,
                    token_count=self._count(chunk_text),
                    parent_symbol=qualified,
                    symbol_type=symbol_type,
                    part=part_no,
                    total_parts=total,
                    is_continuation=is_cont,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(node: Node) -> tuple[Node, Node | None]:
        """Return ``(full_node, inner_definition)`` for a definition node.

        For a ``decorated_definition`` the full node spans the decorators while
        the inner node is the underlying function/class. Returns ``(node,
        None)`` if no inner function/class definition is present.
        """
        if node.type != "decorated_definition":
            return node, node
        for child in node.named_children:
            if child.type in _FUNC_TYPES or child.type in _CLASS_TYPES:
                return node, child
        return node, None

    @staticmethod
    def _def_name(inner: Node) -> str:
        """Return the declared name of a function/class definition node."""
        name_node = inner.child_by_field_name("name")
        if name_node is None or name_node.text is None:
            return "<anonymous>"
        return name_node.text.decode("utf-8", errors="replace")

    def _signature(self, inner: Node, ctx: _FileCtx) -> str:
        """Return a compact single-line signature for overlap headers."""
        body = inner.child_by_field_name("body")
        end = body.start_byte if body is not None else inner.end_byte
        raw = ctx.slice(inner.start_byte, end).strip()
        return " ".join(raw.split())

    @staticmethod
    def _renumber(chunks: list[Chunk]) -> list[Chunk]:
        """Recompute ``part``/``total_parts`` so a split reads 1..N of N."""
        total = len(chunks)
        if total > 1:
            for index, chunk in enumerate(chunks, start=1):
                chunk.part = index
                chunk.total_parts = total
                chunk.is_continuation = index > 1
        return chunks


# ---------------------------------------------------------------------------
# Per-file decoding context
# ---------------------------------------------------------------------------


@dataclass
class _FileCtx:
    """Bundles the source bytes and file metadata for one chunking pass.

    Carrying these together keeps the chunker's internal method signatures
    small and makes byte/line accounting consistent in one place.
    """

    source_bytes: bytes
    file_path: str
    language: str

    def slice(self, start: int, end: int) -> str:
        """Decode a byte slice of the source as UTF-8 text."""
        return self.source_bytes[start:end].decode("utf-8", errors="replace")

    def text(self, node: Node) -> str:
        """Return the decoded source text of *node*."""
        return self.slice(node.start_byte, node.end_byte)

    def line_of(self, byte_offset: int) -> int:
        """Return the 1-based line number containing *byte_offset*."""
        return self.source_bytes[:byte_offset].count(b"\n") + 1

    def line_start_byte(self, byte_offset: int) -> int:
        """Return the byte offset of column 0 of *byte_offset*'s line."""
        return self.source_bytes.rfind(b"\n", 0, byte_offset) + 1


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def chunk_source(
    source: str | bytes,
    language: str = "python",
    file_path: str = "<string>",
    max_tokens: int = 512,
) -> list[Chunk]:
    """Convenience wrapper: chunk *source* with a one-off chunker."""
    return SemanticChunker(max_tokens=max_tokens).chunk_source(
        source, language=language, file_path=file_path
    )

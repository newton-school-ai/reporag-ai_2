"""Context assembler.

Transforms raw retrieval results into a structured, deduplicated context
block for the LLM prompt. Orders chunks by file and line, merges overlaps,
and truncates to fit the context window.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Default token budget if none supplied
_DEFAULT_MAX_TOKENS = 4000

# Overlap merge threshold: if two chunks within the same file overlap by this
# fraction of the smaller chunk's line range, they are merged into one.
_OVERLAP_THRESHOLD = 0.5

# Approximate characters-per-token ratio used for token counting without a
# real tokenizer.  4 chars ~= 1 GPT-4 token for English/code prose.
_CHARS_PER_TOKEN = 4


@dataclass
class AssembledContext:
    """The result of assembling retrieval results into a formatted context block.

    Attributes:
        text: The formatted context string ready to be injected into a prompt.
        total_tokens: Approximate token count of ``text``.
        chunk_count: Number of (merged) chunks included in the output.
        truncated: ``True`` when some chunks were dropped to fit ``max_tokens``.
    """

    text: str
    total_tokens: int
    chunk_count: int
    truncated: bool


@dataclass
class _Chunk:
    """Internal representation of a single context chunk before formatting."""

    file_path: str
    start_line: int
    end_line: int
    code: str
    score: float
    language: str = field(default="")

    @property
    def line_span(self) -> int:
        return max(self.end_line - self.start_line, 0)


def _infer_language(file_path: str) -> str:
    """Infer a markdown fence language tag from the file extension."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "go": "go",
        "rs": "rust",
        "java": "java",
        "rb": "ruby",
        "sh": "bash",
        "md": "markdown",
    }.get(ext, "")


def _approximate_tokens(text: str) -> int:
    """Approximate the token count of a string using char-per-token ratio."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _overlap_fraction(a: _Chunk, b: _Chunk) -> float:
    """Return the overlap fraction of two same-file chunks relative to the smaller.

    Overlap fraction is defined as::

        len(overlap) / len(smaller_span)

    where ``len(x)`` is measured in source lines.
    """
    overlap_start = max(a.start_line, b.start_line)
    overlap_end = min(a.end_line, b.end_line)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_lines = overlap_end - overlap_start
    smaller = min(a.line_span, b.line_span)
    if smaller == 0:
        return 1.0
    return overlap_lines / smaller


def _merge_two(a: _Chunk, b: _Chunk) -> _Chunk:
    """Merge two overlapping chunks into one, preserving all lines."""
    # Determine which chunk starts first so we can splice the texts correctly.
    first, second = (a, b) if a.start_line <= b.start_line else (b, a)

    if second.start_line <= first.end_line:
        # True overlap: take first chunk's text up to where second begins, then
        # append second chunk's text (which covers the tail region).
        first_lines = first.code.splitlines()
        lines_before_overlap = second.start_line - first.start_line
        prefix = "\n".join(first_lines[:lines_before_overlap])
        merged_code = (prefix + "\n" + second.code).lstrip("\n")
    else:
        # Adjacent (no real overlap) -- just concatenate with a blank line gap.
        merged_code = first.code.rstrip("\n") + "\n\n" + second.code.lstrip("\n")

    return _Chunk(
        file_path=first.file_path,
        start_line=first.start_line,
        end_line=max(first.end_line, second.end_line),
        code=merged_code,
        score=max(a.score, b.score),
        language=first.language,
    )


def _merge_chunks(
    chunks: list[_Chunk], overlap_threshold: float = _OVERLAP_THRESHOLD
) -> list[_Chunk]:
    """Merge overlapping chunks within each file (single pass, sorted order)."""
    if not chunks:
        return []

    merged: list[_Chunk] = [chunks[0]]
    for current in chunks[1:]:
        prev = merged[-1]
        if (
            prev.file_path == current.file_path
            and _overlap_fraction(prev, current) >= overlap_threshold
        ):
            merged[-1] = _merge_two(prev, current)
        else:
            merged.append(current)
    return merged


def _format_chunk(chunk: _Chunk) -> str:
    """Format a single chunk into the canonical context block."""
    header = f"## {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
    fence = f"```{chunk.language}" if chunk.language else "```"
    return f"{header}\n{fence}\n{chunk.code.rstrip()}\n```"


class ContextAssembler:
    """Assembles raw retrieval results into a structured LLM prompt context block.

    Pipeline:
    1. Convert ``RetrievalResult`` objects to internal ``_Chunk`` records,
       discarding results without line-number information.
    2. **Prioritise by score** -- highest-ranked chunks are admitted first when
       the token budget is limited.
    3. **Sort by reading order** -- file path, then ``start_line``.
    4. **Merge overlapping chunks** -- adjacent or overlapping chunks within the
       same file are fused into one block when their line-level overlap exceeds
       the :attr:`overlap_threshold`.
    5. **Truncate to token budget** -- chunks are added in order until
       ``max_tokens`` would be exceeded; remaining chunks are dropped.
    6. **Format** -- each surviving chunk is rendered as a markdown code block
       with a ``## file (lines N-M)`` header.

    Args:
        max_tokens: Maximum approximate token budget for the assembled output.
        overlap_threshold: Fraction of the smaller chunk that must overlap before
            two chunks are merged (default ``0.5`` = 50 %).
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        overlap_threshold: float = _OVERLAP_THRESHOLD,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}")
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError(
                f"overlap_threshold must be in [0.0, 1.0], got {overlap_threshold!r}"
            )
        self.max_tokens = max_tokens
        self.overlap_threshold = overlap_threshold

    def assemble(
        self,
        results: Sequence[RetrievalResult],
    ) -> AssembledContext:
        """Assemble retrieval results into a single structured context block.

        Args:
            results: Retrieval results from any retriever (vector, BM25, graph).

        Returns:
            An :class:`AssembledContext` with the formatted text and metadata.
        """
        if not results:
            return AssembledContext(
                text="", total_tokens=0, chunk_count=0, truncated=False
            )

        # 1. Convert to internal chunks, skip results without line metadata.
        raw_chunks: list[_Chunk] = []
        for r in results:
            if r.start_line is None or r.end_line is None:
                logger.debug("Skipping result without line metadata: %s", r.file_path)
                continue
            if not r.chunk_text:
                continue
            raw_chunks.append(
                _Chunk(
                    file_path=r.file_path or "",
                    start_line=r.start_line,
                    end_line=r.end_line,
                    code=r.chunk_text,
                    score=r.score,
                    language=_infer_language(r.file_path or ""),
                )
            )

        if not raw_chunks:
            return AssembledContext(
                text="", total_tokens=0, chunk_count=0, truncated=False
            )

        # 2. Sort by score descending for priority-aware truncation later.
        raw_chunks.sort(key=lambda c: c.score, reverse=True)

        # 3. Sort by reading order (file path, then start line) for merge pass.
        raw_chunks.sort(key=lambda c: (c.file_path, c.start_line))

        # 4. Merge overlapping chunks per file.
        merged = _merge_chunks(raw_chunks, self.overlap_threshold)

        # 5. Re-sort merged chunks by score descending so highest-ranked are
        #    admitted first when truncating.
        merged.sort(key=lambda c: c.score, reverse=True)

        # 6. Admit chunks within token budget.
        admitted: list[_Chunk] = []
        budget = self.max_tokens
        truncated = False

        for chunk in merged:
            formatted = _format_chunk(chunk)
            tokens = _approximate_tokens(formatted)
            if tokens > budget:
                truncated = True
                continue
            admitted.append(chunk)
            budget -= tokens

        if not admitted:
            return AssembledContext(
                text="", total_tokens=0, chunk_count=0, truncated=truncated
            )

        # 7. Final sort of admitted chunks into reading order.
        admitted.sort(key=lambda c: (c.file_path, c.start_line))

        # 8. Format and join.
        sections = [_format_chunk(c) for c in admitted]
        text = "\n\n".join(sections)

        return AssembledContext(
            text=text,
            total_tokens=_approximate_tokens(text),
            chunk_count=len(admitted),
            truncated=truncated,
        )

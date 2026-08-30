"""Line-level citation extraction.

Parses LLM response text for citation markers [file:start_line-end_line],
validates each citation against the retrieved context, and returns
structured Citation objects.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Regex pattern for citation markers: [file_path:start_line-end_line] or [file_path:line]
# Examples: [src/auth.py:10-25], [src/models/user.py:42]
_CITATION_PATTERN = re.compile(r"\[([^:\]\s]+):(\d+)(?:-(\d+))?\]")

# Pre-compiled regex for parsing markdown context headers
_MARKDOWN_HEADER_PATTERN = re.compile(
    r"##\s+([^\s(]+)\s+\(lines\s+(\d+|\?)-(\d+|\?)\)\n```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```"
)


@dataclass(frozen=True)
class Citation:
    """Represents a single line-level citation extracted from an LLM response.

    Attributes:
        file_path: The file path referenced in the citation.
        start_line: Starting line number.
        end_line: Ending line number (same as start_line if single line).
        snippet: The code snippet corresponding to the cited line range if valid.
        valid: True if the file path and line range exist in the retrieved context.
    """

    file_path: str
    start_line: int
    end_line: int
    snippet: str = ""
    valid: bool = True


@dataclass(frozen=True)
class CitationExtractionResult:
    """The result of parsing and validating citations in an LLM response.

    Attributes:
        citations: All extracted citations in order of appearance.
        valid_citations: Subset of citations that were successfully validated against context.
        invalid_citations: Subset of citations flagged as invalid/hallucinated.
        citation_coverage: Ratio of valid citations to total citations (1.0 if no citations).
    """

    citations: list[Citation] = field(default_factory=list)
    valid_citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)
    citation_coverage: float = 1.0


@dataclass
class _ContextBlock:
    file_path: str
    start_line: int
    end_line: int
    code: str
    _cached_lines: list[str] = field(default_factory=list, repr=False)

    @property
    def lines(self) -> list[str]:
        """Lazy split lines cached on context block."""
        if not self._cached_lines and self.code:
            self._cached_lines = self.code.splitlines()
        return self._cached_lines


def _parse_raw_citations(text: str) -> list[tuple[str, int, int]]:
    """Parse raw citation tuples (file_path, start_line, end_line) from text."""
    if not text:
        return []
    matches = _CITATION_PATTERN.findall(text)
    raw_citations: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()

    for match in matches:
        file_path = match[0].strip()
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        s_min, s_max = min(start, end), max(start, end)
        item = (file_path, s_min, s_max)
        if item not in seen:
            seen.add(item)
            raw_citations.append(item)

    return raw_citations


def _extract_context_blocks(
    context: Sequence[RetrievalResult] | str | Sequence[dict[str, Any]] | None,
) -> dict[str, list[_ContextBlock]]:
    """Normalize and index context blocks by file_path for O(1) lookup."""
    if not context:
        return {}

    indexed_blocks: dict[str, list[_ContextBlock]] = {}

    def _add_block(fp: str, start: int, end: int, code: str) -> None:
        if not fp:
            return
        block = _ContextBlock(file_path=fp, start_line=start, end_line=end, code=code)
        indexed_blocks.setdefault(fp, []).append(block)

    if isinstance(context, str):
        for match in _MARKDOWN_HEADER_PATTERN.finditer(context):
            fp = match.group(1).strip()
            start_str = match.group(2)
            end_str = match.group(3)
            code = match.group(4)

            start = int(start_str) if start_str.isdigit() else 1
            end = (
                int(end_str)
                if end_str.isdigit()
                else start + len(code.splitlines()) - 1
            )
            _add_block(fp, start, end, code)

    elif isinstance(context, Sequence):
        for item in context:
            if isinstance(item, RetrievalResult):
                if (
                    item.file_path
                    and item.start_line is not None
                    and item.end_line is not None
                ):
                    _add_block(
                        item.file_path,
                        item.start_line,
                        item.end_line,
                        item.chunk_text or "",
                    )
            elif isinstance(item, dict):
                fp = str(item.get("file_path", ""))
                sl = item.get("start_line")
                el = item.get("end_line")
                code = str(item.get("chunk_text", item.get("code", "")))
                if fp and sl is not None and el is not None:
                    _add_block(fp, int(sl), int(el), code)

    return indexed_blocks


def _extract_snippet(block: _ContextBlock, cite_start: int, cite_end: int) -> str:
    """Extract code lines from block matching the citation range efficiently."""
    lines = block.lines
    if not lines:
        return ""

    rel_start = max(0, cite_start - block.start_line)
    rel_end = min(len(lines), cite_end - block.start_line + 1)

    if rel_start < len(lines) and rel_start < rel_end:
        return "\n".join(lines[rel_start:rel_end])
    return block.code[:200]


def _match_file_path(cite_fp: str, context_fp: str) -> bool:
    """Flexible matching between citation file path and context file path."""
    return (
        cite_fp == context_fp
        or context_fp.endswith(f"/{cite_fp}")
        or cite_fp.endswith(f"/{context_fp}")
    )


class CitationExtractor:
    """Extracts and validates line-level citations from LLM generated response text."""

    def parse_citations(self, text: str) -> list[tuple[str, int, int]]:
        """Parse raw citation markers from text.

        Returns:
            List of (file_path, start_line, end_line) tuples.
        """
        return _parse_raw_citations(text)

    def extract_and_validate(
        self,
        response_text: str,
        context: (
            Sequence[RetrievalResult] | str | Sequence[dict[str, Any]] | None
        ) = None,
    ) -> CitationExtractionResult:
        """Extract citations from response text and validate them against context efficiently.

        Args:
            response_text: The raw LLM output text.
            context: Retrieved context chunks (RetrievalResult sequence, formatted string, or dicts).

        Returns:
            A :class:`CitationExtractionResult` containing parsed citations and validation metadata.
        """
        raw_citations = _parse_raw_citations(response_text)
        if not raw_citations:
            return CitationExtractionResult(
                citations=[],
                valid_citations=[],
                invalid_citations=[],
                citation_coverage=1.0,
            )

        indexed_context = _extract_context_blocks(context)

        parsed_citations: list[Citation] = []
        valid_citations: list[Citation] = []
        invalid_citations: list[Citation] = []

        for fp, c_start, c_end in raw_citations:
            matching_block: _ContextBlock | None = None

            # Efficient lookup: check direct match first, fallback to linear path match
            candidate_blocks: list[_ContextBlock] = []
            if fp in indexed_context:
                candidate_blocks = indexed_context[fp]
            else:
                for ctx_fp, blocks in indexed_context.items():
                    if _match_file_path(fp, ctx_fp):
                        candidate_blocks.extend(blocks)

            # Check overlap condition: max(start) <= min(end)
            for block in candidate_blocks:
                if max(c_start, block.start_line) <= min(c_end, block.end_line):
                    matching_block = block
                    break

            if matching_block:
                snippet = _extract_snippet(matching_block, c_start, c_end)
                citation = Citation(
                    file_path=matching_block.file_path,
                    start_line=c_start,
                    end_line=c_end,
                    snippet=snippet,
                    valid=True,
                )
                valid_citations.append(citation)
            else:
                citation = Citation(
                    file_path=fp,
                    start_line=c_start,
                    end_line=c_end,
                    snippet="",
                    valid=False,
                )
                invalid_citations.append(citation)

            parsed_citations.append(citation)

        coverage = (
            len(valid_citations) / len(parsed_citations) if parsed_citations else 1.0
        )

        return CitationExtractionResult(
            citations=parsed_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            citation_coverage=coverage,
        )

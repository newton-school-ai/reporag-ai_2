"""Context assembler.

Transforms raw retrieval results into a structured, deduplicated context
block for the LLM prompt. Orders chunks by file and line, merges overlaps,
and truncates to fit the context window.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict

from reporag.ingestion.chunker import count_tokens
from reporag.retrieval.vector_search import RetrievalResult


class ContextAssembler:
    """Assembles a list of retrieval results into a formatted prompt context."""

    def __init__(self, max_tokens: int = 4000) -> None:
        """Initialize the context assembler.

        Args:
            max_tokens: Maximum allowed tokens for the fully assembled context string.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}")
        self.max_tokens = max_tokens

    @staticmethod
    def _get_language_from_path(file_path: str) -> str:
        """Get the markdown language identifier from the file extension."""
        if not file_path:
            return ""
        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            return ""

        mapping = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".xml": "xml",
            ".dockerfile": "dockerfile",
            ".txt": "text",
        }
        return mapping.get(ext, ext.lstrip("."))

    def assemble(self, results: list[RetrievalResult]) -> str:
        """Assemble retrieval results into a structured, token-limited string.

        Algorithm:
        1. Selects highest-scored chunks that fit within `max_tokens`.
        2. Groups selected chunks by file path.
        3. Sorts chunks in each file by line number.
        4. Merges chunks with >50% overlap or full containment.
        5. Formats the chunks into line-numbered markdown code blocks.
        6. Enforces post-merge token budget limit defensively.
        """
        if not results:
            return ""

        # 1. Select chunks by score within token budget
        selected: list[RetrievalResult] = []
        current_tokens = 0

        # Ensure results are sorted by score (descending)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        for result in sorted_results:
            # Estimate tokens of the final rendered string including markdown overhead
            start = result.start_line if result.start_line is not None else "?"
            end = result.end_line if result.end_line is not None else "?"
            lang = self._get_language_from_path(result.file_path)
            rendered_chunk = f"## {result.file_path} (lines {start}-{end})\n```{lang}\n{result.chunk_text.strip()}\n```\n\n"

            tokens = count_tokens(rendered_chunk)
            if current_tokens + tokens > self.max_tokens:
                continue
            selected.append(result)
            current_tokens += tokens

        if not selected:
            return ""

        # 2. Group by file_path
        grouped: dict[str, list[RetrievalResult]] = defaultdict(list)
        for r in selected:
            grouped[r.file_path].append(r)

        # 3. Sort, merge, and format for each file
        assembled_parts: list[str] = []
        total_assembled_tokens = 0

        for file_path, chunks in sorted(grouped.items()):
            # Sort by start_line ascending, then end_line descending. Treat None as -1.
            sorted_chunks = sorted(
                chunks,
                key=lambda c: (
                    c.start_line if c.start_line is not None else -1,
                    -(c.end_line if c.end_line is not None else -1),
                ),
            )

            merged_chunks = self._merge_overlapping(sorted_chunks)

            # 4. Format each merged chunk and enforce post-merge budget check
            for mc in merged_chunks:
                start = mc.start_line if mc.start_line is not None else "?"
                end = mc.end_line if mc.end_line is not None else "?"
                lang = self._get_language_from_path(file_path)

                part = f"## {file_path} (lines {start}-{end})\n```{lang}\n{mc.chunk_text.strip()}\n```"
                part_tokens = count_tokens(part + "\n\n")

                if total_assembled_tokens + part_tokens > self.max_tokens:
                    break

                assembled_parts.append(part)
                total_assembled_tokens += part_tokens

        return "\n\n".join(assembled_parts)

    def _merge_overlapping(
        self, chunks: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Merge overlapping chunks within the same file to prevent code duplication."""
        if not chunks:
            return []

        merged: list[RetrievalResult] = []
        # Use copy to avoid mutating the original RetrievalResult objects
        prev = copy.copy(chunks[0])

        for current_orig in chunks[1:]:
            current = copy.copy(current_orig)

            # If line numbers are missing (e.g. docs), we cannot overlap them safely
            if (
                prev.start_line is None
                or current.start_line is None
                or prev.end_line is None
                or current.end_line is None
            ):
                merged.append(prev)
                prev = current
                continue

            # Check if there is an overlap
            if current.start_line <= prev.end_line:
                overlap = prev.end_line - current.start_line + 1
                len_prev = prev.end_line - prev.start_line + 1
                len_curr = current.end_line - current.start_line + 1
                min_len = min(len_prev, len_curr)

                # Division by zero guard for malformed 0-line spans
                if min_len <= 0:
                    merged.append(prev)
                    prev = current
                    continue

                # Merge if overlap is > 50% of smaller chunk, or if `prev` entirely engulfs `current`
                if overlap / min_len > 0.5 or current.end_line <= prev.end_line:
                    new_lines_count = current.end_line - prev.end_line

                    if new_lines_count > 0:
                        # Grab exact new lines from end of `current` to bypass chunker headers
                        curr_lines = current.chunk_text.split("\n")
                        # Handle potential trailing newline splits
                        if curr_lines and curr_lines[-1] == "":
                            curr_lines = curr_lines[:-1]

                        new_text = "\n".join(curr_lines[-new_lines_count:])

                        if not prev.chunk_text.endswith("\n"):
                            prev.chunk_text += "\n"
                        prev.chunk_text += new_text
                        prev.end_line = current.end_line
                else:
                    # Overlap too small to merge, keep separate
                    merged.append(prev)
                    prev = current
            else:
                # No overlap
                merged.append(prev)
                prev = current

        merged.append(prev)
        return merged

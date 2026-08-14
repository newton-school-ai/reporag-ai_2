"""Context assembler.

Transforms a ranked list of :class:`~reporag.retrieval.vector_search.RetrievalResult`
objects -- the output of the RRF + reranker pipeline -- into a single, structured,
token-bounded string that the LLM can reason over.

Why greedy selection and not knapsack?
--------------------------------------
A true 0/1 knapsack would give optimal token packing but is O(n*W).  For
realistic inputs (n <= 50, W <= 8192) greedy-by-rank is indistinguishable in
quality and runs in O(n).  Because the input is already ranked by the reranker,
greedy naturally prioritises the most relevant chunks without extra bookkeeping.

Why merge overlapping chunks?
------------------------------
Both vector search and BM25 can retrieve overlapping windows of the same
function.  Showing duplicate code twice wastes tokens and confuses the LLM
(it may treat the two copies as different).  A scan-line interval merge with a
50% threshold eliminates redundancy while keeping truly distinct nearby chunks
separate (e.g., a class header and its docstring versus a method 200 lines
later).

Why ``frozenset`` for ``files_covered``?
-----------------------------------------
:class:`AssembledContext` is a frozen dataclass.  A ``frozenset`` is the only
hashable, immutable set type in Python, which makes the whole result safely
cacheable and usable as a dict key.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from reporag.ingestion.chunker import count_tokens
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssembledContext:
    """The fully formatted, token-bounded context block ready for the LLM.

    All fields are read-only after construction so callers can safely cache
    or hash the result.

    Attributes:
        text: The complete formatted context string (may be empty when no
            results were provided).
        token_count: Actual BPE token count of ``text``.  Computed from the
            same :func:`~reporag.ingestion.chunker.count_tokens` call used
            during budget selection, so the number is consistent end-to-end.
        included: Number of input chunks that were selected for the context.
        dropped: Number of input chunks excluded because they would have
            exceeded ``max_tokens``.
        files_covered: Unique file paths whose code appears in ``text``.
            Stored as a ``frozenset`` to make the dataclass trivially hashable.
    """

    text: str
    token_count: int
    included: int
    dropped: int
    files_covered: frozenset[str]


# ---------------------------------------------------------------------------
# Internal type
# ---------------------------------------------------------------------------


@dataclass
class _MergedChunk:
    """A post-merge code interval from a single file, ready to be formatted.

    This is an internal working type; callers should not depend on it.

    Attributes:
        file_path: Source file path.
        start_line: First line of the interval (1-based, or 0 when unknown).
        end_line: Last line of the interval (inclusive, or 0 when unknown).
        text: Raw source text of the interval.
        score: Best retrieval score of the merged originals.
        language: Programming language hint for the fenced code block.
    """

    file_path: str
    start_line: int
    end_line: int
    text: str
    score: float
    language: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Converts ranked retrieval results into a structured, token-bounded context block.

    The assembler runs five deterministic steps:

    1. **Priority selection** -- walk the already-ranked ``results`` list and
       greedily add each chunk while the running token total stays within
       ``max_tokens``.  The top-ranked chunk is always included even if it
       alone exceeds the budget (1-chunk minimum guarantee).

    2. **Group and sort** -- group selected chunks by ``file_path``; sort each
       group by ``start_line`` ascending; sort files alphabetically so two
       runs of the same call produce identical output.

    3. **Merge overlapping chunks** -- for each file group, apply a scan-line
       interval merge.  Two chunks are merged when their overlap exceeds 50%
       of the shorter chunk's length, eliminating redundant code that both
       vector and BM25 search often retrieve.

    4. **Format** -- render each merged chunk as a fenced code block with
       actual source line numbers.

    5. **Join** -- blocks within a file are separated by a blank line; files
       are separated by ``\\n---\\n``.

    Args:
        max_tokens: Hard token budget for the assembled context.  Must be > 0.
        encoding: tiktoken encoding name.  Passed through only for documentation
            purposes; the actual counting is delegated to
            :func:`~reporag.ingestion.chunker.count_tokens` which uses
            ``cl100k_base`` internally.

    Raises:
        ValueError: If ``max_tokens <= 0``.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        encoding: str = "cl100k_base",
    ) -> None:
        """Initialise the assembler, validating the token budget."""
        if max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be > 0, got {max_tokens!r}.  "
                "A non-positive budget would produce an empty context on every call."
            )
        self.max_tokens = max_tokens
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(self, results: list[RetrievalResult]) -> AssembledContext:
        """Assemble ranked retrieval results into a formatted, token-bounded context.

        Args:
            results: Already-ranked :class:`~reporag.retrieval.vector_search.RetrievalResult`
                objects (index 0 = best score).  The list is consumed read-only;
                the caller's objects are never mutated.

        Returns:
            An :class:`AssembledContext` with the formatted text and metadata.
            Returns an empty context (all counters zero, ``text=""``) when
            *results* is empty or every chunk has empty ``chunk_text``.
        """
        if not results:
            return AssembledContext(
                text="",
                token_count=0,
                included=0,
                dropped=0,
                files_covered=frozenset(),
            )

        # Step 1 -- greedy priority selection
        selected, dropped = self._select_by_priority(results)

        if not selected:
            return AssembledContext(
                text="",
                token_count=0,
                included=0,
                dropped=dropped,
                files_covered=frozenset(),
            )

        included = len(selected)

        # Step 2 -- group by file and sort
        groups: dict[str, list[RetrievalResult]] = defaultdict(list)
        for r in selected:
            groups[r.file_path].append(r)
        for chunks in groups.values():
            chunks.sort(key=lambda r: (r.start_line or 0))
        sorted_files = sorted(groups.keys())

        # Steps 3 & 4 -- merge overlaps, then format each chunk
        file_blocks: list[str] = []
        for file_path in sorted_files:
            merged = self._merge_overlapping(groups[file_path])
            blocks = [self._format_block(mc) for mc in merged]
            file_blocks.append("\n\n".join(blocks))

        text = "\n---\n".join(file_blocks)
        token_count = count_tokens(text)
        files_covered = frozenset(groups.keys())

        logger.debug(
            "ContextAssembler: included=%d dropped=%d files=%d tokens=%d",
            included,
            dropped,
            len(files_covered),
            token_count,
        )

        return AssembledContext(
            text=text,
            token_count=token_count,
            included=included,
            dropped=dropped,
            files_covered=files_covered,
        )

    # ------------------------------------------------------------------
    # Step 1: priority selection
    # ------------------------------------------------------------------

    def _select_by_priority(
        self,
        results: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], int]:
        """Walk results in rank order, adding chunks greedily until budget is full.

        The 1-chunk minimum guarantee: if the very first non-empty chunk
        exceeds the budget, it is included anyway.  This prevents the
        assembler from ever returning empty context when the caller provided
        at least one result -- an empty context is worse than an over-budget
        one because the LLM would have nothing to reason from.

        Args:
            results: Ranked list (index 0 = best).

        Returns:
            A ``(selected, dropped)`` pair where *selected* holds the chosen
            :class:`~reporag.retrieval.vector_search.RetrievalResult` objects in
            their original rank order and *dropped* is the count of excluded chunks.
        """
        selected: list[RetrievalResult] = []
        dropped = 0
        budget_remaining = self.max_tokens
        first_non_empty_included = False

        for result in results:
            if not result.chunk_text:
                # Empty text -- skip silently; don't count as dropped.
                continue

            tokens = count_tokens(result.chunk_text)

            if tokens <= budget_remaining:
                selected.append(result)
                budget_remaining -= tokens
                first_non_empty_included = True
            elif not first_non_empty_included:
                # 1-chunk minimum guarantee: always include the first result
                # even if it alone blows the budget.
                selected.append(result)
                budget_remaining -= tokens  # goes negative; that is intentional
                first_non_empty_included = True
                logger.debug(
                    "1-chunk minimum: including over-budget chunk from %s (%d tokens > %d budget)",
                    result.file_path,
                    tokens,
                    self.max_tokens,
                )
            else:
                dropped += 1

        return selected, dropped

    # ------------------------------------------------------------------
    # Step 3: scan-line overlap merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_overlapping(
        chunks: list[RetrievalResult],
        threshold: float = 0.5,
    ) -> list[_MergedChunk]:
        """Merge consecutive chunks whose line-range overlap exceeds *threshold*.

        Uses a scan-line approach: walk the list (already sorted by start_line)
        maintaining a running "current" interval.  When the next chunk overlaps
        the current one by more than 50% of the shorter chunk's length, extend
        the interval.  Otherwise flush the current interval and start a new one.

        Why 50%?  A lower threshold (e.g., 1 shared line) would merge chunks
        that are merely adjacent.  A higher threshold (e.g., 80%) would leave
        too many near-duplicates.  50% is the midpoint and works well for the
        typical retrieval scenario where vector + BM25 grab overlapping
        function bodies.

        Args:
            chunks: Sorted (ascending ``start_line``) list of selected results
                from a single file.
            threshold: Merge when ``overlap_lines / min(len_a, len_b) > threshold``.

        Returns:
            A list of :class:`_MergedChunk` objects ready for formatting.
        """
        if not chunks:
            return []

        def _to_merged(r: RetrievalResult) -> _MergedChunk:
            """Convert a RetrievalResult to a _MergedChunk."""
            start = r.start_line or 0
            end = r.end_line or 0
            lang = r.metadata.get("language", "python") if r.metadata else "python"
            return _MergedChunk(
                file_path=r.file_path,
                start_line=start,
                end_line=end,
                text=r.chunk_text,
                score=r.score,
                language=str(lang),
            )

        current = _to_merged(chunks[0])
        merged: list[_MergedChunk] = []

        for r in chunks[1:]:
            candidate = _to_merged(r)

            # Compute overlap between current interval and candidate
            c_start, c_end = current.start_line, current.end_line
            k_start, k_end = candidate.start_line, candidate.end_line

            overlap_lines = min(c_end, k_end) - max(c_start, k_start) + 1

            if overlap_lines > 0:
                len_current = max(c_end - c_start + 1, 1)
                len_candidate = max(k_end - k_start + 1, 1)
                overlap_ratio = overlap_lines / min(len_current, len_candidate)
            else:
                overlap_ratio = 0.0

            if overlap_ratio > threshold:
                # Merge: union of ranges, concatenate text, keep best score
                current = _MergedChunk(
                    file_path=current.file_path,
                    start_line=min(c_start, k_start),
                    end_line=max(c_end, k_end),
                    text=current.text + "\n" + candidate.text,
                    score=max(current.score, candidate.score),
                    language=current.language,
                )
            else:
                merged.append(current)
                current = candidate

        merged.append(current)
        return merged

    # ------------------------------------------------------------------
    # Step 4: format a single merged chunk
    # ------------------------------------------------------------------

    @staticmethod
    def _format_block(chunk: _MergedChunk) -> str:
        """Render a merged chunk as a fenced code block with line numbers.

        The header is ``## {file_path} (lines {start}-{end})``.  Each source
        line is prefixed with its actual line number right-aligned to a width
        determined by the largest line number in this block, e.g.::

            ## src/auth.py (lines 10-15)
            ```python
             10 | def auth():
             11 |     pass
            ```

        When ``start_line`` is 0 (meaning the original result had no line
        information), the header and body omit line numbers entirely -- showing
        unknown line numbers as ``0`` would be worse than showing nothing.

        Args:
            chunk: The merged chunk to format.

        Returns:
            A multi-line string with the header and fenced code block.
        """
        no_line_info = chunk.start_line == 0 and chunk.end_line == 0

        if no_line_info:
            header = f"## {chunk.file_path}"
            body = chunk.text
        else:
            header = f"## {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
            # Right-align line numbers to the width of the largest number
            lines = chunk.text.splitlines()
            num_lines = len(lines)
            last_line_num = chunk.start_line + num_lines - 1
            width = len(str(last_line_num))

            numbered_lines = []
            for i, line in enumerate(lines):
                lineno = chunk.start_line + i
                numbered_lines.append(f"{lineno:>{width}} | {line}")
            body = "\n".join(numbered_lines)

        return f"{header}\n```{chunk.language}\n{body}\n```"

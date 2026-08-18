"""Context assembler.

Transforms raw retrieval results into a structured, deduplicated context
block for the LLM prompt. Orders chunks by file and line, merges overlaps,
and truncates to fit the context window.

Pipeline position
------------------
This module sits directly downstream of Issue 19
(:func:`~reporag.retrieval.fusion.reciprocal_rank_fusion` and
:class:`~reporag.retrieval.reranker.CrossEncoderReranker`): its input is a
``list[RetrievalResult]`` that is already **priority-ordered**, i.e.
``results[0]`` is the single most relevant chunk. Both upstream producers use
"higher score is better" (RRF sums and cross-encoder logits alike), so this
module treats ``RetrievalResult.score`` as the truncation priority directly
-- no re-ranking happens here.

Why merge by line, not by chunk identity
-----------------------------------------
:func:`reciprocal_rank_fusion` already deduplicates results that share the
*exact* ``(file_path, start_line, end_line)`` key. But a vector-search hit
covering lines 10-30 of a file and a BM25 hit covering lines 20-25 of the
*same* file are different keys yet clearly overlapping code the LLM
shouldn't see twice. Because each :class:`RetrievalResult` carries
line-aligned ``chunk_text`` (one source line per line of the range), overlaps
can be resolved *exactly* -- by building a per-line ownership map within each
group of overlapping chunks -- rather than by fuzzy text-diffing. Ties for a
given line go to whichever contributing chunk has the higher score, matching
the same "higher score wins" priority used for truncation.

Usage::

    from reporag.generation.context_assembler import ContextAssembler

    assembler = ContextAssembler(max_tokens=4000)
    context = assembler.assemble(reranked_results)
    print(context.text)
    print(context.total_tokens, context.truncated)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from reporag.config import settings
from reporag.ingestion.chunker import count_tokens
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Default token budget for the assembled context block. Deliberately well
# below typical 8k/16k model context windows, since the system prompt, the
# query, few-shot examples, and generation headroom (Issue 24's
# prompt_builder) all share the same window. ~8 chunks at the chunker's
# default 512-token budget (see SemanticChunker) comfortably fit here.
_DEFAULT_MAX_TOKENS = 4000

# Marker inserted for a line number that falls inside a merged chunk's range
# but wasn't actually covered by any contributing RetrievalResult's
# chunk_text (defensive -- only reachable if a retriever hands back a
# start/end range that undercounts its own text's line count).
_GAP_MARKER = "..."


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AssembledChunk:
    """A single deduplicated, line-merged block of code in the final context.

    Attributes:
        file_path:     Path to the source file this block was drawn from.
        start_line:    1-based start line, or ``None`` when the contributing
                       result carried no line information (e.g. a
                       symbol-level graph-traversal hit).
        end_line:      1-based inclusive end line, or ``None`` (paired with
                       ``start_line is None``).
        text:          The merged source text, one line per source line, with
                       no line-number prefixes and no file header -- callers
                       needing the rendered block should use the ``text``
                       field of the owning :class:`AssembledContext` instead.
        source_score:  The highest ``score`` among the original
                       :class:`~reporag.retrieval.vector_search.RetrievalResult`
                       objects merged into this block. Drives truncation
                       priority.
        merged_from:   Number of original results merged into this block.
                       ``1`` when no merging occurred.
    """

    file_path: str
    start_line: int | None
    end_line: int | None
    text: str
    source_score: float
    merged_from: int = 1


@dataclass
class AssembledContext:
    """The final, prompt-ready context block plus bookkeeping metadata.

    Attributes:
        text:            The fully formatted context block (file headers +
                         line-numbered fenced code, one section per chunk,
                         separated by blank lines) ready to be embedded in an
                         LLM prompt.
        chunks:          The :class:`AssembledChunk` objects included in
                         *text*, ordered by ``(file_path, start_line)`` --
                         the same order they appear in *text*. Used by
                         Issue 25's citation validator to check whether a
                         cited ``[file:start-end]`` range actually appears in
                         the context that was shown to the model.
        total_tokens:    Token count of *text* (via
                         :func:`~reporag.ingestion.chunker.count_tokens`, the
                         same counter used for chunk budgets, so counts are
                         comparable across the pipeline).
        truncated:       ``True`` if one or more candidate chunks were left
                         out to stay within the token budget.
        dropped_chunks:  Number of merged candidate chunks that did not make
                         the cut. ``0`` when ``truncated`` is ``False``.
    """

    text: str
    chunks: list[AssembledChunk] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False
    dropped_chunks: int = 0


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Orders, deduplicates, formats, and truncates retrieval results.

    Args:
        max_tokens: Default token budget for :meth:`assemble` when the
            caller doesn't override it per call. Defaults to
            :data:`_DEFAULT_MAX_TOKENS` (4000).

    Raises:
        ValueError: If *max_tokens* is not positive.
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        """Initialise the assembler with a default token budget."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}.")
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        results: list[RetrievalResult],
        *,
        max_tokens: int | None = None,
    ) -> AssembledContext:
        """Assemble *results* into a single formatted, budget-fit context block.

        Pipeline:

        1. Group by file and merge line-overlapping results (see module
           docstring) into :class:`AssembledChunk` candidates.
        2. Select candidates by descending ``source_score`` (highest-ranked
           first) until adding the next one would exceed *max_tokens*.
           Chunks are never partially included -- a chunk that doesn't fit
           is dropped whole rather than truncated mid-block, so every
           included chunk's line range stays citation-valid.
        3. Re-order the *selected* chunks by ``(file_path, start_line)`` for
           final display, and render each as a Markdown section with a file
           header and line-numbered fenced code.

        Args:
            results: Priority-ordered retrieval results (highest ``score``
                first), typically straight from
                :meth:`~reporag.retrieval.reranker.CrossEncoderReranker.rerank`
                or :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion`.
                Selection is driven entirely by ``score``, not list order,
                but ties still fall back to ``(file_path, start_line)`` for
                determinism.
            max_tokens: Override this call's token budget. ``None`` (default)
                uses ``self.max_tokens``. Must be ``>= 1`` when given.

        Returns:
            An :class:`AssembledContext` with the rendered text and
            bookkeeping. An empty *results* list returns an empty, non
            truncated context.

        Raises:
            ValueError: If *max_tokens* (when given) is ``< 1``.
        """
        budget = self.max_tokens if max_tokens is None else max_tokens
        if budget < 1:
            raise ValueError(f"max_tokens must be >= 1, got {budget!r}.")

        if not results:
            return AssembledContext(text="")

        candidates = self._merge_overlaps(results)

        selected, dropped = self._select_within_budget(candidates, budget)

        # Final display order is independent of selection order: selection
        # is priority-first, display is file/line-first so the LLM reads
        # each file top-to-bottom rather than in relevance-shuffled order.
        ordered = sorted(selected, key=_display_sort_key)

        text = self._render(ordered)
        if dropped:
            logger.debug(
                "Context assembly dropped %d/%d candidate chunk(s) to stay "
                "within a %d-token budget (%d tokens used).",
                dropped,
                len(candidates),
                budget,
                count_tokens(text),
            )
        return AssembledContext(
            text=text,
            chunks=ordered,
            total_tokens=count_tokens(text),
            truncated=dropped > 0,
            dropped_chunks=dropped,
        )

    # ------------------------------------------------------------------
    # Step 1: overlap merging
    # ------------------------------------------------------------------

    def _merge_overlaps(self, results: list[RetrievalResult]) -> list[AssembledChunk]:
        """Group *results* by file and merge line-overlapping ranges.

        Results with ``start_line is None`` or ``end_line is None`` (no line
        anchor -- e.g. a symbol-level hit with no concrete range) can't be
        interval-merged, so each becomes its own standalone
        :class:`AssembledChunk` unconditionally.
        """
        by_file: dict[str, list[RetrievalResult]] = {}
        unanchored: list[AssembledChunk] = []

        for r in results:
            if r.start_line is None or r.end_line is None:
                unanchored.append(
                    AssembledChunk(
                        file_path=r.file_path,
                        start_line=None,
                        end_line=None,
                        text=r.chunk_text,
                        source_score=r.score,
                        merged_from=1,
                    )
                )
                continue
            by_file.setdefault(r.file_path, []).append(r)

        merged: list[AssembledChunk] = list(unanchored)
        for file_path, file_results in by_file.items():
            merged.extend(self._merge_file(file_path, file_results))
        return merged

    def _merge_file(
        self, file_path: str, results: list[RetrievalResult]
    ) -> list[AssembledChunk]:
        """Merge overlapping line ranges within a single file's results."""
        # Group contiguous/overlapping intervals first (requires line order).
        by_line = sorted(results, key=lambda r: (r.start_line, r.end_line))

        groups: list[list[RetrievalResult]] = []
        current_group = [by_line[0]]
        current_end = by_line[0].end_line
        for r in by_line[1:]:
            if r.start_line <= current_end:
                current_group.append(r)
                current_end = max(current_end, r.end_line)
            else:
                groups.append(current_group)
                current_group = [r]
                current_end = r.end_line
        groups.append(current_group)

        return [self._merge_group(file_path, group) for group in groups]

    @staticmethod
    def _merge_group(file_path: str, group: list[RetrievalResult]) -> AssembledChunk:
        """Merge one group of overlapping same-file results into one chunk.

        A single-member group is a pure pass-through (no merge cost). For
        multi-member groups, a per-line ownership map is built by walking
        contributors in descending-score order so the highest-scoring
        contributor's text wins on any line more than one chunk covers --
        the same "higher score wins" priority used for truncation.
        """
        if len(group) == 1:
            r = group[0]
            return AssembledChunk(
                file_path=file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                text=r.chunk_text,
                source_score=r.score,
                merged_from=1,
            )

        start_line = min(r.start_line for r in group)
        end_line = max(r.end_line for r in group)

        line_text: dict[int, str] = {}
        for r in sorted(group, key=lambda r: -r.score):
            lines = r.chunk_text.split("\n")
            expected_len = r.end_line - r.start_line + 1
            for offset, line in enumerate(lines[:expected_len]):
                line_no = r.start_line + offset
                line_text.setdefault(line_no, line)

        merged_text = "\n".join(
            line_text.get(line_no, _GAP_MARKER)
            for line_no in range(start_line, end_line + 1)
        )

        return AssembledChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            text=merged_text,
            source_score=max(r.score for r in group),
            merged_from=len(group),
        )

    # ------------------------------------------------------------------
    # Step 2: budget-aware selection
    # ------------------------------------------------------------------

    def _select_within_budget(
        self, candidates: list[AssembledChunk], budget: int
    ) -> tuple[list[AssembledChunk], int]:
        """Greedily select highest-``source_score`` chunks up to *budget*.

        Each candidate's cost is measured as the token count of its *fully
        rendered* section (file header + fenced, line-numbered code) rather
        than raw chunk text, so the acceptance criterion "total tokens
        within max_tokens" holds for the actual formatted output, not an
        under-count of it. A chunk is included only if it fits whole --
        chunks are never truncated mid-block, since that would produce a
        code block with an inaccurate line range.
        """
        priority_order = sorted(candidates, key=_priority_sort_key)

        selected: list[AssembledChunk] = []
        used_tokens = 0
        for chunk in priority_order:
            cost = count_tokens(self._render([chunk]))
            if used_tokens + cost > budget:
                continue
            selected.append(chunk)
            used_tokens += cost

        dropped = len(candidates) - len(selected)
        return selected, dropped

    # ------------------------------------------------------------------
    # Step 3: rendering
    # ------------------------------------------------------------------

    def _render(self, chunks: list[AssembledChunk]) -> str:
        """Render *chunks*, in the given order, as Markdown sections."""
        return "\n\n".join(self._render_chunk(c) for c in chunks)

    @staticmethod
    def _render_chunk(chunk: AssembledChunk) -> str:
        """Render one chunk as ``## file (lines N-M)`` + a fenced code block.

        Every code line is prefixed with its 1-based source line number so
        the LLM can cite exact lines back (see Issue 25's citation format
        ``[file_path:start_line-end_line]``). Chunks with no line anchor
        (``start_line is None``) get a bare header and an unnumbered fence,
        since there is no line to number against.

        Source text that itself contains a run of backticks (a Markdown
        code sample inside a docstring, a ``.md`` file, this very
        codebase's own Sphinx-style examples, ...) would otherwise close
        the fence early and spill the remainder outside the code block.
        The fence length is chosen via :func:`_fence_for` to always be one
        backtick longer than the longest backtick run already in the body,
        the same escaping strategy CommonMark itself recommends.
        """
        language = _infer_language(chunk.file_path)

        if chunk.start_line is None or chunk.end_line is None:
            header = f"## {chunk.file_path}"
            body = chunk.text
        else:
            header = f"## {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
            width = len(str(chunk.end_line))
            body = "\n".join(
                f"{chunk.start_line + i:>{width}} | {line}"
                for i, line in enumerate(chunk.text.split("\n"))
            )

        fence = _fence_for(body)
        return f"{header}\n{fence}{language}\n{body}\n{fence}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fence_for(body: str) -> str:
    """Return a backtick fence guaranteed not to be closed by *body*'s content.

    Scans for the longest run of consecutive backticks already present in
    *body* and returns a fence one backtick longer (minimum 3, the Markdown
    default). A body with no backticks gets the standard ```` ``` ```` fence;
    a body containing its own ```` ``` ```` code sample gets a 4-backtick
    fence, and so on -- the same nesting-safe convention CommonMark itself
    documents for fencing code that contains fences.
    """
    longest_run = 0
    current_run = 0
    for char in body:
        if char == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return "`" * max(3, longest_run + 1)


def _infer_language(file_path: str) -> str:
    """Infer a Markdown fence language tag from *file_path*'s extension.

    Reuses ``settings.extension_map`` (the same table
    :class:`~reporag.ingestion.chunker.SemanticChunker` uses to pick a
    tree-sitter grammar) so fence tags stay consistent with the rest of the
    pipeline. Falls back to an empty tag (a plain, unhighlighted fence) for
    unrecognised extensions rather than guessing.
    """
    for ext, language in settings.extension_map.items():
        if file_path.endswith(ext):
            return language
    return ""


def _priority_sort_key(chunk: AssembledChunk) -> tuple[float, str, int]:
    """Sort key for budget selection: highest score first, then determinism.

    ``source_score`` is negated so ``sorted()`` (ascending) yields
    highest-score-first. Ties fall back to ``(file_path, start_line)`` so
    reruns of the same input produce byte-identical selection.
    """
    return (-chunk.source_score, chunk.file_path, chunk.start_line or 0)


# Sentinel start-line used to sort unanchored chunks after every real line
# number within their file. int, not float("inf"), so the sort key stays a
# plain comparable tuple of (str, int).
_AFTER_ALL_LINES = 2**63 - 1


def _display_sort_key(chunk: AssembledChunk) -> tuple[str, int]:
    """Sort key for final display: file path, then start line.

    Unanchored chunks (``start_line is None``) sort after every anchored
    chunk in the same file, since there's no line position to place them
    relative to.
    """
    line = chunk.start_line if chunk.start_line is not None else _AFTER_ALL_LINES
    return (chunk.file_path, line)

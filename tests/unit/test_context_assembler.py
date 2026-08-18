"""Unit tests for the context assembler (Issue 23).

Covers every acceptance criterion from the issue:

* chunks ordered by file_path, then start_line,
* overlapping chunks (same file, intersecting line ranges) are merged into
  one block rather than shown twice,
* each rendered chunk is prefixed with a ``## file (lines N-M)`` header,
* the assembled text's total token count stays within ``max_tokens``,
* when truncating, the highest-``score`` candidates are kept and lower-score
  ones are dropped first, regardless of input order,

plus supporting behaviour the module documents: exact-overlap conflict
resolution (higher score wins a shared line), non-overlapping tails of a
lower-priority chunk are preserved, unanchored (no line info) results,
determinism/no-mutation, and input validation.

Fixtures mirror the conventions in ``test_fusion.py`` / ``test_reranker.py``:
a small ``_r()`` builder for minimal ``RetrievalResult`` objects, grouped
into ``Test*`` classes by acceptance criterion.
"""

from __future__ import annotations

import copy

import pytest

from reporag.generation.context_assembler import (
    AssembledChunk,
    AssembledContext,
    ContextAssembler,
)
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _r(
    file_path: str,
    start_line: int | None,
    end_line: int | None = None,
    *,
    score: float = 1.0,
    chunk_text: str | None = None,
    symbol_name: str | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult for tests.

    When *chunk_text* is omitted and both line numbers are given, a
    placeholder line is generated per line in the range (``"L{n}"``) so the
    line-numbered rendering and per-line merge logic have real,
    distinguishable content to check against.
    """
    if end_line is None:
        end_line = start_line
    if chunk_text is None:
        if start_line is not None and end_line is not None:
            chunk_text = "\n".join(f"L{n}" for n in range(start_line, end_line + 1))
        else:
            chunk_text = f"<{file_path}>"
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        chunk_text=chunk_text,
        metadata={},
    )


@pytest.fixture
def assembler() -> ContextAssembler:
    """A generously-budgeted assembler for tests that aren't about truncation."""
    return ContextAssembler(max_tokens=5000)


# ---------------------------------------------------------------------------
# Acceptance: ordering by file, then line
# ---------------------------------------------------------------------------


class TestOrdering:
    """Chunks are ordered by file_path, then start_line, in the output."""

    def test_orders_by_file_path(self, assembler: ContextAssembler) -> None:
        results = [_r("z.py", 1), _r("a.py", 1), _r("m.py", 1)]
        ctx = assembler.assemble(results)
        assert [c.file_path for c in ctx.chunks] == ["a.py", "m.py", "z.py"]

    def test_orders_by_start_line_within_a_file(
        self, assembler: ContextAssembler
    ) -> None:
        results = [_r("a.py", 50, 51), _r("a.py", 1, 2), _r("a.py", 20, 21)]
        ctx = assembler.assemble(results)
        assert [c.start_line for c in ctx.chunks] == [1, 20, 50]

    def test_display_order_independent_of_input_order(
        self, assembler: ContextAssembler
    ) -> None:
        """Selection is score-driven, but display order never depends on the
        order results were passed in -- only on (file_path, start_line)."""
        a = _r("a.py", 1, 2, score=0.1)
        b = _r("b.py", 1, 2, score=0.9)
        ctx_forward = assembler.assemble([a, b])
        ctx_reversed = assembler.assemble([b, a])
        assert ctx_forward.text == ctx_reversed.text

    def test_text_reflects_final_chunk_order(self, assembler: ContextAssembler) -> None:
        results = [_r("z.py", 1, 1), _r("a.py", 1, 1)]
        ctx = assembler.assemble(results)
        assert ctx.text.index("a.py") < ctx.text.index("z.py")


# ---------------------------------------------------------------------------
# Acceptance: overlapping chunks merged
# ---------------------------------------------------------------------------


class TestOverlapMerging:
    """Same-file, line-overlapping results collapse into one chunk."""

    def test_overlapping_ranges_merge_into_one_chunk(
        self, assembler: ContextAssembler
    ) -> None:
        results = [_r("a.py", 1, 10), _r("a.py", 5, 15)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].start_line == 1
        assert ctx.chunks[0].end_line == 15
        assert ctx.chunks[0].merged_from == 2

    def test_non_overlapping_ranges_stay_separate(
        self, assembler: ContextAssembler
    ) -> None:
        results = [_r("a.py", 1, 5), _r("a.py", 20, 25)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 2
        assert all(c.merged_from == 1 for c in ctx.chunks)

    def test_touching_but_not_overlapping_ranges_stay_separate(
        self, assembler: ContextAssembler
    ) -> None:
        """Lines 1-5 and 6-10 are adjacent but do not share a line, so they
        are not merged (the module only merges true line overlap)."""
        results = [_r("a.py", 1, 5), _r("a.py", 6, 10)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 2

    def test_fully_contained_range_merges(self, assembler: ContextAssembler) -> None:
        results = [_r("a.py", 1, 20), _r("a.py", 5, 10)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 1
        assert (ctx.chunks[0].start_line, ctx.chunks[0].end_line) == (1, 20)

    def test_overlap_only_merges_within_same_file(
        self, assembler: ContextAssembler
    ) -> None:
        """Identical line ranges in two different files never merge."""
        results = [_r("a.py", 1, 10), _r("b.py", 1, 10)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 2
        assert {c.file_path for c in ctx.chunks} == {"a.py", "b.py"}

    def test_three_way_chain_overlap_merges_into_one(
        self, assembler: ContextAssembler
    ) -> None:
        """A(1-10) overlaps B(8-15) overlaps C(14-20): all three chain-merge
        into a single block even though A and C don't directly overlap."""
        results = [_r("a.py", 1, 10), _r("a.py", 8, 15), _r("a.py", 14, 20)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 1
        assert (ctx.chunks[0].start_line, ctx.chunks[0].end_line) == (1, 20)
        assert ctx.chunks[0].merged_from == 3

    def test_higher_score_wins_a_contested_line(
        self, assembler: ContextAssembler
    ) -> None:
        """On lines both chunks cover, the higher-score chunk's text wins."""
        high = _r("a.py", 1, 5, score=0.9, chunk_text="HI1\nHI2\nHI3\nHI4\nHI5")
        low = _r("a.py", 3, 7, score=0.1, chunk_text="LO3\nLO4\nLO5\nLO6\nLO7")
        ctx = assembler.assemble([low, high])
        merged = ctx.chunks[0]
        assert (merged.start_line, merged.end_line) == (1, 7)
        # Lines 1-5 come from the higher-score chunk...
        assert merged.text.splitlines()[:5] == ["HI1", "HI2", "HI3", "HI4", "HI5"]
        # ...but the non-overlapping tail (6-7) is still filled from `low`.
        assert merged.text.splitlines()[5:] == ["LO6", "LO7"]

    def test_merge_result_score_is_max_of_contributors(
        self, assembler: ContextAssembler
    ) -> None:
        results = [_r("a.py", 1, 10, score=0.2), _r("a.py", 5, 15, score=0.8)]
        ctx = assembler.assemble(results)
        assert ctx.chunks[0].source_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Acceptance: each chunk prefixed with file + line range
# ---------------------------------------------------------------------------


class TestFormatting:
    """Rendered output has a file+line header and line-numbered code."""

    def test_header_has_file_and_line_range(self, assembler: ContextAssembler) -> None:
        ctx = assembler.assemble([_r("src/app.py", 10, 12)])
        assert "## src/app.py (lines 10-12)" in ctx.text

    def test_code_is_fenced(self, assembler: ContextAssembler) -> None:
        ctx = assembler.assemble([_r("a.py", 1, 2)])
        assert "```python" in ctx.text
        assert ctx.text.rstrip().endswith("```")

    def test_lines_are_numbered_with_source_line_numbers(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.py", 10, 12, chunk_text="one\ntwo\nthree")])
        assert "10 | one" in ctx.text
        assert "11 | two" in ctx.text
        assert "12 | three" in ctx.text

    def test_fence_language_inferred_from_extension(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.js", 1, 1)])
        assert "```javascript" in ctx.text

    def test_unrecognised_extension_gets_bare_fence(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.rs", 1, 1)])
        assert "```\n" in ctx.text

    def test_multiple_chunks_are_separated(self, assembler: ContextAssembler) -> None:
        ctx = assembler.assemble([_r("a.py", 1, 1), _r("b.py", 1, 1)])
        assert "\n\n" in ctx.text


# ---------------------------------------------------------------------------
# Acceptance: total tokens within max_tokens
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """The assembled text's token count never exceeds max_tokens."""

    def test_total_tokens_within_budget(self) -> None:
        assembler = ContextAssembler(max_tokens=40)
        results = [_r(f"f{i}.py", 1, 30, score=1.0 - i * 0.01) for i in range(5)]
        ctx = assembler.assemble(results)
        assert ctx.total_tokens <= 40

    def test_not_truncated_when_everything_fits(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.py", 1, 3)])
        assert ctx.truncated is False
        assert ctx.dropped_chunks == 0

    def test_per_call_max_tokens_overrides_default(self) -> None:
        assembler = ContextAssembler(max_tokens=5000)
        results = [_r(f"f{i}.py", 1, 30, score=1.0 - i * 0.01) for i in range(5)]
        ctx = assembler.assemble(results, max_tokens=40)
        assert ctx.total_tokens <= 40
        assert ctx.truncated is True

    def test_chunk_too_big_for_any_budget_is_dropped_not_split(self) -> None:
        """A single oversized chunk is dropped whole rather than truncated
        mid-block -- every included chunk must stay citation-valid."""
        assembler = ContextAssembler(max_tokens=5)
        ctx = assembler.assemble([_r("a.py", 1, 100)])
        assert ctx.chunks == []
        assert ctx.truncated is True
        assert ctx.dropped_chunks == 1
        assert ctx.text == ""


# ---------------------------------------------------------------------------
# Acceptance: highest-ranked prioritized when truncating
# ---------------------------------------------------------------------------


class TestTruncationPriority:
    """When chunks must be dropped, the lowest-score ones go first."""

    @staticmethod
    def _rendered_cost(assembler: ContextAssembler, result: RetrievalResult) -> int:
        """Token cost of *result* rendered alone, via the real tokenizer.

        Costs are computed through the assembler itself rather than
        hardcoded, since the active token counter (tiktoken vs. a
        length-based fallback) varies by environment and a hardcoded
        constant would be a brittle, environment-specific guess.
        """
        return assembler.assemble([result], max_tokens=10_000).total_tokens

    def test_keeps_higher_score_chunk_when_budget_forces_a_choice(self) -> None:
        assembler = ContextAssembler(max_tokens=10_000)
        high = _r("keep.py", 1, 20, score=0.99)
        low = _r("drop.py", 1, 20, score=0.01)
        # Budget fits exactly one rendered chunk but not both.
        budget = self._rendered_cost(assembler, high)

        # Pass the low-score one first to prove selection isn't input-order
        # driven.
        ctx = assembler.assemble([low, high], max_tokens=budget)
        files = {c.file_path for c in ctx.chunks}
        assert "keep.py" in files
        assert "drop.py" not in files
        assert ctx.truncated is True
        assert ctx.dropped_chunks == 1

    def test_input_order_does_not_affect_which_chunks_are_kept(self) -> None:
        assembler = ContextAssembler(max_tokens=10_000)
        high = _r("keep.py", 1, 20, score=0.99)
        low = _r("drop.py", 1, 20, score=0.01)
        budget = self._rendered_cost(assembler, high)
        forward = assembler.assemble([high, low], max_tokens=budget)
        reversed_ = assembler.assemble([low, high], max_tokens=budget)
        assert {c.file_path for c in forward.chunks} == {
            c.file_path for c in reversed_.chunks
        }

    def test_smaller_lower_priority_chunk_fills_leftover_budget(self) -> None:
        """A greedy pass can still admit a cheap, lower-score chunk once the
        expensive, higher-score one has been paid for."""
        assembler = ContextAssembler(max_tokens=10_000)
        big_high = _r("big.py", 1, 20, score=0.9)
        small_low = _r("small.py", 1, 1, score=0.1)
        # Budget fits both, back to back, with nothing to spare.
        budget = self._rendered_cost(assembler, big_high) + self._rendered_cost(
            assembler, small_low
        )
        ctx = assembler.assemble([small_low, big_high], max_tokens=budget)
        files = {c.file_path for c in ctx.chunks}
        assert "big.py" in files
        assert "small.py" in files


# ---------------------------------------------------------------------------
# Unanchored results (no line information)
# ---------------------------------------------------------------------------


class TestUnanchoredResults:
    """Results with start_line/end_line == None can't be line-merged."""

    def test_unanchored_result_passes_through_standalone(
        self, assembler: ContextAssembler
    ) -> None:
        result = _r("sym.py", None, None, chunk_text="class Foo: ...")
        ctx = assembler.assemble([result])
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].start_line is None
        assert "## sym.py" in ctx.text
        assert "(lines" not in ctx.text.split("```")[0]

    def test_unanchored_results_never_merge_with_anchored_ones(
        self, assembler: ContextAssembler
    ) -> None:
        anchored = _r("a.py", 1, 5)
        unanchored = _r("a.py", None, None)
        ctx = assembler.assemble([anchored, unanchored])
        assert len(ctx.chunks) == 2

    def test_unanchored_sorts_after_anchored_in_same_file(
        self, assembler: ContextAssembler
    ) -> None:
        anchored = _r("a.py", 1, 5)
        unanchored = _r("a.py", None, None)
        ctx = assembler.assemble([unanchored, anchored])
        assert ctx.chunks[0].start_line == 1
        assert ctx.chunks[1].start_line is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and unusual-but-real-world input shapes."""

    def test_embedded_triple_backticks_do_not_break_the_fence(
        self, assembler: ContextAssembler
    ) -> None:
        """Source text containing its own ``` sample (a docstring with a
        fenced example, a markdown file, ...) must not prematurely close
        the surrounding fence -- the whole point of a fence is that
        everything between the opening and closing marker is one block.
        """
        chunk_text = 'def f():\n    """Example:\n    ```\n    f()\n    ```\n    """'
        result = _r("doc.py", 1, 6, chunk_text=chunk_text)
        ctx = assembler.assemble([result])

        lines = ctx.text.splitlines()
        # The rendered block must open and close with a fence *longer* than
        # the embedded ``` run, so exactly two fence lines bound the whole
        # chunk (not four, which would mean the embedded ``` was read as a
        # real close/re-open).
        opening = next(ln for ln in lines if ln.startswith("```"))
        assert opening.startswith(
            "````"
        ), "fence must be longer than the embedded ``` run"
        assert ctx.text.count(opening.rstrip("python")) == 2
        # And every source line -- including the embedded ``` markers --
        # survived as numbered content inside the block.
        assert "3 |     ```" in ctx.text
        assert "5 |     ```" in ctx.text

    def test_embedded_four_backtick_run_gets_a_five_backtick_fence(
        self, assembler: ContextAssembler
    ) -> None:
        chunk_text = "a\n````\nb"
        result = _r("doc.md", 1, 3, chunk_text=chunk_text)
        ctx = assembler.assemble([result])
        assert "`````" in ctx.text

    def test_plain_code_uses_the_standard_triple_backtick_fence(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.py", 1, 1, chunk_text="x = 1")])
        assert "```python" in ctx.text
        assert "````" not in ctx.text

    def test_empty_chunk_text_does_not_crash(self, assembler: ContextAssembler) -> None:
        result = _r("empty.py", 5, 5, chunk_text="")
        ctx = assembler.assemble([result])
        assert len(ctx.chunks) == 1
        assert "## empty.py (lines 5-5)" in ctx.text

    def test_single_line_chunk_renders_correctly(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("a.py", 7, 7, chunk_text="x = 42")])
        assert "## a.py (lines 7-7)" in ctx.text
        assert "7 | x = 42" in ctx.text

    def test_standalone_chunk_with_more_physical_lines_than_declared_range(
        self, assembler: ContextAssembler
    ) -> None:
        """A single (non-merged) result whose chunk_text has more lines than
        its declared start/end range implies (e.g. a trailing blank line)
        is rendered as-is: numbering follows the actual physical lines
        starting at start_line, since there's no second contributor to
        reconcile against."""
        result = _r("a.py", 1, 2, chunk_text="one\ntwo\nthree\n")
        ctx = assembler.assemble([result])
        assert "1 | one" in ctx.text
        assert "2 | two" in ctx.text
        assert "3 | three" in ctx.text

    def test_gap_in_merged_coverage_uses_gap_marker(
        self, assembler: ContextAssembler
    ) -> None:
        """If a contributor's actual chunk_text has fewer physical lines
        than its declared range, and no other contributor in the merge
        group covers the missing line numbers, those lines render as a
        defensive gap marker rather than silently vanishing."""
        short = _r("g.py", 1, 10, score=0.9, chunk_text="only\ntwo\nlines")
        partial_cover = _r("g.py", 5, 8, score=0.1, chunk_text="a\nb\nc\nd")
        ctx = assembler.assemble([short, partial_cover])
        merged = ctx.chunks[0]
        assert (merged.start_line, merged.end_line) == (1, 10)
        # Lines 4, 9, 10 are covered by neither contributor.
        text_lines = merged.text.splitlines()
        assert text_lines[3] == "..."  # line 4
        assert text_lines[8] == "..."  # line 9
        assert text_lines[9] == "..."  # line 10
        # But the lines both/either contributor did cover are real content.
        assert text_lines[0] == "only"
        assert text_lines[4] == "a"  # line 5, from partial_cover

    def test_asymmetric_none_line_numbers_treated_as_unanchored(
        self, assembler: ContextAssembler
    ) -> None:
        """Only one of start_line/end_line being None still disqualifies a
        result from line-based merging -- both must be present."""
        result = RetrievalResult(
            score=0.5,
            file_path="a.py",
            start_line=None,
            end_line=10,
            symbol_name=None,
            chunk_text="text",
            metadata={},
        )
        ctx = assembler.assemble([result])
        assert ctx.chunks[0].start_line is None

    def test_two_unanchored_results_in_same_file_never_merge(
        self, assembler: ContextAssembler
    ) -> None:
        a = _r("sym.py", None, None, chunk_text="class Foo: ...")
        b = _r("sym.py", None, None, chunk_text="class Bar: ...")
        ctx = assembler.assemble([a, b])
        assert len(ctx.chunks) == 2

    def test_tie_break_is_deterministic_when_budget_forces_a_choice(self) -> None:
        """Two equal-score candidates where only one fits: the tiebreak
        (file_path, then start_line) must pick the same one every time,
        regardless of input order."""
        assembler = ContextAssembler(max_tokens=10_000)
        a = _r("a.py", 1, 20, score=0.5)
        z = _r("z.py", 1, 20, score=0.5)
        # Budget fits exactly one rendered chunk but not both (a.py and z.py
        # render to the same cost, so either works as the probe).
        budget = assembler.assemble([a], max_tokens=10_000).total_tokens
        forward = assembler.assemble([a, z], max_tokens=budget)
        reversed_ = assembler.assemble([z, a], max_tokens=budget)
        assert [c.file_path for c in forward.chunks] == ["a.py"]
        assert [c.file_path for c in reversed_.chunks] == ["a.py"]

    def test_file_path_with_no_extension_gets_bare_fence_language(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("Makefile", 1, 1, chunk_text="all:")])
        assert "```\n" in ctx.text

    def test_file_path_with_multiple_dots_matches_final_extension(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([_r("src/app.test.py", 1, 1, chunk_text="x = 1")])
        assert "```python" in ctx.text

    def test_zero_score_chunk_is_still_included_when_budget_allows(
        self, assembler: ContextAssembler
    ) -> None:
        """A score of exactly 0.0 is a valid (if unlikely) low-relevance
        score, not a sentinel for "exclude" -- it still competes for
        inclusion normally."""
        ctx = assembler.assemble([_r("a.py", 1, 1, score=0.0)])
        assert len(ctx.chunks) == 1

    def test_many_small_non_overlapping_chunks_across_one_file(
        self, assembler: ContextAssembler
    ) -> None:
        """Several small, well-separated ranges in the same file all
        survive as distinct chunks and stay correctly ordered."""
        results = [_r("a.py", n, n + 1, score=1.0 - n * 0.01) for n in (1, 10, 20, 30)]
        ctx = assembler.assemble(results)
        assert len(ctx.chunks) == 4
        assert [c.start_line for c in ctx.chunks] == [1, 10, 20, 30]


# ---------------------------------------------------------------------------
# Non-mutation / determinism
# ---------------------------------------------------------------------------


class TestNonMutationAndDeterminism:
    def test_does_not_mutate_input_results(self, assembler: ContextAssembler) -> None:
        results = [_r("a.py", 1, 5, score=0.7), _r("b.py", 1, 5, score=0.3)]
        originals = copy.deepcopy(results)
        assembler.assemble(results)
        for orig, after in zip(originals, results, strict=True):
            assert orig.score == after.score
            assert orig.chunk_text == after.chunk_text

    def test_repeated_calls_are_identical(self, assembler: ContextAssembler) -> None:
        results = [_r("a.py", 1, 5), _r("b.py", 10, 12), _r("a.py", 3, 8)]
        first = assembler.assemble(results)
        second = assembler.assemble(results)
        assert first.text == second.text
        assert first.total_tokens == second.total_tokens


# ---------------------------------------------------------------------------
# Empty input / input validation
# ---------------------------------------------------------------------------


class TestEmptyAndValidation:
    def test_empty_results_returns_empty_context(
        self, assembler: ContextAssembler
    ) -> None:
        ctx = assembler.assemble([])
        assert ctx == AssembledContext(text="")
        assert ctx.chunks == []
        assert ctx.total_tokens == 0
        assert ctx.truncated is False

    def test_constructor_rejects_non_positive_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ContextAssembler(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ContextAssembler(max_tokens=-5)

    def test_assemble_rejects_non_positive_max_tokens_override(
        self, assembler: ContextAssembler
    ) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            assembler.assemble([_r("a.py", 1, 1)], max_tokens=0)

    def test_default_max_tokens_used_when_not_overridden(self) -> None:
        assembler = ContextAssembler(max_tokens=42)
        assert assembler.max_tokens == 42


# ---------------------------------------------------------------------------
# End-to-end style smoke test mirroring realistic pipeline usage
# ---------------------------------------------------------------------------


class TestRealisticUsage:
    """A shape resembling reranked output flowing straight into assembly."""

    def test_reranked_results_assemble_into_ordered_deduped_context(self) -> None:
        # Two retrievers surfaced overlapping windows of the same function,
        # plus two unrelated hits elsewhere -- the shape fusion.py/reranker.py
        # would hand off.
        reranked = [
            _r("auth.py", 10, 25, score=0.95),  # top hit after rerank
            _r("session.py", 1, 5, score=0.80),
            _r("auth.py", 20, 30, score=0.40),  # overlaps the top hit
        ]
        assembler = ContextAssembler(max_tokens=5000)
        ctx = assembler.assemble(reranked)

        # auth.py's two overlapping hits merged into one 10-30 block.
        auth_chunks = [c for c in ctx.chunks if c.file_path == "auth.py"]
        assert len(auth_chunks) == 1
        assert (auth_chunks[0].start_line, auth_chunks[0].end_line) == (10, 30)

        # Display order is file-path order: auth.py before session.py.
        assert ctx.text.index("auth.py") < ctx.text.index("session.py")

        assert isinstance(ctx.chunks[0], AssembledChunk)
        assert ctx.truncated is False

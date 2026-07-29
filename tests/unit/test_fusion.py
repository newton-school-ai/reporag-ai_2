"""Unit tests for Reciprocal Rank Fusion (Issue 19).

Covers every acceptance criterion of Issue 19's RRF portion:

* fuses 2-3 ranked lists into a single ranking,
* items present in one list but not another are still scored (only lists
  where the item is present contribute -- no implicit tail penalty),
* fused output is sorted by descending RRF score with a deterministic
  tiebreak so two reruns produce identical ordering,
* the input result objects and their metadata are never mutated,
* input validation (negative k, bad top_k, non-list inputs) raises.

Fixtures deliberately use distinct (file_path, start_line, end_line) keys
matching the dedup key :mod:`reporag.retrieval.vector_search.VectorSearch`
uses, plus some overlapping keys to exercise the dedup-then-fuse path.
"""

from __future__ import annotations

import copy

import pytest

from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _r(
    file_path: str,
    start_line: int,
    *,
    end_line: int | None = None,
    score: float = 1.0,
    chunk_text: str = "",
    symbol_name: str | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult for tests."""
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line if end_line is not None else start_line,
        symbol_name=symbol_name,
        chunk_text=chunk_text or f"chunk {file_path}:{start_line}",
        metadata={"source": file_path},
    )


# ---------------------------------------------------------------------------
# Core RRF correctness
# ---------------------------------------------------------------------------


class TestRRFBasics:
    """Verify the RRF formula itself and the basic fusion contract."""

    def test_rrf_score_is_one_over_k_plus_rank_for_single_list(self) -> None:
        """A single list of two items yields 1/(k+1) and 1/(k+2)."""
        items = [_r("a.py", 1), _r("b.py", 1)]
        fused = reciprocal_rank_fusion([items], k=60)

        assert len(fused) == 2
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[1].score == pytest.approx(1 / 62)

    def test_two_lists_item_in_both_accumulates(self) -> None:
        """An item at rank 1 in two lists scores 2 * (1 / (k + 1))."""
        shared = _r("shared.py", 1)
        list_a = [shared, _r("only_a.py", 1)]
        list_b = [shared, _r("only_b.py", 1)]

        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        by_file = {r.file_path: r.score for r in fused}
        assert by_file["shared.py"] == pytest.approx(2 * (1 / 61))
        assert by_file["only_a.py"] == pytest.approx(1 / 62)
        assert by_file["only_b.py"] == pytest.approx(1 / 62)
        # The shared item, appearing at rank 1 in both lists, wins overall.
        assert fused[0].file_path == "shared.py"

    def test_three_lists_accumulate_from_all_three(self) -> None:
        shared = _r("center.py", 1)
        fused = reciprocal_rank_fusion([[shared], [shared], [shared]], k=60)
        assert fused[0].score == pytest.approx(3 * (1 / 61))

    def test_higher_rank_in_list_contributes_less(self) -> None:
        """Rank 1 > rank 2 contribution: the worse-ranked item scores less."""
        items = [_r("a.py", 1), _r("b.py", 1), _r("c.py", 1)]
        fused = reciprocal_rank_fusion([items], k=60)
        assert fused[0].file_path == "a.py"
        assert fused[1].file_path == "b.py"
        assert fused[2].file_path == "c.py"
        assert fused[0].score > fused[1].score > fused[2].score


# ---------------------------------------------------------------------------
# Acceptance: missing items
# ---------------------------------------------------------------------------


class TestMissingItems:
    """Items in one list but not another still get scored from their lists."""

    def test_item_only_in_one_list_still_appears(self) -> None:
        list_a = [_r("a.py", 1), _r("b.py", 1)]
        list_b = [_r("c.py", 1)]  # b.py is missing from list_b
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        files = {r.file_path for r in fused}
        # b.py only appears in list_a -- it must still survive the fusion.
        assert "b.py" in files
        by_file = {r.file_path: r.score for r in fused}
        assert by_file["b.py"] == pytest.approx(1 / 62)

    def test_missing_item_not_penalised_with_implicit_tail(self) -> None:
        """An item absent from a list contributes exactly 0 from that list.

        No implicit "tail rank" penalty is added for lists where the item is
        missing -- only lists where the item actually appears contribute.
        This pins the "no tail penalty" property so a future change cannot
        silently start penalising items that only one retriever surfaced.
        """
        single_list_only = _r("a.py", 1)
        fused = reciprocal_rank_fusion(
            [[single_list_only], [_r("b.py", 1), _r("c.py", 1)]], k=60
        )
        by_file = {r.file_path: r.score for r in fused}
        # a.py is absent from the second list; its score is exactly 1/61,
        # i.e. ONLY its contribution from the first list -- no tail penalty
        # was added for "not being in list 2".
        assert by_file["a.py"] == pytest.approx(1 / 61)
        # And b.py/c.py get their own single-list, single-position scores.
        assert by_file["b.py"] == pytest.approx(1 / 61)
        assert by_file["c.py"] == pytest.approx(1 / 62)


# ---------------------------------------------------------------------------
# Acceptance: deterministic ordering / tiebreak
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    """Two reruns of the same call must produce identical ordering."""

    def test_ties_break_by_first_seen_position(self) -> None:
        """Two items with identical RRF score keep their first-seen order.

        first_item and second_item each appear at rank 1 in one list and rank 2
        in the other, so both get the same total RRF score. The deterministic
        tiebreak must place the one discovered first (first_item) ahead, and
        this must be reproducible across reruns.
        """
        first_item = _r("first.py", 1)
        second_item = _r("second.py", 1)
        list_a = [first_item, second_item]
        list_b = [second_item, first_item]

        fused = reciprocal_rank_fusion([list_a, list_b], k=60)

        # Same total RRF score (rank 1 in one list + rank 2 in the other).
        assert fused[0].score == pytest.approx(fused[1].score)
        # first_item was discovered first -> it ranks ahead on the tiebreak.
        assert fused[0].file_path == "first.py"
        assert fused[1].file_path == "second.py"
        # And it is reproducible.
        fused_again = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert [r.file_path for r in fused] == [r.file_path for r in fused_again]

    def test_repeated_calls_identical(self) -> None:
        list_a = [_r("a.py", i) for i in range(1, 6)]
        list_b = [_r("b.py", i) for i in range(2, 7)]
        first = reciprocal_rank_fusion([list_a, list_b])
        second = reciprocal_rank_fusion([list_a, list_b])
        assert [r.file_path for r in first] == [r.file_path for r in second]
        assert [r.score for r in first] == [r.score for r in second]


# ---------------------------------------------------------------------------
# Acceptance: top_k cap
# ---------------------------------------------------------------------------


class TestTopK:
    """``top_k`` caps the output but defaults to "all"."""

    def test_top_k_caps_output(self) -> None:
        items = [_r(f"f{i}.py", 1) for i in range(10)]
        fused = reciprocal_rank_fusion([items], top_k=3)
        assert len(fused) == 3
        # The cap preserves the top-3 by RRF score (rank order).
        assert [r.file_path for r in fused] == ["f0.py", "f1.py", "f2.py"]

    def test_top_k_none_returns_everything(self) -> None:
        items = [_r(f"f{i}.py", 1) for i in range(7)]
        fused = reciprocal_rank_fusion([items], top_k=None)
        assert len(fused) == 7


# ---------------------------------------------------------------------------
# Dedup across lists (same identity key)
# ---------------------------------------------------------------------------


class TestDedup:
    """Same (file_path, start_line, end_line) across lists is fused once."""

    def test_duplicate_chunk_across_lists_scored_once_not_twice(self) -> None:
        # Same identity at rank 1 in both lists. NOT double-counted beyond
        # the legitimate accumulation: it should get 2 * (1/61), not 2 again
        # from appearing twice in one list.
        shared = _r("dup.py", 1)
        fused = reciprocal_rank_fusion([[shared, shared], [shared]], k=60)
        by_file = {r.file_path: r.score for r in fused}
        # The two in the first list dedup to one within the list (same key),
        # then accumulate with the one in the second list -> 2 * (1/61).
        assert by_file["dup.py"] == pytest.approx(2 * (1 / 61))
        assert len(fused) == 1

    def test_keeps_first_seen_metadata_when_dedup(self) -> None:
        a = _r("dup.py", 1, chunk_text="from vector")
        b = RetrievalResult(
            score=1.0,
            file_path="dup.py",
            start_line=1,
            end_line=1,
            symbol_name=None,
            chunk_text="from bm25",
            metadata={"source": "bm25"},
        )
        fused = reciprocal_rank_fusion([[a], [b]], k=60)
        # The first list to surface the item wins provenance.
        assert fused[0].chunk_text == "from vector"
        assert fused[0].metadata == {"source": "dup.py"}


# ---------------------------------------------------------------------------
# Non-mutation of inputs
# ---------------------------------------------------------------------------


class TestNonMutation:
    """The caller's result objects and lists must be left untouched."""

    def test_input_scores_unchanged(self) -> None:
        items = [_r("a.py", 1, score=0.9), _r("b.py", 1, score=0.5)]
        originals = copy.deepcopy(items)
        reciprocal_rank_fusion([items], k=60)
        for orig, after in zip(originals, items, strict=True):
            assert orig.score == after.score

    def test_does_not_mutate_input_metadata(self) -> None:
        item = _r("a.py", 1)
        original_meta = dict(item.metadata)
        reciprocal_rank_fusion([[item]], k=60)
        assert item.metadata == original_meta


# ---------------------------------------------------------------------------
# Input validation / friendly coercion
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 0"):
            reciprocal_rank_fusion([[_r("a.py", 1)]], k=-1)

    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reciprocal_rank_fusion([[_r("a.py", 1)]], top_k=0)

    def test_non_list_element_raises(self) -> None:
        """An element that is neither a list nor a bare RetrievalResult list
        is rejected -- bare ints are not a callable shape.
        """
        with pytest.raises(TypeError, match="must be a list"):
            reciprocal_rank_fusion([2, 3], k=60)

    def test_empty_inputs_return_empty(self) -> None:
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[]], k=60) == []
        assert reciprocal_rank_fusion([[], [], []], k=60) == []

    def test_k_zero_rank_one_item_skipped_to_avoid_div_by_zero(self) -> None:
        """k=0 with a rank-1 item would divide by zero; it gets skipped."""
        items = [_r("a.py", 1), _r("b.py", 1)]
        fused = reciprocal_rank_fusion([items], k=0)
        # rank 1 item (a.py) is skipped -> only b.py (rank 2, 1/(0+2)=0.5).
        assert len(fused) == 1
        assert fused[0].file_path == "b.py"
        assert fused[0].score == pytest.approx(1 / 2)

    def test_bare_list_coerced_to_single_list(self) -> None:
        """A single list passed without the outer wrapper is accepted."""
        items = [_r("a.py", 1), _r("b.py", 1)]
        fused = reciprocal_rank_fusion(items, k=60)
        assert len(fused) == 2


# ---------------------------------------------------------------------------
# End-to-end style smoke test mirroring the issue spec usage
# ---------------------------------------------------------------------------


class TestIssueSpecUsage:
    """The exact shape from the issue's "How to test locally" snippet."""

    def test_fuse_three_lists_then_rerank_shape(self) -> None:
        # Stand in for the issue's vector/bm25/graph outputs.
        vector_results = [_r("auth.py", 10), _r("auth.py", 40), _r("login.py", 1)]
        bm25_results = [_r("auth.py", 10), _r("session.py", 5)]
        graph_results = [_r("auth.py", 10), _r("router.py", 3)]

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results, graph_results], k=60
        )
        # auth.py:10 appears at rank 1 in all three lists -> 3 * (1/61).
        assert fused[0].file_path == "auth.py"
        assert fused[0].start_line == 10
        assert fused[0].score == pytest.approx(3 * (1 / 61))
        # Total unique items = 5 (auth.py:10 dedups across lists).
        assert len(fused) == 5

"""Unit tests for the Reciprocal Rank Fusion module (Issue 19).

Builds small hand-crafted ranked lists (rather than running real
vector/BM25/graph searches) so the RRF math itself -- the acceptance
criteria's actual subject -- is tested in isolation and offline.
"""

from __future__ import annotations

import math

import pytest

from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.vector_search import RetrievalResult


def _result(
    file_path: str,
    score: float = 1.0,
    start_line: int | None = 1,
    end_line: int | None = 5,
    symbol_name: str | None = None,
    chunk_text: str = "code",
    metadata: dict | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult for a given (file, line-range) identity."""
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        chunk_text=chunk_text,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Core RRF math
# ---------------------------------------------------------------------------


def test_fuses_two_lists_with_correct_scores() -> None:
    """RRF score for an item present in both lists is the sum of 1/(k+rank)."""
    a = _result("a.py")
    b = _result("b.py")

    list1 = [a, b]  # a rank 1, b rank 2
    list2 = [b, a]  # b rank 1, a rank 2

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    by_path = {r.file_path: r.score for r in fused}
    expected = 1 / (60 + 1) + 1 / (60 + 2)
    assert math.isclose(by_path["a.py"], expected, rel_tol=1e-9)
    assert math.isclose(by_path["b.py"], expected, rel_tol=1e-9)
    # Both items score identically (symmetric ranks) -- order between them
    # is not asserted, but both must be present.
    assert set(by_path) == {"a.py", "b.py"}


def test_fuses_three_lists() -> None:
    """RRF correctly sums contributions across three ranked lists."""
    a, b, c = _result("a.py"), _result("b.py"), _result("c.py")

    vector_list = [a, b, c]
    bm25_list = [b, a, c]
    graph_list = [a, c, b]

    fused = reciprocal_rank_fusion([vector_list, bm25_list, graph_list], k=60)
    by_path = {r.file_path: r.score for r in fused}

    expected_a = 1 / 61 + 1 / 62 + 1 / 61  # ranks 1, 2, 1
    expected_b = 1 / 62 + 1 / 61 + 1 / 63  # ranks 2, 1, 3
    expected_c = 1 / 63 + 1 / 63 + 1 / 62  # ranks 3, 3, 2

    assert math.isclose(by_path["a.py"], expected_a, rel_tol=1e-9)
    assert math.isclose(by_path["b.py"], expected_b, rel_tol=1e-9)
    assert math.isclose(by_path["c.py"], expected_c, rel_tol=1e-9)


def test_result_sorted_by_rrf_score_descending() -> None:
    """The fused list is sorted best-first by RRF score."""
    a, b, c = _result("a.py"), _result("b.py"), _result("c.py")
    fused = reciprocal_rank_fusion([[a, b, c], [a, b, c]], k=60)
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0].file_path == "a.py"  # consistently rank 1 in both lists


# ---------------------------------------------------------------------------
# Handling items missing from some lists
# ---------------------------------------------------------------------------


def test_item_missing_from_one_list_only_counts_present_lists() -> None:
    """An item absent from a list is not penalized -- it just gets 0 from it."""
    a = _result("a.py")
    b = _result("b.py")  # only appears in list1

    list1 = [b, a]  # b rank 1, a rank 2
    list2 = [a]  # a rank 1, b absent

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    by_path = {r.file_path: r.score for r in fused}

    expected_a = 1 / 62 + 1 / 61  # rank 2 in list1, rank 1 in list2
    expected_b = 1 / 61  # rank 1 in list1 only

    assert math.isclose(by_path["a.py"], expected_a, rel_tol=1e-9)
    assert math.isclose(by_path["b.py"], expected_b, rel_tol=1e-9)
    assert len(fused) == 2


def test_item_present_in_only_one_of_three_lists() -> None:
    """An item unique to one of three lists still appears in the fused output."""
    unique = _result("unique.py")
    common = _result("common.py")

    fused = reciprocal_rank_fusion([[common], [common], [unique, common]], k=60)
    paths = {r.file_path for r in fused}
    assert paths == {"common.py", "unique.py"}
    # common.py appears in all 3 lists, unique.py only in 1 -> common wins
    by_path = {r.file_path: r.score for r in fused}
    assert by_path["common.py"] > by_path["unique.py"]


def test_empty_ranked_lists_returns_empty() -> None:
    """No lists at all fuses to an empty result."""
    assert reciprocal_rank_fusion([]) == []


def test_all_empty_lists_returns_empty() -> None:
    """Lists that are individually empty fuse to an empty result."""
    assert reciprocal_rank_fusion([[], [], []]) == []


def test_single_list_still_gets_fused() -> None:
    """A single ranked list is still re-scored (not just passed through)."""
    a, b = _result("a.py"), _result("b.py")
    fused = reciprocal_rank_fusion([[a, b]], k=60)
    assert len(fused) == 2
    assert fused[0].file_path == "a.py"
    assert math.isclose(fused[0].score, 1 / 61, rel_tol=1e-9)
    assert math.isclose(fused[1].score, 1 / 62, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Configurable k
# ---------------------------------------------------------------------------


def test_default_k_uses_settings_rrf_constant() -> None:
    """Omitting k falls back to settings.rrf_constant."""
    from reporag.config import settings

    a = _result("a.py")
    fused_default = reciprocal_rank_fusion([[a]])
    fused_explicit = reciprocal_rank_fusion([[a]], k=settings.rrf_constant)
    assert fused_default[0].score == fused_explicit[0].score


def test_smaller_k_increases_scores() -> None:
    """A smaller k produces larger RRF scores (1/(k+r) grows as k shrinks)."""
    a = _result("a.py")
    fused_small_k = reciprocal_rank_fusion([[a]], k=1)
    fused_large_k = reciprocal_rank_fusion([[a]], k=1000)
    assert fused_small_k[0].score > fused_large_k[0].score


def test_k_must_be_positive() -> None:
    """k <= 0 raises ValueError."""
    a = _result("a.py")
    with pytest.raises(ValueError, match="k must be > 0"):
        reciprocal_rank_fusion([[a]], k=0)
    with pytest.raises(ValueError, match="k must be > 0"):
        reciprocal_rank_fusion([[a]], k=-5)


# ---------------------------------------------------------------------------
# Weights and source names
# ---------------------------------------------------------------------------


def test_weights_scale_each_lists_contribution() -> None:
    """A list weighted 2x contributes twice as much to an item's score."""
    a = _result("a.py")
    fused = reciprocal_rank_fusion([[a], [a]], k=60, weights=[2.0, 1.0])
    expected = 2.0 / 61 + 1.0 / 61
    assert math.isclose(fused[0].score, expected, rel_tol=1e-9)


def test_weights_length_mismatch_raises() -> None:
    a = _result("a.py")
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[a], [a]], weights=[1.0])


def test_source_names_length_mismatch_raises() -> None:
    a = _result("a.py")
    with pytest.raises(ValueError, match="source_names"):
        reciprocal_rank_fusion([[a], [a]], source_names=["vector"])


def test_source_names_recorded_in_metadata() -> None:
    """Per-source scores/ranks land in metadata under the given labels."""
    a_vec = _result("a.py", score=0.9)
    a_bm25 = _result("a.py", score=12.3)

    fused = reciprocal_rank_fusion([[a_vec], [a_bm25]], source_names=["vector", "bm25"])
    meta = fused[0].metadata
    assert meta["rrf_source_scores"] == {"vector": 0.9, "bm25": 12.3}
    assert meta["rrf_source_ranks"] == {"vector": 1, "bm25": 1}


def test_default_source_names_are_positional() -> None:
    a = _result("a.py")
    fused = reciprocal_rank_fusion([[a], [a]])
    assert set(fused[0].metadata["rrf_source_scores"]) == {"list_0", "list_1"}


# ---------------------------------------------------------------------------
# Representative result selection
# ---------------------------------------------------------------------------


def test_representative_prefers_longer_chunk_text() -> None:
    """When the same item appears twice, the fuller chunk_text is kept."""
    short = _result("a.py", chunk_text="def f(): ...")
    long = _result(
        "a.py", chunk_text="def f():\n    # a full implementation\n    return 42"
    )

    fused = reciprocal_rank_fusion([[short], [long]])
    assert fused[0].chunk_text == long.chunk_text


def test_representative_preserves_symbol_name() -> None:
    a = _result("a.py", symbol_name="my_function")
    fused = reciprocal_rank_fusion([[a]])
    assert fused[0].symbol_name == "my_function"


def test_original_metadata_is_preserved_alongside_rrf_fields() -> None:
    a = _result("a.py", metadata={"language": "python"})
    fused = reciprocal_rank_fusion([[a]])
    assert fused[0].metadata["language"] == "python"
    assert "rrf_source_scores" in fused[0].metadata


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_fusion_does_not_mutate_callers_original_metadata() -> None:
    """The input RetrievalResult's metadata dict is never modified in place."""
    original_metadata = {"language": "python"}
    a = _result("a.py", metadata=original_metadata)
    reciprocal_rank_fusion([[a]])
    assert original_metadata == {"language": "python"}
    assert "rrf_source_scores" not in original_metadata


def test_duplicate_item_within_a_single_list_is_deduped_not_double_counted() -> None:
    """A caller-side bug (same item twice in one list) no longer inflates its score.

    RRF assumes each input list is a clean ranking with no repeats. If an
    item somehow appears twice within one list, only its first (best-ranked)
    occurrence counts towards that list's contribution -- the later repeat
    is skipped rather than adding a second contribution. Upstream callers
    (VectorSearch, BM25Search) already dedupe their own output, so this is
    a defensive guard rather than something expected to occur in practice.
    """
    a = _result("a.py")
    b = _result("b.py")
    fused = reciprocal_rank_fusion([[a, b, a]])  # a at rank 1, repeated at rank 3
    by_path = {r.file_path: r.score for r in fused}
    expected_a = 1 / 61  # only the first occurrence (rank 1) counts
    expected_b = 1 / 62  # b's rank is unaffected by a's later repeat
    assert math.isclose(by_path["a.py"], expected_a, rel_tol=1e-9)
    assert math.isclose(by_path["b.py"], expected_b, rel_tol=1e-9)


def test_duplicate_across_different_lists_is_still_counted_once_per_list() -> None:
    """Deduping is per-list only -- the same item legitimately appearing in
    multiple different lists still gets credit from each of them."""
    a = _result("a.py")
    fused = reciprocal_rank_fusion([[a], [a]])
    expected = 1 / 61 + 1 / 61
    assert math.isclose(fused[0].score, expected, rel_tol=1e-9)


def test_distinct_location_less_results_are_not_merged() -> None:
    """Two genuinely different results with no location info stay distinct.

    Malformed or synthetic results (e.g. graph nodes with no resolvable
    file location) that all carry empty/None location fields fall back to
    a content-hash identity (symbol_name + chunk_text) instead of colliding
    on a shared ``("", None, None)`` key, so different content is never
    silently merged just because both are missing a location.
    """
    x = _result("", start_line=None, end_line=None, chunk_text="def foo(): pass")
    y = _result("", start_line=None, end_line=None, chunk_text="def bar(): pass")
    fused = reciprocal_rank_fusion([[x], [y]])
    assert len(fused) == 2
    assert {r.chunk_text for r in fused} == {"def foo(): pass", "def bar(): pass"}


def test_identical_location_less_results_are_still_merged() -> None:
    """Two location-less results with the SAME content are correctly recognized
    as the same item and merged (the content-hash fallback isn't a blanket
    "never merge" rule -- it only stops *distinct* content from colliding)."""
    x = _result(
        "",
        start_line=None,
        end_line=None,
        symbol_name="foo",
        chunk_text="def foo(): pass",
    )
    y = _result(
        "",
        start_line=None,
        end_line=None,
        symbol_name="foo",
        chunk_text="def foo(): pass",
    )
    fused = reciprocal_rank_fusion([[x], [y]])
    assert len(fused) == 1
    expected = 1 / 61 + 1 / 61
    assert math.isclose(fused[0].score, expected, rel_tol=1e-9)


def test_representative_tie_break_prefers_first_list_when_chunk_text_equal() -> None:
    """When chunk_text length ties, the earliest-encountered occurrence wins."""
    from_vector = _result("a.py", chunk_text="same length", metadata={"src": "vector"})
    from_bm25 = _result("a.py", chunk_text="same length", metadata={"src": "bm25"})
    fused = reciprocal_rank_fusion([[from_vector], [from_bm25]])
    assert fused[0].metadata["src"] == "vector"

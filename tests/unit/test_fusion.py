from __future__ import annotations

import pytest

from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.vector_search import RetrievalResult


def make_result(file_path: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=1,
        end_line=10,
        symbol_name="test_symbol",
        chunk_text="def test_symbol(): pass",
        metadata={},
    )


class TestReciprocalRankFusion:
    def test_single_list(self) -> None:
        """A single list should return the same ordering, re-scored with RRF."""
        lst = [
            make_result("file_a.py"),
            make_result("file_b.py"),
        ]
        fused = reciprocal_rank_fusion([lst], k=60)

        assert len(fused) == 2
        assert fused[0].file_path == "file_a.py"
        assert fused[1].file_path == "file_b.py"

        assert fused[0].score == pytest.approx(1.0 / (60 + 1))
        assert fused[1].score == pytest.approx(1.0 / (60 + 2))

    def test_multiple_lists_merge(self) -> None:
        """Items appearing in multiple lists should have their RRF scores summed."""
        lst1 = [make_result("file_a.py"), make_result("file_b.py")]
        lst2 = [make_result("file_b.py"), make_result("file_c.py")]

        fused = reciprocal_rank_fusion([lst1, lst2], k=60)

        assert len(fused) == 3
        # file_b.py appears at rank 2 in lst1 and rank 1 in lst2
        expected_b = (1.0 / 62) + (1.0 / 61)
        expected_a = 1.0 / 61
        expected_c = 1.0 / 62

        assert fused[0].file_path == "file_b.py"
        assert fused[0].score == pytest.approx(expected_b)

        assert fused[1].file_path == "file_a.py"
        assert fused[1].score == pytest.approx(expected_a)

        assert fused[2].file_path == "file_c.py"
        assert fused[2].score == pytest.approx(expected_c)

    def test_empty_lists(self) -> None:
        """Empty lists should be handled gracefully."""
        fused = reciprocal_rank_fusion([[], []])
        assert len(fused) == 0

    def test_partially_empty_lists(self) -> None:
        """Fusion works when some retrievers return nothing."""
        lst = [make_result("file_a.py")]
        fused = reciprocal_rank_fusion([lst, [], []], k=60)

        assert len(fused) == 1
        assert fused[0].file_path == "file_a.py"
        assert fused[0].score == pytest.approx(1.0 / 61)

    def test_configurable_k(self) -> None:
        """The constant k should be configurable."""
        lst = [make_result("file_a.py")]
        fused = reciprocal_rank_fusion([lst], k=10)

        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1.0 / (10 + 1))

    def test_duplicate_documents_in_same_list(self) -> None:
        """If a retriever returns duplicates, only the highest rank is counted."""
        lst = [make_result("file_a.py"), make_result("file_a.py")]
        fused = reciprocal_rank_fusion([lst], k=60)

        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1.0 / 61)

    def test_deterministic_ordering_on_ties(self) -> None:
        """Items with the exact same RRF score sort deterministically by attributes."""
        # Both appear at rank 1 in their respective lists, so score is identical
        lst1 = [make_result("file_z.py")]
        lst2 = [make_result("file_a.py")]
        fused = reciprocal_rank_fusion([lst1, lst2], k=60)

        assert len(fused) == 2
        assert fused[0].score == fused[1].score
        # 'file_a.py' should come before 'file_z.py' lexicographically
        assert fused[0].file_path == "file_a.py"
        assert fused[1].file_path == "file_z.py"

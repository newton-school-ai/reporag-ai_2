from __future__ import annotations

from reporag.retrieval.reranker import CrossEncoderReranker
from reporag.retrieval.vector_search import RetrievalResult


class MockModel:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Simple mock logic: length of chunk text (just for deterministic sorting)
        return [float(len(chunk)) for _, chunk in pairs]


def make_result(chunk_text: str) -> RetrievalResult:
    return RetrievalResult(
        score=1.0,
        file_path="file.py",
        start_line=1,
        end_line=10,
        symbol_name="sym",
        chunk_text=chunk_text,
        metadata={},
    )


class TestCrossEncoderReranker:
    def test_rerank_empty(self) -> None:
        """Reranking an empty list should return an empty list."""
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    def test_rerank_ordering_and_top_k(self) -> None:
        """The reranker should update scores and sort candidates descending."""
        reranker = CrossEncoderReranker()

        # Inject the mock model directly so we don't download the real one
        reranker._model = MockModel()

        candidates = [
            make_result("short"),
            make_result("a very long chunk of text indeed"),
            make_result("medium length"),
        ]

        reranked = reranker.rerank("query", candidates, top_k=2)

        assert len(reranked) == 2
        # "a very long chunk of text indeed" is the longest (score 32)
        assert reranked[0].chunk_text == "a very long chunk of text indeed"
        assert reranked[0].score == 32.0

        # "medium length" is next (score 13)
        assert reranked[1].chunk_text == "medium length"
        assert reranked[1].score == 13.0

    def test_deterministic_ordering_on_ties(self) -> None:
        """Candidates with tied reranker scores sort deterministically."""
        reranker = CrossEncoderReranker()
        reranker._model = MockModel()

        # Both chunks have length 4, so their mock scores will be 4.0
        c1 = make_result("four")
        c1.file_path = "z_file.py"

        c2 = make_result("four")
        c2.file_path = "a_file.py"

        reranked = reranker.rerank("query", [c1, c2], top_k=2)

        assert len(reranked) == 2
        assert reranked[0].score == reranked[1].score == 4.0
        # a_file.py should precede z_file.py
        assert reranked[0].file_path == "a_file.py"
        assert reranked[1].file_path == "z_file.py"

    def test_fusion_integration(self) -> None:
        """Tests that fusion output passes cleanly into the reranker."""
        from reporag.retrieval.fusion import reciprocal_rank_fusion

        c1 = make_result("short")
        c1.file_path = "short.py"

        c2 = make_result("longest text chunk")
        c2.file_path = "long.py"

        lst1 = [c1, c2]
        lst2 = [c2]

        fused = reciprocal_rank_fusion([lst1, lst2], k=60)
        # fused has 2 distinct items

        reranker = CrossEncoderReranker()
        reranker._model = MockModel()

        reranked = reranker.rerank("query", fused, top_k=1)

        assert len(reranked) == 1
        assert reranked[0].chunk_text == "longest text chunk"
        # Original score should have been replaced by the model's score
        assert reranked[0].score == len("longest text chunk")

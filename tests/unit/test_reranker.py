"""Unit tests for the cross-encoder reranker module (Issue 19).

Uses a fake ``CrossEncoder`` (duck-typed on ``.predict(pairs, **kwargs)``)
injected via the ``model=`` constructor argument, so these tests run
offline and fast without downloading ``cross-encoder/ms-marco-MiniLM-L-6-v2``,
while still exercising the real batching/scoring/reordering logic.
"""

from __future__ import annotations

import time

import pytest

from reporag.retrieval.reranker import CrossEncoderReranker
from reporag.retrieval.vector_search import RetrievalResult


def _result(
    file_path: str,
    chunk_text: str,
    score: float = 1.0,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=1,
        end_line=5,
        symbol_name=None,
        chunk_text=chunk_text,
        metadata=metadata or {},
    )


class FakeCrossEncoder:
    """A deterministic fake scoring (query, chunk) pairs by keyword overlap.

    Score = number of query words also present in the chunk text. This is
    enough to build cases where the "correct" (keyword-relevant) chunk
    isn't first in the RRF-fused input order, so reranking has to actually
    reorder to be verified as working.
    """

    def __init__(self) -> None:
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.predict_calls.append(list(pairs))
        scores = []
        for query, chunk in pairs:
            query_words = set(query.lower().split())
            chunk_words = set(chunk.lower().split())
            scores.append(float(len(query_words & chunk_words)))
        return scores


@pytest.fixture
def fake_model() -> FakeCrossEncoder:
    return FakeCrossEncoder()


@pytest.fixture
def reranker(fake_model: FakeCrossEncoder) -> CrossEncoderReranker:
    return CrossEncoderReranker(model=fake_model)


# ---------------------------------------------------------------------------
# Basic reranking / reordering
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_cross_encoder_score(reranker: CrossEncoderReranker) -> None:
    """A candidate irrelevant by keyword overlap moves behind a relevant one."""
    irrelevant = _result("b.py", "totally unrelated boilerplate")
    relevant = _result("a.py", "authenticate user session token")

    # Fed in with the irrelevant one first (e.g. it ranked higher via RRF).
    reranked = reranker.rerank("authenticate user", [irrelevant, relevant])

    assert reranked[0].file_path == "a.py"
    assert reranked[1].file_path == "b.py"


def test_rerank_outperforms_input_order(reranker: CrossEncoderReranker) -> None:
    """Reranked top-1 is the truly best-matching candidate, unlike the input order."""
    candidates = [
        _result("noise.py", "print hello world"),
        _result("close.py", "get user by name"),
        _result("best.py", "get user by id retrieves a user session"),
    ]
    reranked = reranker.rerank("get user by id", candidates)
    assert reranked[0].file_path == "best.py"


def test_rerank_score_overwrites_original_score(reranker: CrossEncoderReranker) -> None:
    """The result's .score becomes the cross-encoder score, not the input score."""
    candidate = _result("a.py", "get user by id", score=0.1234)
    reranked = reranker.rerank("get user by id", [candidate])
    assert reranked[0].score != 0.1234
    assert reranked[0].score == reranked[0].metadata["rerank_score"]


def test_pre_rerank_score_preserved_in_metadata(reranker: CrossEncoderReranker) -> None:
    """The original (pre-rerank) score is retained for inspection."""
    candidate = _result("a.py", "get user by id", score=0.5)
    reranked = reranker.rerank("get user by id", [candidate])
    assert reranked[0].metadata["pre_rerank_score"] == 0.5


def test_original_metadata_preserved(reranker: CrossEncoderReranker) -> None:
    candidate = _result("a.py", "get user by id", metadata={"language": "python"})
    reranked = reranker.rerank("query", [candidate])
    assert reranked[0].metadata["language"] == "python"


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


def test_rerank_respects_top_k(reranker: CrossEncoderReranker) -> None:
    candidates = [
        _result("a.py", "get user by id"),
        _result("b.py", "get user by name"),
        _result("c.py", "delete session"),
    ]
    reranked = reranker.rerank("get user", candidates, top_k=2)
    assert len(reranked) == 2


def test_rerank_top_k_none_returns_all(reranker: CrossEncoderReranker) -> None:
    candidates = [_result("a.py", "x"), _result("b.py", "y"), _result("c.py", "z")]
    reranked = reranker.rerank("query", candidates, top_k=None)
    assert len(reranked) == 3


def test_rerank_top_k_zero_raises(reranker: CrossEncoderReranker) -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        reranker.rerank("query", [_result("a.py", "x")], top_k=0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rerank_empty_candidates_returns_empty_without_loading_model(
    fake_model: FakeCrossEncoder,
) -> None:
    """Empty candidates short-circuits before the model is ever touched."""
    reranker = CrossEncoderReranker(model=fake_model)
    result = reranker.rerank("query", [])
    assert result == []
    assert fake_model.predict_calls == []


def test_rerank_single_candidate(reranker: CrossEncoderReranker) -> None:
    reranked = reranker.rerank("query", [_result("a.py", "query text")])
    assert len(reranked) == 1


# ---------------------------------------------------------------------------
# Batched scoring (one predict() call, not N)
# ---------------------------------------------------------------------------


def test_scoring_uses_a_single_batched_predict_call(
    reranker: CrossEncoderReranker, fake_model: FakeCrossEncoder
) -> None:
    """All candidates are scored in exactly one predict() call, not per-item."""
    candidates = [_result(f"{i}.py", f"chunk {i}") for i in range(20)]
    reranker.rerank("query", candidates)
    assert len(fake_model.predict_calls) == 1
    assert len(fake_model.predict_calls[0]) == 20


def test_pairs_passed_to_model_match_query_and_chunk_text(
    reranker: CrossEncoderReranker, fake_model: FakeCrossEncoder
) -> None:
    candidates = [_result("a.py", "chunk one"), _result("b.py", "chunk two")]
    reranker.rerank("my query", candidates)
    assert fake_model.predict_calls[0] == [
        ("my query", "chunk one"),
        ("my query", "chunk two"),
    ]


# ---------------------------------------------------------------------------
# Performance: reranking ~20 candidates should be fast
# ---------------------------------------------------------------------------


def test_reranking_20_candidates_is_fast(reranker: CrossEncoderReranker) -> None:
    """With a lightweight scoring function, 20 candidates rerank well under 500ms.

    This exercises the batching/sorting overhead in ``rerank()`` itself; it
    does not measure real cross-encoder model inference latency (that
    depends on hardware and should be benchmarked separately with the real
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model).
    """
    candidates = [
        _result(f"{i}.py", f"some chunk of code number {i}") for i in range(20)
    ]
    start = time.perf_counter()
    reranker.rerank("some query about code", candidates)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_constructing_reranker_does_not_load_model() -> None:
    """Just constructing a CrossEncoderReranker touches no network/model."""
    reranker = CrossEncoderReranker()
    assert reranker._model is None  # not loaded yet


def test_injected_model_is_used_without_loading_from_hub(
    fake_model: FakeCrossEncoder,
) -> None:
    reranker = CrossEncoderReranker(model=fake_model)
    assert reranker.model is fake_model


def test_default_model_name_from_settings() -> None:
    from reporag.config import settings

    reranker = CrossEncoderReranker()
    assert reranker.model_name == settings.reranker_model


def test_custom_model_name_overrides_settings() -> None:
    reranker = CrossEncoderReranker(model_name="custom/cross-encoder")
    assert reranker.model_name == "custom/cross-encoder"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_top_k_larger_than_candidate_count_returns_all(
    reranker: CrossEncoderReranker,
) -> None:
    """top_k exceeding the candidate count is not an error -- just returns everything."""
    candidates = [_result("a.py", "x"), _result("b.py", "y")]
    reranked = reranker.rerank("query", candidates, top_k=50)
    assert len(reranked) == 2


def test_negative_top_k_raises(reranker: CrossEncoderReranker) -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        reranker.rerank("query", [_result("a.py", "x")], top_k=-1)


def test_empty_query_does_not_raise(reranker: CrossEncoderReranker) -> None:
    """An empty-string query is scored like any other -- not a special case."""
    candidates = [_result("a.py", "x"), _result("b.py", "y")]
    reranked = reranker.rerank("", candidates)
    assert len(reranked) == 2


def test_candidate_with_empty_chunk_text_does_not_crash(
    reranker: CrossEncoderReranker,
) -> None:
    """A candidate with empty chunk_text is scored (likely low) rather than erroring."""
    candidates = [_result("a.py", ""), _result("b.py", "some real content")]
    reranked = reranker.rerank("query", candidates)
    assert len(reranked) == 2


def test_rerank_does_not_mutate_input_list_or_candidates(
    reranker: CrossEncoderReranker,
) -> None:
    """rerank() returns new RetrievalResult objects; the input list/order is untouched."""
    candidates = [_result("a.py", "x"), _result("b.py", "yyyyy")]
    original_order = [c.file_path for c in candidates]
    original_scores = [c.score for c in candidates]

    reranker.rerank("query", candidates)

    assert [c.file_path for c in candidates] == original_order
    assert [c.score for c in candidates] == original_scores


def test_all_candidates_tie_in_score_returns_all_without_error(
    reranker: CrossEncoderReranker,
) -> None:
    """Every candidate scoring identically is a valid (if uninformative) outcome."""
    candidates = [_result(f"{i}.py", "same") for i in range(5)]
    reranked = reranker.rerank("q", candidates)
    assert len(reranked) == 5
    assert len({r.score for r in reranked}) == 1

"""Unit tests for the cross-encoder reranker (Issue 19).

Covers every acceptance criterion of Issue 19's reranker portion:

* reranks candidates and reorders by ``rerank_score``,
* batched single ``predict`` call is used (verifies the intimacy the issue
  spec asks for, and keeps reranking latency under the 500 ms / 20-candidate
  budget on CPU),
* deterministic tiebreak by original input position,
* lazy model loading: construction does NOT load the model, and an empty
  candidate list short-circuits before any load happens,
* the caller's input results and metadata are never mutated, and the
  returned ``metadata`` carries the new ``"rerank_score"`` key,
* input validation (top_k < 1, max_length < 1).

A ``_FakeCrossEncoder`` whose ``predict`` is controllable stands in for the
real HuggingFace ``CrossEncoder``, keeping every test network-free -- the
same seam
:class:`~reporag.embedding.doc_embedder.DocEmbedder` already uses in
``test_doc_embedder.py``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from reporag.retrieval.reranker import CrossEncoderReranker
from reporag.retrieval.vector_search import RetrievalResult

# ============================================================================
# Test doubles
# ============================================================================


class _FakeCrossEncoder:
    """Minimal stand-in for ``sentence_transformers.CrossEncoder``.

    Records every call to ``predict`` so tests can assert batching and
    pair contents, and returns a caller-supplied sequence of scores aligned
    with the input pairs (default: flat zeros, i.e. no reordering signal
    beyond the deterministic tiebreak).
    """

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls.append(pairs)
        if self._scores is None:
            return [0.0] * len(pairs)
        if len(self._scores) != len(pairs):
            raise AssertionError(
                f"_FakeCrossEncoder configured with {len(self._scores)} "
                f"scores but predict got {len(pairs)} pairs"
            )
        return list(self._scores)


def _r(
    file_path: str,
    start_line: int,
    *,
    chunk_text: str = "",
    score: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult for reranker tests."""
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line,
        symbol_name=None,
        chunk_text=chunk_text or f"chunk {file_path}:{start_line}",
        metadata=metadata if metadata is not None else {"source": file_path},
    )


# ============================================================================
# Construction / lazy loading / validation
# ============================================================================


class TestConstruction:
    def test_construction_does_not_load_model(self) -> None:
        """Building the reranker must be side-effect-free (no model load)."""
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker.is_loaded is False
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_pre_injected_instance_is_respected_not_loaded(self) -> None:
        """Passing a model instance is the test seam; ``_ensure_loaded`` is a no-op."""
        fake = _FakeCrossEncoder()
        reranker = CrossEncoderReranker(fake, device="cpu")
        assert reranker.is_loaded is False
        reranker.rerank("q", [_r("a.py", 1, chunk_text="t")])
        # Already provided, so is_loaded becomes True without ever importing
        # sentence_transformers.
        assert reranker.is_loaded is True

    def test_max_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_length must be >= 1"):
            CrossEncoderReranker(max_length=0)

    def test_top_k_must_be_positive(self) -> None:
        fake = _FakeCrossEncoder(scores=[0.5])
        reranker = CrossEncoderReranker(fake)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reranker.rerank("q", [_r("a.py", 1)], top_k=0)

    def test_repr_shows_state(self) -> None:
        fake = _FakeCrossEncoder()
        reranker = CrossEncoderReranker(fake)
        text = repr(reranker)
        assert "CrossEncoderReranker" in text
        assert "loaded=False" in text


# ============================================================================
# Reranking correctness
# ============================================================================


class TestRerank:
    def test_empty_candidates_short_circuits_without_loading(self) -> None:
        """No candidates -> no model load (keeps the empty path side-effect-free)."""
        reranker = CrossEncoderReranker(model="cross-encoder/x")
        assert reranker.rerank("q", []) == []
        # The model must NOT have been loaded for an empty candidate list.
        assert reranker.is_loaded is False

    def test_reranks_by_cross_encoder_score_descending(self) -> None:
        """The highest cross-encoder score must surface as the top result."""
        candidates = [
            _r("a.py", 1, chunk_text="a"),
            _r("b.py", 1, chunk_text="b"),
            _r("c.py", 1, chunk_text="c"),
        ]
        # Cross-encoder says candidate B is most relevant, then A, then C.
        fake = _FakeCrossEncoder(scores=[0.1, 0.9, 0.05])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)

        assert [r.file_path for r in reranked] == ["b.py", "a.py", "c.py"]
        assert reranked[0].score == pytest.approx(0.9)
        assert reranked[1].score == pytest.approx(0.1)
        assert reranked[2].score == pytest.approx(0.05)

    def test_pairs_are_query_and_chunk_text(self) -> None:
        """``predict`` must receive ``(query, chunk_text)`` pairs in order."""
        candidates = [
            _r("a.py", 1, chunk_text="alpha"),
            _r("b.py", 1, chunk_text="beta"),
        ]
        fake = _FakeCrossEncoder(scores=[0.3, 0.7])
        CrossEncoderReranker(fake, device="cpu").rerank("my query", candidates)

        assert fake.predict_calls == [[("my query", "alpha"), ("my query", "beta")]]

    def test_single_batched_predict_call(self) -> None:
        """All pairs are scored in exactly one ``predict`` call (issue: batching)."""
        candidates = [_r(f"f{i}.py", i, chunk_text=f"t{i}") for i in range(5)]
        fake = _FakeCrossEncoder(scores=[0.1] * 5)
        CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        assert len(fake.predict_calls) == 1

    def test_top_k_truncates(self) -> None:
        candidates = [
            _r("a.py", 1, chunk_text="a"),
            _r("b.py", 1, chunk_text="b"),
            _r("c.py", 1, chunk_text="c"),
        ]
        fake = _FakeCrossEncoder(scores=[0.1, 0.9, 0.5])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank(
            "q", candidates, top_k=2
        )
        assert len(reranked) == 2
        # Top 2 by score are B (0.9) and C (0.5).
        assert [r.file_path for r in reranked] == ["b.py", "c.py"]

    def test_top_k_none_returns_all(self) -> None:
        candidates = [_r(f"f{i}.py", i, chunk_text=f"t{i}") for i in range(4)]
        fake = _FakeCrossEncoder(scores=[0.4, 0.1, 0.9, 0.3])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        assert len(reranked) == 4


# ============================================================================
# Tiebreak determinism
# ============================================================================


class TestTiebreak:
    def test_ties_break_by_original_input_position(self) -> None:
        """Equal cross-encoder scores keep the candidate's original order.

        Without a deterministic secondary key, two equal-score candidates
        could reorder nondeterministically across Python reruns (dict / set
        ordering), which would make a reranker hard to test against. The
        reranker must fall back to original input index.
        """
        candidates = [
            _r("a.py", 1, chunk_text="a"),
            _r("b.py", 1, chunk_text="b"),
            _r("c.py", 1, chunk_text="c"),
        ]
        fake = _FakeCrossEncoder(scores=[0.5, 0.5, 0.9])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        # c.py (0.9) is alone at the top; the 0.5 tie between a and b keeps
        # their original order: a before b.
        assert [r.file_path for r in reranked] == ["c.py", "a.py", "b.py"]

    def test_rerank_is_reproducible(self) -> None:
        candidates = [
            _r("a.py", 1, chunk_text="a"),
            _r("b.py", 1, chunk_text="b"),
        ]
        fake = _FakeCrossEncoder(scores=[0.2, 0.2])
        r1 = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        r2 = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        assert [r.file_path for r in r1] == [r.file_path for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]


# ============================================================================
# Non-mutation / provenance preservation
# ============================================================================


class TestNonMutation:
    def test_input_candidates_not_mutated(self) -> None:
        candidates = [_r("a.py", 1, chunk_text="a", metadata={"k": "v"})]
        snapshots = [copy.deepcopy(c) for c in candidates]
        fake = _FakeCrossEncoder(scores=[0.123])
        CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        for snap, after in zip(snapshots, candidates, strict=True):
            assert snap.score == after.score
            assert snap.metadata == after.metadata
            assert snap.chunk_text == after.chunk_text

    def test_output_score_is_rerank_score(self) -> None:
        candidates = [_r("a.py", 1, chunk_text="a")]
        fake = _FakeCrossEncoder(scores=[0.77])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        assert reranked[0].score == pytest.approx(0.77)

    def test_output_metadata_carries_rerank_score(self) -> None:
        """The returned metadata keeps the original keys plus ``rerank_score``."""
        candidates = [_r("a.py", 1, chunk_text="a", metadata={"source": "vector"})]
        fake = _FakeCrossEncoder(scores=[0.42])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank("q", candidates)
        metadata = reranked[0].metadata
        # Original metadata preserved.
        assert metadata["source"] == "vector"
        # New key added.
        assert metadata["rerank_score"] == pytest.approx(0.42)

    def test_other_fields_preserved(self) -> None:
        candidates = [
            _r(
                "auth.py",
                10,
                chunk_text="def login(): ...",
                metadata={"symbol_type": "function"},
            )
        ]
        fake = _FakeCrossEncoder(scores=[0.9])
        reranked = CrossEncoderReranker(fake, device="cpu").rerank(
            "auth flow", candidates
        )
        out = reranked[0]
        assert out.file_path == "auth.py"
        assert out.start_line == 10
        assert out.end_line == 10
        assert out.chunk_text == "def login(): ..."
        assert out.metadata["symbol_type"] == "function"


# ============================================================================
# End-to-end style: fuse then rerank (mirrors the issue spec snippet)
# ============================================================================


class TestFuseThenRerank:
    """Exercises the exact pipeline the issue spec's "How to test" shows."""

    def test_fused_then_reranked_pipeline(self) -> None:
        from reporag.retrieval.fusion import reciprocal_rank_fusion

        vector = [_r("auth.py", 10, chunk_text="def authenticate(user)")]
        bm25 = [
            _r("auth.py", 10, chunk_text="def authenticate(user)"),
            _r("session.py", 5, chunk_text="class Session"),
        ]
        graph = [
            _r("auth.py", 10, chunk_text="def authenticate(user)"),
            _r("router.py", 3, chunk_text="def route"),
        ]

        fused = reciprocal_rank_fusion([vector, bm25, graph], k=60)
        top_candidates = fused[:3]

        # Cross-encoder prefers the auth chunk over the others.
        scores_by_file = {
            "auth.py": 0.95,
            "session.py": 0.1,
            "router.py": 0.2,
        }
        # The fake scores must line up with the fused order/top_k slice.
        ordered_scores = [scores_by_file[r.file_path] for r in top_candidates]
        fake = _FakeCrossEncoder(scores=ordered_scores)
        reranker = CrossEncoderReranker(fake, device="cpu")
        reranked = reranker.rerank("auth flow", top_candidates)

        assert reranked[0].file_path == "auth.py"
        assert reranked[0].score == pytest.approx(0.95)
        # The fused score is overwritten with the cross-encoder score, but
        # the other metadata (file, line, chunk_text) survives intact.
        assert reranked[0].start_line == 10
        assert "def authenticate(user)" in reranked[0].chunk_text

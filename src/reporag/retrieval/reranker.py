"""Cross-encoder reranker.

Scores (query, chunk) pairs using a cross-encoder model. The cross-encoder
sees both query and document together, producing more accurate relevance
scores than bi-encoder retrieval.

Why
---
Vector search and BM25 both score a query against a chunk *independently* --
a bi-encoder embeds each in isolation and compares vectors; BM25 compares
token overlap. A cross-encoder instead feeds the query and chunk through the
same transformer *together*, so attention runs across both texts at once.
That produces substantially more accurate relevance judgments, at the cost
of not being usable as a first-pass retriever over an entire corpus (it
can't be precomputed or indexed -- every candidate has to be scored against
the query at query time). It is therefore only run over the small candidate
set that :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion` has
already narrowed down, not the full corpus.

Design
------
Mirrors the lazy-loading pattern established in
:class:`~reporag.embedding.code_embedder.CodeEmbedder`: constructing a
:class:`CrossEncoderReranker` does no network or GPU work. The
``sentence-transformers`` ``CrossEncoder`` (wrapping
``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default, see
``settings.reranker_model``) is loaded on first use, or a pre-built model
(or duck-typed fake) can be injected directly -- how the unit test suite
runs entirely offline. Candidates are scored in a single batched
``predict()`` call so reranking the ~20 candidates a typical RRF fusion
step hands off stays well under the 500ms acceptance budget.
"""

from __future__ import annotations

import logging
from typing import Any

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidate results by cross-encoder relevance to a query.

    Args:
        model_name: A ``sentence-transformers`` cross-encoder model id.
            Defaults to ``settings.reranker_model``
            (``cross-encoder/ms-marco-MiniLM-L-6-v2``).
        device: Torch device string passed to ``CrossEncoder`` on load
            (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``). ``None`` (default)
            lets ``sentence-transformers`` auto-select.
        model: A pre-built ``CrossEncoder``, or any duck-typed object
            exposing ``.predict(pairs, batch_size=..., show_progress_bar=...)
            -> Sequence[float]``, injected for tests or to reuse a model
            already loaded elsewhere. Skips ``CrossEncoder(model_name)``
            entirely when provided.
        batch_size: Candidate pairs scored per internal batch, passed
            straight through to ``CrossEncoder.predict``.
        max_length: Max combined token length for a ``(query, chunk)``
            pair; longer pairs are truncated by the underlying tokenizer.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        model: Any | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        """Configure the reranker; no model loading happens here."""
        self.model_name = model_name or settings.reranker_model
        self._device = device
        self._model = model
        self.batch_size = batch_size
        self.max_length = max_length

    @property
    def model(self) -> Any:
        """The underlying CrossEncoder, loaded lazily on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker '%s'", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self._device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Score and reorder *candidates* by cross-encoder relevance to *query*.

        - **Why it exists**: The final ranking step after RRF fusion --
          candidates have already survived a coarse first pass, and this
          produces the fine-grained ordering actually shown to the user.
        - **Algorithm**: Builds one ``(query, chunk_text)`` pair per
          candidate and scores all of them in a single batched
          ``model.predict()`` call (batched internally per
          :attr:`batch_size`), then sorts by the resulting score
          descending.
        - **Score bookkeeping**: The cross-encoder score becomes each
          result's new ``.score`` (overwriting whatever score it carried
          in -- RRF, vector, or BM25). That prior score is preserved in
          ``metadata["pre_rerank_score"]`` and the raw cross-encoder score
          is also duplicated under ``metadata["rerank_score"]``, so callers
          can inspect both without losing either.
        - **Edge cases**: An empty *candidates* returns ``[]`` without
          touching the model, so a reranker that's never used never pays
          the load cost. ``top_k=None`` (default) returns every candidate
          reordered rather than truncating.

        Args:
            query: The user's search query.
            candidates: Results to rerank, in any order (typically the
                output of
                :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion`).
            top_k: Maximum number of results to return. ``None`` returns
                all of *candidates*, reordered.

        Returns:
            *candidates*, reordered by cross-encoder score descending and
            truncated to *top_k* if given.

        Raises:
            ValueError: If ``top_k`` is given and is less than 1.
        """
        if not candidates:
            return []
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        pairs = [(query, c.chunk_text) for c in candidates]
        raw_scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        reranked: list[RetrievalResult] = []
        for candidate, raw_score in zip(candidates, raw_scores, strict=True):
            score = float(raw_score)
            metadata = dict(candidate.metadata)
            metadata["pre_rerank_score"] = candidate.score
            metadata["rerank_score"] = score
            reranked.append(
                RetrievalResult(
                    score=score,
                    file_path=candidate.file_path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    symbol_name=candidate.symbol_name,
                    chunk_text=candidate.chunk_text,
                    metadata=metadata,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

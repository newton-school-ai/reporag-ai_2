"""Cross-encoder reranker.

Scores (query, chunk) pairs using a cross-encoder model. The cross-encoder
sees both query and document together, producing more accurate relevance
scores than bi-encoder retrieval.
"""

import logging
from dataclasses import replace
from typing import Any

from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Scores (query, chunk) pairs using a cross-encoder model."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Get the cross-encoder model, loading it lazily if not provided."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        """Rerank candidates using the cross-encoder and return the top-k."""
        if not candidates:
            return []

        pairs = [(query, c.chunk_text) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores, strict=True):
            reranked.append(replace(candidate, score=float(score)))

        reranked.sort(
            key=lambda x: (-x.score, x.file_path, x.start_line or 0, x.end_line or 0)
        )
        return reranked[:top_k]

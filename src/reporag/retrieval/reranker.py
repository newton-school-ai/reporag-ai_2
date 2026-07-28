"""Cross-Encoder reranker.

Reranks candidate chunks relative to a query using a CrossEncoder model.
Produces final top-k sorted by rerank scores.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


def _resolve_device(preference: str = "auto") -> str:
    """Resolve the compute accelerator device.

    Auto order: CUDA -> MPS -> CPU. Falls back to CPU if preference is unavailable.
    """
    if preference in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if preference in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """Reranks candidate results using a Hugging Face Cross-Encoder sequence classification model.

    Features lazy model loading and accelerator device auto-detection.

    Args:
        model: Hugging Face model identifier or path. Defaults to settings.reranker_model.
        device: "auto", "cuda", "mps", or "cpu".
        max_length: Maximum sequence length.
        model_instance: Optional pre-loaded model instance (useful for testing).
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str = "auto",
        max_length: int = 512,
        model_instance: Any | None = None,
    ) -> None:
        self.model_name = model or settings.reranker_model
        self.device = _resolve_device(device)
        self.max_length = max_length
        self._model = model_instance
        self._loaded = model_instance is not None

    def _ensure_loaded(self) -> None:
        """Lazily downloads (if needed) and loads the Cross-Encoder model on the target device."""
        if self._loaded:
            return

        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading CrossEncoder model '%s' on %s", self.model_name, self.device
        )
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        self._loaded = True

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Score and re-rank candidates relative to the query using the Cross-Encoder.

        Args:
            query: The search query string.
            candidates: List of RetrievalResult candidates to rank.

        Returns:
            A new list of RetrievalResult objects sorted by rerank_score descending.
            Each returned result has its score and rerank_score attributes updated to
            the cross-encoder score.
        """
        if not candidates:
            return []

        self._ensure_loaded()
        assert self._model is not None

        # Build (query, text) pairs for the cross-encoder
        pairs = [(query, c.chunk_text) for c in candidates]

        # Get scores from the cross-encoder
        scores = self._model.predict(pairs)

        reranked_results = []
        for c, score in zip(candidates, scores, strict=False):
            score_val = float(score)

            # Build a new result object with updated scores
            copied = RetrievalResult(
                score=score_val,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
                chunk_text=c.chunk_text,
                metadata=c.metadata,
                rerank_score=score_val,
            )
            reranked_results.append(copied)

        # Sort descending by the cross-encoder score
        reranked_results.sort(key=lambda r: r.score, reverse=True)
        return reranked_results

"""Reciprocal Rank Fusion (RRF).

Merges ranked lists from vector, BM25, and graph retrieval into a single
fused ranking. Handles items present in some lists but not others.
"""

from __future__ import annotations

import logging

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists of RetrievalResult objects using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{m in M} 1 / (k + rank_m(d))
        where rank_m(d) is the 1-based rank of document d in ranked list m.

    Args:
        ranked_lists: A list of ranked lists of RetrievalResult objects.
        k: The constant parameter for RRF. If None, defaults to settings.rrf_constant.

    Returns:
        A new list of RetrievalResult objects sorted by their RRF score descending.

    Raises:
        ValueError: If k is less than 1.
    """
    if k is None:
        k = settings.rrf_constant

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    if not ranked_lists:
        return []

    # Map from unique chunk key (file_path, start_line, end_line) to cumulative RRF score
    scores: dict[tuple[str, int | None, int | None], float] = {}

    # Map from unique chunk key to the first/highest-ranked RetrievalResult
    representative_results: dict[
        tuple[str, int | None, int | None], RetrievalResult
    ] = {}

    for ranked_list in ranked_lists:
        for idx, item in enumerate(ranked_list):
            rank = idx + 1
            key = (item.file_path, item.start_line, item.end_line)

            # Sum reciprocal ranks
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            # Keep the highest-ranked (first encountered) candidate as representative
            if key not in representative_results:
                representative_results[key] = item

    # Construct the final merged and sorted results list
    fused_results = []
    for key, rrf_score in scores.items():
        orig_item = representative_results[key]
        fused_results.append(
            RetrievalResult(
                score=rrf_score,
                file_path=orig_item.file_path,
                start_line=orig_item.start_line,
                end_line=orig_item.end_line,
                symbol_name=orig_item.symbol_name,
                chunk_text=orig_item.chunk_text,
                metadata=orig_item.metadata,
                rerank_score=orig_item.rerank_score,
            )
        )

    # Sort descending by RRF score
    fused_results.sort(key=lambda r: r.score, reverse=True)
    return fused_results

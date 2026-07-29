"""Reciprocal Rank Fusion (RRF).

Merges ranked lists from vector, BM25, and graph retrieval into a single
fused ranking. Handles items present in some lists but not others.
"""

from dataclasses import replace
from typing import Any

from reporag.retrieval.vector_search import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]], k: int = 60
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion (RRF).

    Merges ranked lists from vector, BM25, and graph retrieval into a single
    fused ranking. Handles items present in some lists but not others.
    """
    scores: dict[tuple[str, int | None, int | None], float] = {}
    items: dict[tuple[str, int | None, int | None], RetrievalResult] = {}
    metadata_accumulator: dict[
        tuple[str, int | None, int | None], list[dict[str, Any]]
    ] = {}

    for lst in ranked_lists:
        seen_in_this_list: set[tuple[str, int | None, int | None]] = set()
        for rank, item in enumerate(lst, start=1):
            key = (item.file_path, item.start_line, item.end_line)
            # Only count a chunk's best rank from a single retriever
            if key in seen_in_this_list:
                continue
            seen_in_this_list.add(key)

            if key not in items:
                items[key] = item
                metadata_accumulator[key] = [dict(item.metadata)]
            else:
                metadata_accumulator[key].append(dict(item.metadata))

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = []
    for key, score in scores.items():
        merged_meta = {}
        for m in metadata_accumulator[key]:
            merged_meta.update(m)
        fused.append(replace(items[key], score=score, metadata=merged_meta))

    fused.sort(
        key=lambda x: (-x.score, x.file_path, x.start_line or 0, x.end_line or 0)
    )
    return fused

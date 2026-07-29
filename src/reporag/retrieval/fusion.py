"""Reciprocal Rank Fusion (RRF).

Merges ranked lists from vector, BM25, and graph retrieval into a single
fused ranking. Handles items present in some lists but not others.

Why
---
Each retrieval path (:mod:`~reporag.retrieval.vector_search`,
:mod:`~reporag.retrieval.bm25_search`, :mod:`~reporag.retrieval.graph_traversal`)
produces scores on incomparable scales -- cosine similarity, BM25's
unbounded tf-idf-derived score, and a synthetic hop-distance score. RRF
sidesteps the need to normalize or calibrate any of them: it only looks at
*rank position* within each list, which is directly comparable across
sources.

Algorithm
---------
For an item appearing at rank ``r`` (1-indexed, counting only its first
occurrence if a list somehow contains a duplicate -- see "Duplicate
handling" below) in a ranked list, its contribution to that list's RRF
score is ``1 / (k + r)``. An item's final RRF score is the sum of its
contributions across every list it appears in; lists it is absent from
simply contribute nothing (a missing item is not penalized beyond not
receiving credit). ``k`` dampens the influence of an item's exact top rank
-- a larger ``k`` flattens the curve so that ranking #1 vs #3 in one list
matters less; a smaller ``k`` sharpens the curve towards whichever list
ranks an item highest.

Duplicate handling
-------------------
Identity across lists is keyed on ``(file_path, start_line, end_line)``,
falling back to a ``symbol_name``/``chunk_text`` content hash when a result
has no location info at all (e.g. a malformed graph node), so two
genuinely different results never collide onto the same key just because
both are missing location data. Within a single list, if the same item
somehow appears more than once (a caller-side bug -- upstream retrievers
already dedupe their own output), only its first/best-ranked occurrence is
counted; later repeats are skipped rather than inflating that list's
contribution.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Identity key for "the same underlying chunk" across independently ranked
# lists. Mirrors the dedup key already used in
# VectorSearch.search (file_path, start_line, end_line) -- but see
# _result_key below for the fallback this alone can't handle.
ResultKey = tuple[str, int | None, int | None]

# Sentinel prefix marking a fallback (content-hash-based) key, so it can
# never collide with a real file_path a retriever might produce.
_NO_LOCATION_PREFIX = "__no_location__:"


def _result_key(result: RetrievalResult) -> ResultKey:
    """Identity key used to recognize the same chunk across ranked lists.

    Normally ``(file_path, start_line, end_line)`` is a reliable identity --
    it's the same convention ``VectorSearch.search`` already dedupes on.
    But a result with **no** location info at all (empty ``file_path`` and
    both line numbers ``None`` -- e.g. a malformed or synthetic graph node)
    would otherwise collapse every such result onto the same
    ``("", None, None)`` key, silently merging distinct items. In that case,
    fall back to hashing ``symbol_name`` + ``chunk_text`` instead, so two
    results are only treated as "the same item" when they actually share
    content, not just a missing location.
    """
    if result.file_path or result.start_line is not None or result.end_line is not None:
        return (result.file_path, result.start_line, result.end_line)

    content_hash = hashlib.sha256(
        f"{result.symbol_name or ''}|{result.chunk_text}".encode()
    ).hexdigest()
    return (f"{_NO_LOCATION_PREFIX}{content_hash}", None, None)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    *,
    k: int | None = None,
    weights: Sequence[float] | None = None,
    source_names: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """Fuse 2 or more ranked result lists into a single RRF-ranked list.

    - **Why it exists**: The merge point where vector search, BM25, and
      graph traversal (Issues 16-18) come together into one ranking that
      the cross-encoder reranker
      (:meth:`~reporag.retrieval.reranker.CrossEncoderReranker.rerank`)
      then refines further.
    - **Algorithm**: Walks each list in rank order and adds
      ``weight / (k + rank)`` to that item's running RRF score, keyed on
      ``(file_path, start_line, end_line)`` -- or a content hash of
      ``symbol_name``/``chunk_text`` when a result has no location info at
      all, so two distinct location-less results (e.g. malformed graph
      nodes) are never merged just because both are missing a location. An
      item's final score is the sum across every list it appeared in -- a
      list it's absent from simply contributes nothing, so no item is
      penalized for a source that didn't retrieve it. If the same item
      appears more than once *within* a single list, only its first
      (best-ranked) occurrence counts towards that list's contribution.
    - **Representative result**: When the same item appears in multiple
      lists, the merged ``RetrievalResult`` keeps the fields (file_path,
      chunk_text, symbol_name, ...) from whichever occurrence has the
      longest ``chunk_text``. Graph traversal only ever synthesizes a short
      signature+docstring snippet (see
      :mod:`~reporag.retrieval.graph_traversal`), so this favors a fuller
      vector/BM25 occurrence when one exists, without hardcoding source
      names or list order.
    - **Edge cases**: An empty *ranked_lists*, or lists that are all empty,
      returns ``[]``. A single list still gets fused (equivalent to a
      ``1 / (k + rank)`` re-scoring pass).

    Args:
        ranked_lists: 2 or more already-ranked lists of results, one per
            retrieval source, each sorted best-first. A single list is
            also accepted.
        k: The RRF constant. Larger values reduce the influence of an
            item's exact top rank. Defaults to ``settings.rrf_constant``.
        weights: Optional per-list multiplier (e.g. to trust vector search
            more than BM25). Must be the same length as *ranked_lists* if
            given; defaults to ``1.0`` for every list.
        source_names: Optional per-list label recorded in each merged
            result's ``metadata["rrf_source_scores"]`` /
            ``metadata["rrf_source_ranks"]`` (e.g.
            ``["vector", "bm25", "graph"]``), useful for debugging which
            sources contributed to a result. Defaults to ``"list_0"``,
            ``"list_1"``, ...

    Returns:
        Merged results sorted by RRF score descending. Each result's
        ``.score`` is overwritten with its RRF score; the original
        per-source scores and ranks are preserved in
        ``metadata["rrf_source_scores"]`` / ``metadata["rrf_source_ranks"]``
        (each a ``{source_name: value}`` dict).

    Raises:
        ValueError: If ``k <= 0``, or if *weights* or *source_names* is
            given with a length that doesn't match *ranked_lists*.
    """
    if k is None:
        k = settings.rrf_constant
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k!r}")

    n_lists = len(ranked_lists)

    if weights is not None and len(weights) != n_lists:
        raise ValueError(
            f"weights has {len(weights)} entries but ranked_lists has "
            f"{n_lists} lists"
        )
    if source_names is not None and len(source_names) != n_lists:
        raise ValueError(
            f"source_names has {len(source_names)} entries but "
            f"ranked_lists has {n_lists} lists"
        )

    resolved_weights = list(weights) if weights is not None else [1.0] * n_lists
    resolved_names = (
        list(source_names)
        if source_names is not None
        else [f"list_{i}" for i in range(n_lists)]
    )

    scores: dict[ResultKey, float] = {}
    representative: dict[ResultKey, RetrievalResult] = {}
    source_scores: dict[ResultKey, dict[str, float]] = {}
    source_ranks: dict[ResultKey, dict[str, int]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = resolved_weights[list_idx]
        name = resolved_names[list_idx]
        seen_in_this_list: set[ResultKey] = set()
        rank = 0
        for result in ranked_list:
            key = _result_key(result)
            if key in seen_in_this_list:
                # Defensive dedup: RRF assumes each input list is a clean
                # ranking with no repeats. If the same item appears twice
                # in one list (a caller-side bug -- upstream retrievers
                # like VectorSearch/BM25Search already dedupe their own
                # output), only its first/best-ranked occurrence counts,
                # so a duplicate can never inflate that list's contribution.
                logger.debug(
                    "reciprocal_rank_fusion: skipping duplicate item in "
                    "list %r (key=%r)",
                    name,
                    key,
                )
                continue
            seen_in_this_list.add(key)
            rank += 1

            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            source_scores.setdefault(key, {})[name] = result.score
            source_ranks.setdefault(key, {})[name] = rank

            current = representative.get(key)
            if current is None or len(result.chunk_text) > len(current.chunk_text):
                representative[key] = result

    fused: list[RetrievalResult] = []
    for key, rrf_score in scores.items():
        base = representative[key]
        merged_metadata = dict(base.metadata)
        merged_metadata["rrf_source_scores"] = source_scores[key]
        merged_metadata["rrf_source_ranks"] = source_ranks[key]
        fused.append(
            RetrievalResult(
                score=rrf_score,
                file_path=base.file_path,
                start_line=base.start_line,
                end_line=base.end_line,
                symbol_name=base.symbol_name,
                chunk_text=base.chunk_text,
                metadata=merged_metadata,
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused

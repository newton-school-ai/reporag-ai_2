"""Reciprocal Rank Fusion (RRF).

Merges ranked lists from vector, BM25, and graph retrieval into a single
fused ranking. Handles items present in some lists but not others.

Why RRF and not score normalisation?
-------------------------------------
Each retrieval method returns scores on a different, often incomparable
scale: vector cosine similarity lives in ``[-1, 1]``, BM25 idf-weighted
frequencies are unbounded above, and the graph retriever uses a synthetic
``1 / (distance + 1)`` pseudo-score (see
:mod:`reporag.retrieval.graph_traversal`). Score normalisation (e.g.
min-max) is fragile -- one outlier score distorts every other rank, and
zero/negative cosines break min-max outright. Reciprocal Rank Fusion sidesteps
calibration entirely by operating on **ranks** rather than scores: an item's
contribution from a list is ``1 / (k + rank)``, where *k* is a smoothing
constant and *rank* is its 1-based position in that list. The fused score is
the simple sum across all lists. This is rank-stable (a single new item can
never catapult an unrelated item) and tunable via one intuitive constant *k*
(larger *k* dampens the advantage of being ranked first, giving deeper lists
more equal weight).

Identity / dedup decision
-------------------------
Two results are treated as the *same document* when their
``(file_path, start_line, end_line)`` triple matches -- the same key
:mod:`reporag.retrieval.vector_search.VectorSearch` already uses for
cross-collection deduplication. The retriever's original ``RetrievalResult``
is preserved; only the ``score`` is replaced with the fused RRF score so the
returned objects stay drop-in compatible with everything that consumes a
``RetrievalResult`` (e.g. :class:`~reporag.retrieval.reranker.CrossEncoderReranker`,
Issue 19).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)


# Public sentinel for "no top_k cap" -- mirrors the issue spec's default
# usage ``fused[:20]`` where the caller slices after fusion. A huge int is
# used instead of ``None`` inside the implementation so a single slice at
# the end suffices.
_FUSE_SENTINEL_ALL = 10**9


def _result_key(result: RetrievalResult) -> tuple[str, int | None, int | None]:
    """Return the stable identity key used to deduplicate results across lists.

    Mirrors the key used by
    :class:`~reporag.retrieval.vector_search.VectorSearch` so a chunk that
    appears in both the vector and BM25 outputs (common for exact identifier
    queries) is fused once, not double-counted.
    """
    return (result.file_path, result.start_line, result.end_line)


def _validate_inputs(
    ranked_lists: Iterable[list[RetrievalResult]],
    k: int,
    top_k: int | None,
) -> tuple[list[list[RetrievalResult]], int, int]:
    """Validate fusion inputs and return normalised copies.

    Coerces a bare single list (the common typo of calling
    ``reciprocal_rank_fusion(vector_results, k=60)`` instead of
    ``reciprocal_rank_fusion([vector_results], k=60)``) into a single-element
    list of lists so callers get a fused/deduplicated ranking either way.
    """
    materialised = list(ranked_lists)

    # Support the friendly degenerate case: a single RetrievalResult list
    # passed directly. This is the most likely call-site mistake (the issue
    # spec shows ``[list_a, list_b, list_c]`` but a forgetful caller might
    # pass just ``list_a``), and fusion of one list is a well-defined no-op
    # dedup, so we accept it rather than raise.
    bare_list = bool(materialised) and isinstance(materialised[0], RetrievalResult)
    if bare_list:
        logger.debug(
            "reciprocal_rank_fusion received a single list instead of a list "
            "of lists; treating it as one ranked list."
        )
        materialised = [materialised]  # type: ignore[list-item]

    validated: list[list[RetrievalResult]] = []
    for i, lst in enumerate(materialised):
        if not isinstance(lst, list):
            raise TypeError(
                f"ranked_lists[{i}] must be a list of RetrievalResult, "
                f"got {type(lst).__name__}."
            )
        validated.append(lst)

    if k <= 0:
        raise ValueError(f"RRF constant k must be > 0, got {k!r}.")

    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k!r}.")
        effective_top_k = top_k
    else:
        # No explicit cap: keep every fused item (common for callers that
        # want to hand the full ranking to a downstream reranker, e.g.
        # ``fused[:20]`` in the issue spec).
        effective_top_k = _FUSE_SENTINEL_ALL

    return validated, k, effective_top_k


# Public sentinel for "no top_k cap" -- mirrors the issue spec's default
# usage ``fused[:20]`` where the caller slices after fusion.
_FUSE_SENTINEL_ALL = 10**9


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists into a single ranking via RRF.

    RRF score for an item is::

        score(item) = sum over each list L where item appears:
                          1 / (k + rank_in_L(item))

    Items present in only some lists contribute from those lists only
    (handling the "missing item" acceptance criterion naturally). Items
    absent from a list contribute nothing from that list -- they are *not*
    penalised with an implicit tail rank, which keeps the long tail of a
    single deep list from being buried by empty positions in shallower
    lists.

    Args:
        ranked_lists: An iterable of ranked result lists. Each inner list
            must already be sorted by the producing retriever's native score
            in **descending** order: rank 1 is ``results[0]``, rank 2 is
            ``results[1]``, and so on. A bare ``list[RetrievalResult]``
            (missing the outer list wrapper) is gracefully coerced into a
            single-list fusion.
        k: The RRF smoothing constant. Larger *k* dampens the advantage of
            being ranked first and gives deeper lists more equal weight.
            Defaults to ``60`` (the value in the original RRF paper and
            ``settings.rrf_constant``). Must be ``>= 1`` (a non-positive *k*
            would make ``1 / (k + rank)`` undefined or negative for rank 1).
        top_k: If given, return at most this many fused results. ``None``
            (the default) returns the **full** fused ranking so a downstream
            cross-encoder reranker can slice ``fused[:20]`` as in the issue
            spec. Must be ``>= 1`` when provided.

    Returns:
        A new list of :class:`~reporag.retrieval.vector_search.RetrievalResult`
        objects sorted by fused RRF score descending, ties broken by the
        earliest (best) rank seen across all input lists so the ordering is
        deterministic. Each returned result's ``score`` is the RRF score;
        all other fields (``file_path``, ``start_line``, ``chunk_text``,
        ``metadata``, ...) are copied from the *first* list that surfaced
        the item, so no provenance is lost. The default ``top_k=None``
        returns every item that appeared in at least one input list.

    Raises:
        ValueError: If *k* is ``<= 0`` or *top_k* (when given) is ``< 1``.
        TypeError: If an element of *ranked_lists* is not a list.

    Example:
        >>> fused = reciprocal_rank_fusion(
        ...     [vector_results, bm25_results, graph_results], k=60
        ... )
        >>> reranked = reranker.rerank(query="auth flow", candidates=fused[:20])
    """
    lists, k, effective_top_k = _validate_inputs(ranked_lists, k, top_k)

    # rrf_score[key] -> running sum of 1 / (k + rank).  best_rank[key] is the
    # smallest (1-based) rank the item achieved across all lists, used purely
    # as a deterministic tiebreaker.  first_seen[key] is the 1-based order in
    # which the item was first encountered across all lists; the deterministic
    # sort key ties out a float-equal-score + equal-best-rank by falling back
    # to first-seen order so reruns of the same call produce byte-identical
    # output.  representative[key] is the first RetrievalResult we saw for
    # that key; its metadata is the one returned.
    rrf_score: dict[tuple[str, int | None, int | None], float] = {}
    best_rank: dict[tuple[str, int | None, int | None], int] = {}
    first_seen: dict[tuple[str, int | None, int | None], int] = {}
    representative: dict[tuple[str, int | None, int | None], RetrievalResult] = {}

    discovery_counter = 0

    for lst in lists:
        # Within a single list the same result key may appear more than
        # once (pathological but possible if a retriever emits duplicates).
        # We count each key at most once per list, at its *best* (earliest)
        # rank, mirroring the dedup semantics
        # :class:`~reporag.retrieval.vector_search.VectorSearch` already
        # applies across collections.
        seen_in_list: set[tuple[str, int | None, int | None]] = set()
        for position, result in enumerate(lst):
            key = _result_key(result)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            rank = position + 1  # ranks are 1-based.
            contribution = 1.0 / (k + rank)
            rrf_score[key] = rrf_score.get(key, 0.0) + contribution
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
                representative[key] = result
            if key not in first_seen:
                discovery_counter += 1
                first_seen[key] = discovery_counter

    # Build the fused output: deterministic sort by (rrf_score desc,
    # best_rank asc, first_seen asc).  The integer tiebreakers keep fused
    # output stable across runs and avoid the nondeterminism of a
    # float-only tiebreak (two items with equal score AND equal best rank
    # fall back to the order they were first discovered).
    fused_results: list[RetrievalResult] = []
    for key in sorted(
        representative.keys(),
        key=lambda ky: (
            -rrf_score[ky],
            best_rank[ky],
            first_seen[ky],
        ),
    ):
        original = representative[key]
        # Copy to avoid mutating the caller's RetrievalResult objects (some
        # callers reuse the same result list across calls).
        fused_results.append(
            RetrievalResult(
                score=rrf_score[key],
                file_path=original.file_path,
                start_line=original.start_line,
                end_line=original.end_line,
                symbol_name=original.symbol_name,
                chunk_text=original.chunk_text,
                metadata=original.metadata,
            )
        )

    return fused_results[:effective_top_k]

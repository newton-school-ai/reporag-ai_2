"""Vector semantic search with configurable top-k.

Queries Qdrant with embedded query vectors and returns top-k results ranked
by cosine similarity.  Supports filtering by language, file-path glob, symbol
type, and repository ID.

Why two collections?
--------------------
RepoRAG maintains *two* Qdrant collections:

* ``reporag_code`` -- code chunks embedded by
  :class:`~reporag.embedding.code_embedder.CodeEmbedder` (768-dim,
  CodeBERT / UniXcoder).  These capture *programming-language semantics*:
  variable naming patterns, control-flow idioms, API-usage similarity.
* ``reporag_docs`` -- documentation embeddings from
  :class:`~reporag.embedding.doc_embedder.DocEmbedder` (384-dim,
  all-MiniLM-L6-v2).  These capture *natural-language intent*: a query
  phrased in English ("how does auth work?") matches documentation even
  when the code uses different identifiers.

Each collection requires its *own* embedder to produce a query vector of the
correct dimensionality.  The two ranked lists are merged by score and
deduplicated on ``(file_path, start_line)`` so the caller receives a single,
unified ranking.

Design
------
:class:`VectorSearch` is the public entry point.  It mirrors the
dependency-injection and lazy-loading patterns established by
:class:`~reporag.embedding.index_builder.HybridIndexBuilder`:

* **Lazy Qdrant client** -- constructed on first access from
  ``settings.qdrant_url`` when not injected, so construction is cheap and
  tests stay network-free.
* **Duck-typed embedders** -- any object with an
  ``embed(str) -> ndarray`` method works; the module never imports
  ``CodeEmbedder`` or ``DocEmbedder`` at runtime.
* **Graceful degradation** -- Qdrant errors are caught, logged, and return
  empty results rather than crashing the caller.  A missing collection is
  treated the same way.
* **Path-glob post-filter** -- Qdrant has no native glob support, so
  ``path_glob`` filtering uses :func:`fnmatch.fnmatch` in Python after
  Qdrant returns results.  When a glob is active the per-collection fetch
  limit is inflated by an over-fetch factor so post-filtering still yields
  enough results.
* **Score-floor filtering** -- results below ``min_score`` are dropped
  before merging, preventing low-confidence noise from reaching the caller.
* **Observability** -- :meth:`VectorSearch.search_with_stats` returns a
  diagnostics dict alongside results (latency, per-collection counts,
  filter/dedup discard counts) so callers can log or expose metrics.

The shared :class:`RetrievalResult` dataclass is the output contract for
*every* M5 retriever (Issues 16--19).  Its ``source`` field lets downstream
fusion (Issue 19) know which retriever produced each result, and
``point_id`` lets it cross-reference results from different retrievers that
share the same underlying chunk.

Usage
-----
::

    from reporag.embedding.code_embedder import CodeEmbedder
    from reporag.embedding.doc_embedder import DocEmbedder
    from reporag.retrieval.vector_search import VectorSearch

    vs = VectorSearch(
        code_embedder=CodeEmbedder(),
        doc_embedder=DocEmbedder(),
    )
    results = vs.search("How does authentication work?", top_k=10)
    for r in results:
        print(f"{r.score:.3f} | {r.file_path}:{r.start_line} | {r.symbol_name}")
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from reporag.config import settings

logger = logging.getLogger(__name__)

# Over-fetch multiplier when path_glob is active.  Qdrant cannot filter by
# glob natively, so we fetch extra results and filter in Python.  3x is a
# reasonable trade-off: even if two-thirds of results are discarded the
# caller still gets enough.
_GLOB_OVERFETCH_FACTOR = 3

# Default minimum cosine-similarity score.  Results below this floor are
# almost certainly noise and are dropped before merging.  Set to 0.0 to
# disable (accept everything Qdrant returns).
_DEFAULT_MIN_SCORE = 0.0


# ---------------------------------------------------------------------------
# Shared retrieval result (output contract for Issues 16, 17, 18, 19)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """One retrieved chunk or document with its relevance score.

    This is the common output type for *every* M5 retriever -- vector,
    BM25, and graph -- so that the fusion layer (Issue 19) can merge
    heterogeneous ranked lists without knowing which retriever produced
    each item.

    Attributes:
        score:       Retriever-specific relevance score (cosine similarity
                     for vector search, BM25 score for keyword search, etc.).
                     Higher is always better.
        file_path:   Repository-relative path of the source file.
        start_line:  First line of the chunk (1-based).
        end_line:    Last line of the chunk (1-based, inclusive).
        symbol_name: Qualified name of the enclosing symbol, if known
                     (e.g. ``"reporag.api.main.health"``).
        chunk_text:  The raw text of the chunk or documentation passage.
        source:      Which retriever produced this result.  One of
                     ``"vector_code"``, ``"vector_doc"``, ``"bm25"``,
                     ``"graph"``.
        point_id:    The Qdrant point ID (or BM25 doc_id).  Used by
                     downstream fusion to cross-reference results from
                     different retrievers that share the same underlying
                     chunk.
        metadata:    Full payload dict from the underlying store for
                     downstream use (e.g. ``language``, ``chunk_kind``).
    """

    score: float
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    chunk_text: str = ""
    source: str = "vector_code"
    point_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> tuple[str, int]:
        """Identity key for deduplication during merge.

        Two results covering the same file location are considered
        duplicates even if they came from different collections (e.g. a
        docstring and the code it documents living on the same line).
        """
        return (self.file_path, self.start_line)

    def __repr__(self) -> str:  # pragma: no cover
        sym = f" ({self.symbol_name})" if self.symbol_name else ""
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"{self.file_path}:{self.start_line}-{self.end_line}{sym}, "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Qdrant filter builders
# ---------------------------------------------------------------------------


def build_qdrant_filter(
    *,
    language: str | None = None,
    symbol_type: str | None = None,
    repo_id: str | None = None,
) -> Any:
    """Build a Qdrant ``Filter`` from user-facing search parameters.

    Only non-``None`` parameters generate a ``FieldCondition``.  If every
    parameter is ``None`` the function returns ``None`` (no filter), which
    Qdrant interprets as "match everything".

    ``path_glob`` is intentionally *not* handled here -- Qdrant has no
    native glob/wildcard support on string payloads, so path filtering is
    applied as a Python-side post-filter after Qdrant returns results.

    Args:
        language:    Exact match on the ``language`` payload field
                     (e.g. ``"python"``).
        symbol_type: Exact match on ``symbol_type`` (code collection only).
        repo_id:     Exact match on ``repo_id``.

    Returns:
        A :class:`qdrant_client.models.Filter` with a ``must`` clause, or
        ``None`` when no conditions are specified.
    """
    from qdrant_client import models

    conditions: list[models.FieldCondition] = []

    if language is not None:
        conditions.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=language),
            )
        )
    if symbol_type is not None:
        conditions.append(
            models.FieldCondition(
                key="symbol_type",
                match=models.MatchValue(value=symbol_type),
            )
        )
    if repo_id is not None:
        conditions.append(
            models.FieldCondition(
                key="repo_id",
                match=models.MatchValue(value=repo_id),
            )
        )

    return models.Filter(must=conditions) if conditions else None


def _build_doc_filter(
    *,
    language: str | None = None,
    doc_type: str | None = None,
    repo_id: str | None = None,
) -> Any:
    """Build a Qdrant ``Filter`` for the documentation collection.

    The docs collection uses ``doc_type`` (docstring / comment / readme)
    instead of ``symbol_type``, so it gets its own builder.
    """
    from qdrant_client import models

    conditions: list[models.FieldCondition] = []

    if language is not None:
        conditions.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=language),
            )
        )
    if doc_type is not None:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type),
            )
        )
    if repo_id is not None:
        conditions.append(
            models.FieldCondition(
                key="repo_id",
                match=models.MatchValue(value=repo_id),
            )
        )

    return models.Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# ScoredPoint -> RetrievalResult converters
# ---------------------------------------------------------------------------


def _code_point_to_result(point: Any) -> RetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` from the code collection.

    The code collection payload schema (defined by
    :meth:`~reporag.embedding.index_builder.HybridIndexBuilder._build_code_payload`)
    carries ``content``, ``symbol``, ``symbol_type``, ``file_path``,
    ``start_line``, ``end_line``, etc.
    """
    payload = point.payload or {}
    return RetrievalResult(
        score=float(point.score),
        file_path=payload.get("file_path", ""),
        start_line=payload.get("start_line", 0),
        end_line=payload.get("end_line", 0),
        symbol_name=payload.get("symbol") or payload.get("qualified_name"),
        chunk_text=payload.get("content", ""),
        source="vector_code",
        point_id=str(point.id) if point.id is not None else None,
        metadata=dict(payload),
    )


def _doc_point_to_result(point: Any) -> RetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` from the documentation collection.

    The docs collection payload schema (defined by
    :meth:`~reporag.embedding.index_builder.HybridIndexBuilder._build_doc_payload`)
    carries ``text``, ``symbol_id``, ``doc_type``, ``file_path``,
    ``start_line``, ``end_line``, etc.
    """
    payload = point.payload or {}
    return RetrievalResult(
        score=float(point.score),
        file_path=payload.get("file_path", ""),
        start_line=payload.get("start_line", 0),
        end_line=payload.get("end_line", 0),
        symbol_name=payload.get("symbol_id"),
        chunk_text=payload.get("text", ""),
        source="vector_doc",
        point_id=str(point.id) if point.id is not None else None,
        metadata=dict(payload),
    )


# ---------------------------------------------------------------------------
# Merge + dedup
# ---------------------------------------------------------------------------


def _merge_results(
    *result_lists: list[RetrievalResult],
    top_k: int,
    path_glob: str | None = None,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[RetrievalResult]:
    """Merge multiple ranked lists into a single deduplicated ranking.

    Processing pipeline:

    1. **Concatenate** all input lists.
    2. **Score floor** -- drop results with ``score < min_score``.
    3. **Sort** by score descending (stable sort preserves intra-list order
       for ties).
    4. **Deduplicate** on ``(file_path, start_line)``, keeping the
       highest-scoring variant.  A docstring and its parent function may
       share the same start line; only the more relevant one survives.
    5. **Path-glob filter** -- if ``path_glob`` is set, drop results whose
       ``file_path`` does not match (via :func:`fnmatch.fnmatch`).
    6. **Truncate** to ``top_k``.
    """
    combined: list[RetrievalResult] = []
    for result_list in result_lists:
        combined.extend(result_list)

    # Score floor
    if min_score > 0:
        combined = [r for r in combined if r.score >= min_score]

    # Sort by score descending (stable sort preserves order for ties)
    combined.sort(key=lambda r: r.score, reverse=True)

    # Deduplicate by (file_path, start_line), keeping highest score
    seen: set[tuple[str, int]] = set()
    deduped: list[RetrievalResult] = []
    for result in combined:
        key = result.dedup_key
        if key not in seen:
            seen.add(key)
            deduped.append(result)

    # Path-glob post-filter
    if path_glob:
        deduped = [r for r in deduped if fnmatch.fnmatch(r.file_path, path_glob)]

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# VectorSearch
# ---------------------------------------------------------------------------


class VectorSearch:
    """Semantic vector search across Qdrant code and doc collections.

    Embeds the query with the appropriate model for each collection,
    searches both, and returns a merged, deduplicated list of
    :class:`RetrievalResult` objects sorted by cosine similarity.

    The constructor accepts pre-built embedder and client objects so that
    tests can inject lightweight fakes without touching the network or
    loading ML models.

    Args:
        code_embedder: A :class:`~reporag.embedding.code_embedder.CodeEmbedder`
            (or any object with an ``embed(str) -> ndarray`` method producing
            768-dim vectors).
        doc_embedder:  A :class:`~reporag.embedding.doc_embedder.DocEmbedder`
            (or any object with an ``embed(str) -> ndarray`` method producing
            384-dim vectors).
        qdrant_client: A ``QdrantClient`` instance (or injectable fake).
            Constructed lazily from ``settings.qdrant_url`` if not provided.
        collection_code: Override ``settings.qdrant_collection_code``.
        collection_docs: Override ``settings.qdrant_collection_docs``.
        min_score: Drop results below this cosine-similarity floor.
            Defaults to 0.0 (accept everything).
    """

    def __init__(
        self,
        code_embedder: Any,
        doc_embedder: Any,
        qdrant_client: Any | None = None,
        *,
        collection_code: str | None = None,
        collection_docs: str | None = None,
        min_score: float = _DEFAULT_MIN_SCORE,
    ) -> None:
        self.code_embedder = code_embedder
        self.doc_embedder = doc_embedder
        self._client = qdrant_client
        self.collection_code = collection_code or settings.qdrant_collection_code
        self.collection_docs = collection_docs or settings.qdrant_collection_docs
        self.min_score = min_score

    # ------------------------------------------------------------------
    # Lazy Qdrant client (mirrors HybridIndexBuilder)
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The Qdrant client, constructed on first access if not injected."""
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
        path_glob: str | None = None,
        symbol_type: str | None = None,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search both code and doc collections and merge results.

        Embeds the query with both the code and doc embedders, searches
        each collection independently, then merges, deduplicates, and
        optionally filters the combined ranking.

        Args:
            query:       Natural-language or code query string.
            top_k:       Maximum results to return.  Defaults to
                         ``settings.vector_search_top_k``.
            language:    Filter to a specific language (e.g. ``"python"``).
            path_glob:   Filter by file path glob (e.g. ``"src/**/*.py"``).
                         Applied as a post-filter via :func:`fnmatch.fnmatch`.
            symbol_type: Filter by symbol type (e.g. ``"function"``,
                         ``"class"``).  Only applies to the code collection.
            repo_id:     Filter to a specific repository.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects sorted by
            cosine similarity (descending).
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k
        t0 = time.monotonic()

        # Over-fetch when glob filtering is active so post-filtering still
        # yields enough results.
        fetch_k = (
            effective_top_k * _GLOB_OVERFETCH_FACTOR if path_glob else effective_top_k
        )

        code_results = self.search_code(
            query,
            top_k=fetch_k,
            language=language,
            symbol_type=symbol_type,
            repo_id=repo_id,
        )
        doc_results = self.search_docs(
            query,
            top_k=fetch_k,
            language=language,
            doc_type=None,
            repo_id=repo_id,
        )

        merged = _merge_results(
            code_results,
            doc_results,
            top_k=effective_top_k,
            path_glob=path_glob,
            min_score=self.min_score,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "VectorSearch: query=%r top_k=%d returned=%d code=%d docs=%d "
            "elapsed=%.1fms",
            query[:80],
            effective_top_k,
            len(merged),
            len(code_results),
            len(doc_results),
            elapsed_ms,
        )

        return merged

    def search_with_stats(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
        path_glob: str | None = None,
        symbol_type: str | None = None,
        repo_id: str | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Like :meth:`search`, but also returns a diagnostics dict.

        The second element of the returned tuple contains:

        * ``latency_ms`` -- total wall-clock time for the search.
        * ``code_count`` -- raw results from the code collection.
        * ``doc_count``  -- raw results from the doc collection.
        * ``merged_count`` -- final count after merge + dedup + filter.
        * ``query`` -- the query string (truncated to 120 chars).
        * ``top_k`` -- the effective top_k used.
        * ``filters`` -- which filters were active.
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k
        t0 = time.monotonic()

        fetch_k = (
            effective_top_k * _GLOB_OVERFETCH_FACTOR if path_glob else effective_top_k
        )

        code_results = self.search_code(
            query,
            top_k=fetch_k,
            language=language,
            symbol_type=symbol_type,
            repo_id=repo_id,
        )
        doc_results = self.search_docs(
            query,
            top_k=fetch_k,
            language=language,
            doc_type=None,
            repo_id=repo_id,
        )

        merged = _merge_results(
            code_results,
            doc_results,
            top_k=effective_top_k,
            path_glob=path_glob,
            min_score=self.min_score,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        stats: dict[str, Any] = {
            "latency_ms": round(elapsed_ms, 2),
            "code_count": len(code_results),
            "doc_count": len(doc_results),
            "merged_count": len(merged),
            "query": query[:120],
            "top_k": effective_top_k,
            "filters": {
                "language": language,
                "path_glob": path_glob,
                "symbol_type": symbol_type,
                "repo_id": repo_id,
            },
        }

        logger.info(
            "VectorSearch(stats): query=%r top_k=%d returned=%d "
            "code=%d docs=%d elapsed=%.1fms",
            query[:80],
            effective_top_k,
            len(merged),
            len(code_results),
            len(doc_results),
            elapsed_ms,
        )

        return merged, stats

    def search_code(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
        symbol_type: str | None = None,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search only the code embeddings collection.

        Uses the injected code embedder to produce a query vector of the
        correct dimensionality (768-dim for CodeBERT / UniXcoder), then
        queries the ``reporag_code`` Qdrant collection.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects from the code
            collection, sorted by cosine similarity.
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k

        query_vector = self.code_embedder.embed(query).tolist()
        query_filter = build_qdrant_filter(
            language=language,
            symbol_type=symbol_type,
            repo_id=repo_id,
        )

        try:
            scored_points = self.client.search(
                collection_name=self.collection_code,
                query_vector=query_vector,
                limit=effective_top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception:
            logger.exception(
                "Qdrant search failed on collection %r", self.collection_code
            )
            return []

        return [_code_point_to_result(p) for p in scored_points]

    def search_docs(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
        doc_type: str | None = None,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search only the documentation embeddings collection.

        Uses the injected doc embedder to produce a query vector of the
        correct dimensionality (384-dim for all-MiniLM-L6-v2), then
        queries the ``reporag_docs`` Qdrant collection.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects from the doc
            collection, sorted by cosine similarity.
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k

        query_vector = self.doc_embedder.embed(query).tolist()
        query_filter = _build_doc_filter(
            language=language,
            doc_type=doc_type,
            repo_id=repo_id,
        )

        try:
            scored_points = self.client.search(
                collection_name=self.collection_docs,
                query_vector=query_vector,
                limit=effective_top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception:
            logger.exception(
                "Qdrant search failed on collection %r", self.collection_docs
            )
            return []

        return [_doc_point_to_result(p) for p in scored_points]

    def health_check(self) -> dict[str, Any]:
        """Verify that both Qdrant collections exist and are queryable.

        Returns a dict with per-collection status (``True`` / ``False``)
        and an overall ``healthy`` flag.  Useful for readiness probes
        and startup diagnostics.
        """
        results: dict[str, Any] = {"healthy": True}

        for label, collection_name in (
            ("code_collection", self.collection_code),
            ("docs_collection", self.collection_docs),
        ):
            try:
                exists = False
                if hasattr(self.client, "collection_exists"):
                    exists = bool(self.client.collection_exists(collection_name))
                else:
                    existing = self.client.get_collections()
                    exists = any(
                        c.name == collection_name for c in existing.collections
                    )
                results[label] = exists
                if not exists:
                    results["healthy"] = False
            except Exception:
                logger.exception(
                    "Health check failed for collection %r", collection_name
                )
                results[label] = False
                results["healthy"] = False

        return results

    def __repr__(self) -> str:
        return (
            f"VectorSearch(code={self.collection_code!r}, "
            f"docs={self.collection_docs!r}, "
            f"min_score={self.min_score})"
        )

"""Vector semantic search.

Queries Qdrant with an embedded query vector. Returns top-k results with
scores and metadata payloads. Supports filtering by language, file path
glob, symbol type, and repository ID.

Design
------
:class:`VectorSearch` searches **two** Qdrant collections:

* ``reporag_code`` -- code chunks embedded by
  :class:`~reporag.embedding.code_embedder.CodeEmbedder` (768-dim,
  CodeBERT / UniXcoder).
* ``reporag_docs`` -- documentation embeddings from
  :class:`~reporag.embedding.doc_embedder.DocEmbedder` (384-dim,
  all-MiniLM-L6-v2).

Each collection requires its *own* embedder to produce a query vector of the
correct dimensionality.  The two ranked lists are merged by cosine-similarity
score and deduplicated on ``(file_path, start_line)`` so the caller sees a
single, unified ranking.

The shared :class:`RetrievalResult` dataclass is the output contract for all
M5 retrievers (Issues 16--19).  Its ``source`` field lets downstream fusion
(Issue 19) know which retriever produced each result.

Usage::

    from reporag.embedding.code_embedder import CodeEmbedder
    from reporag.embedding.doc_embedder import DocEmbedder
    from reporag.retrieval.vector_search import VectorSearch

    vs = VectorSearch(
        code_embedder=CodeEmbedder(),
        doc_embedder=DocEmbedder(),
    )
    results = vs.search("How does authentication work?", top_k=10)
    for r in results:
        print(f"{r.file_path}:{r.start_line}  score={r.score:.3f}")
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from reporag.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared retrieval result (used by Issues 16, 17, 18, 19)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """One retrieved chunk / document with its relevance score.

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
        source:      Which retriever produced this result. One of
                     ``"vector_code"``, ``"vector_doc"``, ``"bm25"``,
                     ``"graph"``.
        point_id:    The Qdrant point ID (or BM25 doc_id). Used by
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

    def __repr__(self) -> str:  # pragma: no cover
        sym = f" ({self.symbol_name})" if self.symbol_name else ""
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"{self.file_path}:{self.start_line}-{self.end_line}{sym}, "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Qdrant filter builder
# ---------------------------------------------------------------------------


def build_filter(
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
    native glob support, so path filtering is done as a post-filter in
    Python via :func:`fnmatch.fnmatch` after Qdrant returns results.

    Args:
        language:    Exact match on the ``language`` payload field
                     (e.g. ``"python"``).
        symbol_type: Exact match on ``symbol_type`` (code collection).
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
    """Build a Qdrant filter for the *docs* collection.

    The docs collection uses ``doc_type`` instead of ``symbol_type``, so
    this is a thin variant of :func:`build_filter`.
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
# Scored-point to RetrievalResult conversion
# ---------------------------------------------------------------------------


def _code_point_to_result(point: Any) -> RetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` from the *code* collection."""
    payload: dict[str, Any] = point.payload or {}
    return RetrievalResult(
        score=float(point.score),
        file_path=payload.get("file_path", ""),
        start_line=int(payload.get("start_line", 0)),
        end_line=int(payload.get("end_line", 0)),
        symbol_name=payload.get("qualified_name") or payload.get("symbol"),
        chunk_text=payload.get("content", ""),
        source="vector_code",
        point_id=str(point.id) if point.id is not None else None,
        metadata=payload,
    )


def _doc_point_to_result(point: Any) -> RetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` from the *docs* collection."""
    payload: dict[str, Any] = point.payload or {}
    return RetrievalResult(
        score=float(point.score),
        file_path=payload.get("file_path", ""),
        start_line=int(payload.get("start_line", 0)),
        end_line=int(payload.get("end_line", 0)),
        symbol_name=payload.get("symbol_id"),
        chunk_text=payload.get("text", ""),
        source="vector_doc",
        point_id=str(point.id) if point.id is not None else None,
        metadata=payload,
    )


# ---------------------------------------------------------------------------
# Merge + deduplication
# ---------------------------------------------------------------------------


def _merge_results(
    *result_lists: list[RetrievalResult],
    top_k: int,
    path_glob: str | None = None,
) -> list[RetrievalResult]:
    """Merge multiple ranked lists, deduplicate, and return the top-k.

    Deduplication key is ``(file_path, start_line)`` -- when the same
    location appears in both the code and doc results, only the
    higher-scored entry survives.

    If *path_glob* is given, results whose ``file_path`` does not match
    the pattern (via :func:`fnmatch.fnmatch`) are discarded **before**
    ranking.  This provides true glob semantics (``src/**/*.py``) that
    Qdrant cannot express natively.
    """
    combined: list[RetrievalResult] = []
    for result_list in result_lists:
        combined.extend(result_list)

    # Post-filter by path glob if requested
    if path_glob:
        combined = [r for r in combined if fnmatch.fnmatch(r.file_path, path_glob)]

    # Sort by score descending
    combined.sort(key=lambda r: r.score, reverse=True)

    # Deduplicate by (file_path, start_line), keeping the first (highest score)
    seen: set[tuple[str, int]] = set()
    deduped: list[RetrievalResult] = []
    for result in combined:
        key = (result.file_path, result.start_line)
        if key not in seen:
            seen.add(key)
            deduped.append(result)

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# VectorSearch
# ---------------------------------------------------------------------------


class VectorSearch:
    """Semantic vector search across Qdrant code and doc collections.

    Embeds the query with the appropriate model for each collection,
    searches both, and returns a merged, deduplicated list of
    :class:`RetrievalResult` objects sorted by cosine similarity.

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
    """

    def __init__(
        self,
        code_embedder: Any,
        doc_embedder: Any,
        qdrant_client: Any | None = None,
        *,
        collection_code: str | None = None,
        collection_docs: str | None = None,
    ) -> None:
        self.code_embedder = code_embedder
        self.doc_embedder = doc_embedder
        self._client = qdrant_client
        self.collection_code = collection_code or settings.qdrant_collection_code
        self.collection_docs = collection_docs or settings.qdrant_collection_docs

    # ------------------------------------------------------------------
    # Lazy Qdrant client
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

        Args:
            query:       Natural-language or code query string.
            top_k:       Maximum results to return.  Defaults to
                         ``settings.vector_search_top_k``.
            language:    Filter to a specific language (e.g. ``"python"``).
            path_glob:   Filter by file path glob (e.g. ``"src/**/*.py"``).
                         Applied as a post-filter via :func:`fnmatch.fnmatch`.
            symbol_type: Filter by symbol type (e.g. ``"function"``,
                         ``"class"``).
            repo_id:     Filter to a specific repository.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects sorted by
            cosine similarity (descending).
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k
        t0 = time.monotonic()

        # Fetch more than top_k from each collection when glob filtering
        # is active, since some results will be discarded post-filter.
        fetch_k = effective_top_k * 3 if path_glob else effective_top_k

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

        Uses :class:`~reporag.embedding.code_embedder.CodeEmbedder` to
        produce a 768-dim query vector.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects from the code
            collection, sorted by cosine similarity.
        """
        effective_top_k = top_k if top_k is not None else settings.vector_search_top_k

        query_vector = self.code_embedder.embed(query).tolist()
        query_filter = build_filter(
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

        Uses :class:`~reporag.embedding.doc_embedder.DocEmbedder` to
        produce a 384-dim query vector.

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

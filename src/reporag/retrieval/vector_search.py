"""Vector semantic search.

Queries Qdrant with embedded query vectors and returns top-k results with
scores and metadata payloads. Supports filtering by language, file path,
symbol type, and repo_id. Searches both the code and doc collections
independently, then merges and deduplicates the results.

Why two collections?
    Issue #15 (HybridIndexBuilder) stores code chunks in ``reporag_code``
    (768-dim UniXcoder vectors) and documentation in ``reporag_docs``
    (384-dim MiniLM vectors).  Because these are *different embedding spaces*,
    each query must be embedded separately before searching its collection.
    Sending a 768-dim code vector to the 384-dim doc collection (or vice versa)
    is a hard Qdrant error, not a quality issue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models

from reporag.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols -- lightweight structural types, no heavy runtime imports
# ---------------------------------------------------------------------------


class EmbeddingVector(Protocol):
    """Anything that can convert itself to a plain Python list of floats.

    Both ``np.ndarray`` (returned by CodeEmbedder and DocEmbedder) and any
    test stub that returns a plain list satisfy this protocol.  Declaring it
    here means this module has no hard runtime dependency on NumPy.
    """

    def tolist(self) -> list[float]: ...


class Embedder(Protocol):
    """Structural type for any model that can embed a query string.

    Both :class:`~reporag.embedding.code_embedder.CodeEmbedder` and
    :class:`~reporag.embedding.doc_embedder.DocEmbedder` satisfy this
    protocol via their ``embed(text) -> np.ndarray`` method.
    """

    def embed(self, text: str) -> EmbeddingVector: ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Immutable value object representing one ranked search hit.

    Fields match the acceptance criteria for Issue #16 and are compatible
    with the merge/re-rank step in Issue #19 (fusion).

    Attributes:
        score:      Cosine similarity score returned by Qdrant (0-1).
        file_path:  Repository-relative path to the source file.
        lines:      ``(start_line, end_line)`` span of the chunk, 1-indexed.
        symbol:     Qualified symbol name if present (``None`` for module-level
                    or prose-only chunks).
        chunk_text: The raw text stored in the payload (``content`` for code
                    chunks, ``text`` for doc chunks).
        point_id:   Stable Qdrant point UUID assigned by HybridIndexBuilder.
                    Useful for cross-referencing against BM25/graph results in
                    Issue #19.
        source:     ``"vector_code"`` or ``"vector_doc"`` -- identifies which
                    collection this hit came from.
    """

    score: float
    file_path: str
    lines: tuple[int, int]
    symbol: str | None
    chunk_text: str
    point_id: str
    source: str


# ---------------------------------------------------------------------------
# Main search class
# ---------------------------------------------------------------------------


class VectorSearch:
    """Semantic search over the code and doc Qdrant collections.

    Dependency-injects both embedders and the Qdrant client so that all
    three can be swapped for fakes in unit tests without any network or model
    access.  The real Qdrant client is constructed lazily on first use,
    mirroring the pattern in :class:`~reporag.embedding.index_builder.HybridIndexBuilder`.
    """

    def __init__(
        self,
        code_embedder: Embedder,
        doc_embedder: Embedder,
        client: QdrantClient | None = None,
    ) -> None:
        """Initializes VectorSearch.

        Args:
            code_embedder: Embeds queries into the 768-dim code vector space.
                           Must satisfy the :class:`Embedder` protocol
                           (i.e. have an ``embed(text: str)`` method).
            doc_embedder:  Embeds queries into the 384-dim doc vector space.
                           Same protocol requirement as ``code_embedder``.
            client:        Optional pre-built QdrantClient.  When ``None``,
                           a client is created on first access from
                           ``settings.qdrant_url``.
        """
        self.code_embedder = code_embedder
        self.doc_embedder = doc_embedder
        self._client = client
        # Cache collection names from settings so callers and tests can
        # introspect them without re-reading settings.
        self.code_collection: str = settings.qdrant_collection_code
        self.docs_collection: str = settings.qdrant_collection_docs

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """Lazily initialize and cache the Qdrant client.

        Deferring construction until the first search call keeps module import
        cheap and allows tests to inject a fake after ``__init__`` runs.

        Returns:
            A connected :class:`~qdrant_client.QdrantClient` instance.
        """
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        language: str | None,
        file_path: str | None,
        repo_id: str | None,
        symbol_type: str | None = None,
    ) -> models.Filter | None:
        """Build a Qdrant must-filter from optional criteria.

        Returns ``None`` when no criteria are given so that Qdrant skips
        filter evaluation entirely (cheaper than an empty ``must`` list).

        All filter fields are indexed by :class:`~reporag.embedding.index_builder.HybridIndexBuilder`
        as keyword payload indexes, so each ``FieldCondition`` here hits an
        index rather than a full scan.

        Args:
            language:    Filter to a specific programming language.
            file_path:   Exact file-path match (Qdrant ``MatchValue``).
            repo_id:     Scope results to a single repository.
            symbol_type: Code-only filter (e.g. ``"function"``, ``"class"``).
                         Pass ``None`` for doc-collection filters.

        Returns:
            A :class:`~qdrant_client.http.models.Filter` or ``None``.
        """
        must: list[models.FieldCondition] = []
        if language:
            must.append(
                models.FieldCondition(
                    key="language", match=models.MatchValue(value=language)
                )
            )
        if file_path:
            must.append(
                models.FieldCondition(
                    key="file_path", match=models.MatchValue(value=file_path)
                )
            )
        if repo_id:
            must.append(
                models.FieldCondition(
                    key="repo_id", match=models.MatchValue(value=repo_id)
                )
            )
        if symbol_type:
            must.append(
                models.FieldCondition(
                    key="symbol_type", match=models.MatchValue(value=symbol_type)
                )
            )
        return models.Filter(must=must) if must else None

    def _search_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        """Execute a single Qdrant search, returning an empty list on failure.

        Isolating the ``client.search`` call here means a failure in one
        collection never prevents the other from being searched -- the caller
        (:meth:`search`) always receives whatever partial results are available.

        Args:
            collection_name: Qdrant collection to query.
            query_vector:    Pre-embedded query as a plain ``list[float]``
                             already in the collection's vector space.
            limit:           Maximum hits to retrieve from Qdrant.
            query_filter:    Optional filter to pass to Qdrant.

        Returns:
            A list of :class:`~qdrant_client.http.models.ScoredPoint` objects,
            or an empty list if the search raised an exception.
        """
        try:
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception:
            logger.exception("Search failed for collection '%s'", collection_name)
            return []

    def _code_hit_to_result(self, hit: models.ScoredPoint) -> RetrievalResult:
        """Convert a code-collection hit to a :class:`RetrievalResult`.

        Reads the subset of keys written by
        :class:`~reporag.embedding.index_builder.HybridIndexBuilder` into the
        code collection payload: ``file_path``, ``start_line``, ``end_line``,
        ``symbol``, and ``content``.  The full payload schema also contains
        ``language``, ``symbol_type``, ``repo_id``, ``parent_symbol``,
        ``qualified_name``, ``chunk_kind``, ``token_count``, and
        ``is_continuation``; those fields are available in ``hit.payload``
        for callers that need them but are not mapped into
        :class:`RetrievalResult`.

        Args:
            hit: A :class:`~qdrant_client.http.models.ScoredPoint` from the
                 code collection.

        Returns:
            A :class:`RetrievalResult` with ``source="vector_code"``.
        """
        payload = hit.payload or {}
        start_line: int = payload.get("start_line", 0)
        return RetrievalResult(
            score=hit.score,
            file_path=payload.get("file_path", ""),
            lines=(start_line, payload.get("end_line", start_line)),
            symbol=payload.get("symbol"),
            chunk_text=payload.get("content", ""),
            point_id=str(hit.id),
            source="vector_code",
        )

    def _doc_hit_to_result(self, hit: models.ScoredPoint) -> RetrievalResult:
        """Convert a doc-collection hit to a :class:`RetrievalResult`.

        Reads the subset of keys written by
        :class:`~reporag.embedding.index_builder.HybridIndexBuilder` into the
        doc collection payload: ``file_path``, ``start_line``, ``end_line``,
        ``symbol_id``, and ``text``.  The full payload schema also contains
        ``repo_id``, ``doc_type``, ``language``, and optionally ``metadata``;
        those fields are available in ``hit.payload`` for callers that need
        them.  Note that doc payloads use ``symbol_id`` (not ``symbol``) and
        ``text`` (not ``content``).

        Args:
            hit: A :class:`~qdrant_client.http.models.ScoredPoint` from the
                 doc collection.

        Returns:
            A :class:`RetrievalResult` with ``source="vector_doc"``.
        """
        payload = hit.payload or {}
        start_line: int = payload.get("start_line", 0)
        return RetrievalResult(
            score=hit.score,
            file_path=payload.get("file_path", ""),
            lines=(start_line, payload.get("end_line", start_line)),
            symbol=payload.get("symbol_id"),
            chunk_text=payload.get("text", ""),
            point_id=str(hit.id),
            source="vector_doc",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
        file_path: str | None = None,
        symbol_type: str | None = None,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search both collections and return merged, deduplicated top-k results.

        The method embeds the query separately for each collection (768-dim for
        code, 384-dim for docs), searches them independently, merges the hits,
        sorts by cosine similarity, deduplicates, and trims to ``top_k``.

        Failures in either collection are caught and logged independently so
        that partial results from the healthy collection are always returned.

        Args:
            query:       Raw natural-language or code query string.
            top_k:       Number of results to return.  Defaults to
                         ``settings.vector_search_top_k``.  Must be positive.
            language:    Restrict results to a programming language
                         (e.g. ``"python"``).
            file_path:   Restrict results to an exact file path.
            symbol_type: Restrict to a symbol kind (e.g. ``"function"``).
                         When set, the doc collection is skipped because doc
                         payloads have no ``symbol_type`` field.
            repo_id:     Restrict results to a single repository -- essential
                         when multiple repos share one Qdrant instance.

        Returns:
            Up to ``top_k`` :class:`RetrievalResult` objects sorted by cosine
            similarity descending.  The final count may be less than ``top_k``
            after deduplication.

        Raises:
            ValueError: If the resolved ``top_k`` is not a positive integer.
        """
        start_time = time.perf_counter()
        limit = top_k if top_k is not None else settings.vector_search_top_k
        if limit <= 0:
            raise ValueError(f"top_k must be a positive integer, got {limit}")

        # Build the code filter first (includes optional symbol_type).
        # The doc filter shares language/file_path/repo_id but never
        # symbol_type -- that field does not exist in the doc payload schema.
        code_filter = self._build_filter(language, file_path, repo_id, symbol_type)
        doc_filter = self._build_filter(language, file_path, repo_id)

        merged: list[RetrievalResult] = []

        code_hits = self._search_collection(
            self.code_collection,
            self.code_embedder.embed(query).tolist(),
            limit,
            code_filter,
        )
        merged.extend(self._code_hit_to_result(h) for h in code_hits)

        # Skip doc collection when symbol_type is set -- doc payloads carry no
        # symbol_type, so filtering on it would always return zero results.
        if not symbol_type:
            doc_hits = self._search_collection(
                self.docs_collection,
                self.doc_embedder.embed(query).tolist(),
                limit,
                doc_filter,
            )
            merged.extend(self._doc_hit_to_result(h) for h in doc_hits)

        merged.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate on (source, point_id).
        #
        # Issue #15 assigns every indexed point a *stable, deterministic* UUID
        # via uuid5((repo_id, file_path, symbol, span, part)).  That means each
        # unique chunk has one unique point_id regardless of how many times it
        # is re-indexed.  Including ``source`` in the key prevents a code hit
        # and a doc hit for the same physical symbol from incorrectly collapsing
        # into one result -- they represent different views (raw code vs.
        # docstring) and may both be useful to a downstream re-ranker.
        deduped: list[RetrievalResult] = []
        seen: set[tuple[str, str]] = set()
        for res in merged:
            key = (res.source, res.point_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(res)
            if len(deduped) >= limit:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "vector_search done | query=%r top_k=%d merged=%d returned=%d latency_ms=%.2f",
            query,
            limit,
            len(merged),
            len(deduped),
            elapsed_ms,
        )
        return deduped

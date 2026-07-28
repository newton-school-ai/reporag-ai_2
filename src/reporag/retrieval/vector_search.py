"""Vector semantic search.

Queries Qdrant with an embedded query vector. Returns top-k results with
scores and metadata payloads. Supports filtering by language, file path,
and symbol type.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client.models import Filter

    from reporag.embedding.code_embedder import CodeEmbedder
    from reporag.embedding.doc_embedder import DocEmbedder

from reporag.config import settings

logger = logging.getLogger(__name__)

# Compiled pattern that matches any fnmatch wildcard character.
# Using this is more reliable than manually checking individual chars.
_GLOB_CHARS_RE = re.compile(r"[*?\[\]]")


def _is_glob(pattern: str) -> bool:
    """Return True if *pattern* contains any fnmatch wildcard character."""
    return bool(_GLOB_CHARS_RE.search(pattern))


def _symbol_name_from_code_payload(payload: dict[str, Any]) -> str | None:
    """Extract the best human-readable symbol name from a code-chunk payload."""
    return (
        payload.get("qualified_name")
        or payload.get("symbol")
        or payload.get("parent_symbol")
    )


@dataclass
class RetrievalResult:
    """A single retrieved result containing code chunk text and metadata."""

    score: float
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol_name: str | None
    chunk_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None


class VectorSearch:
    """Performs vector semantic search over code and documentation collections in Qdrant.

    Queries both collections **concurrently** using query embeddings, combines the
    results, deduplicates candidates, and supports filtering by language, symbol type,
    and file path (exact or glob pattern).

    Args:
        client: A pre-built Qdrant client (or duck-typed fake).  When omitted a real
            ``QdrantClient`` is created lazily on first use.
        qdrant_url: Override ``settings.qdrant_url``.  Use ``":memory:"`` for tests.
        collection_code: Override ``settings.qdrant_collection_code``.
        collection_docs: Override ``settings.qdrant_collection_docs``.
        code_embedder: Override the default ``CodeEmbedder`` (useful in tests).
        doc_embedder: Override the default ``DocEmbedder`` (useful in tests).
        glob_candidate_multiplier: When a glob file-path filter is used, Qdrant is
            queried for ``max(100, top_k * glob_candidate_multiplier)`` candidates so
            that enough results survive the Python-side fnmatch filter.  Defaults to 5.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        qdrant_url: str | None = None,
        collection_code: str | None = None,
        collection_docs: str | None = None,
        code_embedder: CodeEmbedder | None = None,
        doc_embedder: DocEmbedder | None = None,
        glob_candidate_multiplier: int = 5,
    ) -> None:
        """Initialize VectorSearch with optional custom client and config overrides."""
        self._client = client
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_code = collection_code or settings.qdrant_collection_code
        self.collection_docs = collection_docs or settings.qdrant_collection_docs
        if code_embedder is None:
            from reporag.embedding.code_embedder import CodeEmbedder

            code_embedder = CodeEmbedder()
        self.code_embedder = code_embedder

        if doc_embedder is None:
            from reporag.embedding.doc_embedder import DocEmbedder

            doc_embedder = DocEmbedder()
        self.doc_embedder = doc_embedder

        self._glob_candidate_multiplier = glob_candidate_multiplier

    @property
    def client(self) -> Any:
        """Get the Qdrant client, constructing it lazily if not provided."""
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.info("Connecting to Qdrant at %s", self.qdrant_url)
            if self.qdrant_url == ":memory:":
                self._client = QdrantClient(location=":memory:")
            else:
                url = self.qdrant_url
                if not url.startswith(("http://", "https://")):
                    url = f"http://{url}"
                self._client = QdrantClient(url=url)
        return self._client

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_code_filter(
        self,
        language: str | None,
        symbol_type: str | None,
        file_path_exact: str | None,
    ) -> Filter | None:
        """Build a Qdrant filter for the code collection."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = []
        if language:
            must.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        if symbol_type:
            must.append(
                FieldCondition(key="symbol_type", match=MatchValue(value=symbol_type))
            )
        if file_path_exact:
            must.append(
                FieldCondition(key="file_path", match=MatchValue(value=file_path_exact))
            )
        return Filter(must=must) if must else None

    def _build_docs_filter(
        self,
        language: str | None,
        symbol_type: str | None,
        file_path_exact: str | None,
    ) -> Filter | None:
        """Build a Qdrant filter for the doc collection.

        Symbol-type filtering uses the nested key ``metadata.symbol_type`` because
        :class:`~reporag.embedding.doc_embedder.DocEmbedder` stores it there.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = []
        if language:
            must.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        if symbol_type:
            must.append(
                FieldCondition(
                    key="metadata.symbol_type", match=MatchValue(value=symbol_type)
                )
            )
        if file_path_exact:
            must.append(
                FieldCondition(key="file_path", match=MatchValue(value=file_path_exact))
            )
        return Filter(must=must) if must else None

    def _qdrant_search(
        self,
        collection: str,
        vector: list[float],
        query_filter: Filter | None,
        limit: int,
        score_threshold: float | None,
    ) -> list[Any]:
        """Execute a single Qdrant similarity search, returning raw scored points.

        Exceptions are intentionally **not** caught here -- they propagate to the
        ``ThreadPoolExecutor`` future so the ``as_completed`` loop in
        :meth:`search` can detect whether one or both collections failed and
        decide whether to raise or degrade gracefully.
        """
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

    @staticmethod
    def _point_to_code_result(point: Any) -> RetrievalResult:
        """Convert a Qdrant scored point from the code collection to a RetrievalResult."""
        p: dict[str, Any] = point.payload or {}
        return RetrievalResult(
            score=point.score,
            file_path=p.get("file_path") or "",
            start_line=p.get("start_line"),
            end_line=p.get("end_line"),
            symbol_name=_symbol_name_from_code_payload(p),
            chunk_text=p.get("content") or "",
            metadata=p,
        )

    @staticmethod
    def _point_to_doc_result(point: Any) -> RetrievalResult:
        """Convert a Qdrant scored point from the doc collection to a RetrievalResult."""
        p: dict[str, Any] = point.payload or {}
        return RetrievalResult(
            score=point.score,
            file_path=p.get("file_path") or "",
            start_line=p.get("start_line"),
            end_line=p.get("end_line"),
            symbol_name=p.get("symbol_id"),
            chunk_text=p.get("text") or p.get("content") or "",
            metadata=p,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        file_path: str | None = None,
        symbol_type: str | None = None,
        score_threshold: float | None = 0.0,
    ) -> list[RetrievalResult]:
        """Perform semantic search over code and doc collections, returning top_k results.

        Both Qdrant collections are queried **concurrently** to minimise latency.

        Args:
            query: Natural language or code query string.
            top_k: Maximum number of results to return.
            language: Filter results to a specific programming language.
            file_path: Filter by exact file path, or a glob pattern (e.g.
                ``"src/**/*.py"``).  Exact paths are pushed to Qdrant; glob
                patterns are applied in Python after retrieval.
            symbol_type: Filter by symbol type (e.g. ``"function"``, ``"class"``).
            score_threshold: Minimum cosine similarity score (inclusive).  Results
                below this threshold are dropped.  Defaults to ``0.0`` to exclude
                orthogonal and negative matches.  Set to ``None`` to return all
                results regardless of score.

        Returns:
            Deduplicated list of :class:`RetrievalResult` objects sorted by score
            descending, containing at most *top_k* items.

        Raises:
            RuntimeError: If **both** collection searches fail.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        # Determine the effective file_path filter mode
        glob_pattern: str | None = None
        exact_file_path: str | None = None
        if file_path:
            if _is_glob(file_path):
                glob_pattern = file_path
            else:
                exact_file_path = file_path

        # When a glob is in play, fetch more candidates so the post-filter
        # has enough material to fill top_k slots.
        qdrant_limit = (
            max(100, top_k * self._glob_candidate_multiplier) if glob_pattern else top_k
        )

        # Build filters for each collection
        filter_code = self._build_code_filter(language, symbol_type, exact_file_path)
        filter_docs = self._build_docs_filter(language, symbol_type, exact_file_path)

        # Embed query vectors (sequential; each model is already GPU-accelerated)
        query_vector_code = self.code_embedder.embed(query).tolist()
        query_vector_docs = self.doc_embedder.embed(query).tolist()

        # Query both collections concurrently to reduce wall-clock latency
        code_points: list[Any] = []
        doc_points: list[Any] = []
        code_failed = False
        doc_failed = False

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_code = pool.submit(
                self._qdrant_search,
                self.collection_code,
                query_vector_code,
                filter_code,
                qdrant_limit,
                score_threshold,
            )
            future_docs = pool.submit(
                self._qdrant_search,
                self.collection_docs,
                query_vector_docs,
                filter_docs,
                qdrant_limit,
                score_threshold,
            )

            for fut in as_completed((future_code, future_docs)):
                try:
                    result = fut.result()
                    if fut is future_code:
                        code_points = result
                    else:
                        doc_points = result
                except Exception as exc:
                    if fut is future_code:
                        code_failed = True
                        logger.warning(
                            "Code collection search raised an exception: %s", exc
                        )
                    else:
                        doc_failed = True
                        logger.warning(
                            "Doc collection search raised an exception: %s", exc
                        )

        if code_failed and doc_failed:
            raise RuntimeError(
                "Both vector searches failed.  Check Qdrant connectivity and logs."
            )

        # Convert raw Qdrant points to RetrievalResult objects
        code_results = [self._point_to_code_result(p) for p in code_points]
        doc_results = [self._point_to_doc_result(p) for p in doc_points]

        # Python-side score threshold guard (protects against fakes/mocks that
        # ignore the score_threshold param in client.search).
        # To match the docstring, thresholds are inclusive (>=), but the default
        # threshold of 0.0 uses strict (>) comparison to exclude orthogonal (0.0)
        # and negative matches.
        if score_threshold is not None:
            if score_threshold == 0.0:
                code_results = [r for r in code_results if r.score > 0.0]
                doc_results = [r for r in doc_results if r.score > 0.0]
            else:
                code_results = [r for r in code_results if r.score >= score_threshold]
                doc_results = [r for r in doc_results if r.score >= score_threshold]

        # Python-side glob filter (Qdrant has no native wildcard support)
        if glob_pattern:
            code_results = [
                r
                for r in code_results
                if r.file_path and fnmatch.fnmatchcase(r.file_path, glob_pattern)
            ]
            doc_results = [
                r
                for r in doc_results
                if r.file_path and fnmatch.fnmatchcase(r.file_path, glob_pattern)
            ]

        # Python-side symbol_type guard for doc results.
        #
        # The doc collection stores symbol_type inside a nested ``metadata``
        # sub-dict (key path ``metadata.symbol_type``) because
        # :class:`~reporag.embedding.doc_embedder.DocEmbedder` puts it there.
        # Qdrant's nested-key filter syntax is not supported by the in-memory
        # client used in tests and is unreliable across older server versions,
        # so we always apply the filter here in Python as well.  For code
        # results the Qdrant-side filter already operated on the top-level
        # ``symbol_type`` key, so no second pass is needed there.
        if symbol_type:

            def _doc_symbol_type(r: RetrievalResult) -> str | None:
                """Safely extract symbol_type from a doc result's nested metadata.

                DocEmbedder stores ``symbol_type`` inside a sub-dict under the
                key ``"metadata"`` (e.g. ``payload["metadata"]["symbol_type"]``).
                Guard against malformed payloads where that sub-value might be
                ``None`` or a non-dict type so we never raise ``AttributeError``.
                """
                sub = r.metadata.get("metadata")
                if not isinstance(sub, dict):
                    return None
                return sub.get("symbol_type")

            doc_results = [r for r in doc_results if _doc_symbol_type(r) == symbol_type]

        # Merge and deduplicate: same (file, start, end) -> keep higher score
        seen: dict[tuple[str, int | None, int | None], RetrievalResult] = {}
        for r in code_results + doc_results:
            key = (r.file_path, r.start_line, r.end_line)
            existing = seen.get(key)
            if existing is None or r.score > existing.score:
                seen[key] = r

        # Sort by score descending and cap at top_k
        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return deduped[:top_k]

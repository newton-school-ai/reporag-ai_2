"""BM25 sparse keyword search.

Queries the BM25 index with code-aware tokenization. Excels at finding
exact identifier matches that vector search may miss.

Design
------
:class:`BM25Search` wraps :class:`~reporag.embedding.index_builder.BM25Index`
with post-filtering, exact-name boosting, and :class:`RetrievalResult` output
conversion.

The BM25 index stores only **code** chunks (not documentation prose), so
every result from this retriever is a code chunk.  There is no ``language``
filter because the BM25 metadata does not include a ``language`` field --
use :class:`~reporag.retrieval.vector_search.VectorSearch` for
language-scoped search and let the fusion layer (Issue 19) combine both.

Usage::

    from reporag.retrieval.bm25_search import BM25Search

    searcher = BM25Search()
    searcher.load_index("data/my-repo.bm25.pkl")
    results = searcher.search("authenticate_user", top_k=10)
    for r in results:
        print(f"{r.score:.3f} | {r.file_path}:{r.start_line} | {r.symbol_name}")
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult, _is_glob

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 sparse keyword search for code identifiers.

    Wraps :class:`~reporag.embedding.index_builder.BM25Index` with
    post-filtering, exact-name boosting, and :class:`RetrievalResult`
    output conversion.

    Args:
        bm25_index: A pre-built :class:`BM25Index` instance (useful in tests).
            When ``None`` (the default), the caller must invoke
            :meth:`load_index` before searching.
    """

    def __init__(
        self,
        bm25_index: Any | None = None,
    ) -> None:
        self._index = bm25_index

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def load_index(self, path: str | Path | None = None) -> None:
        """Load a persisted BM25 index from *path*.

        Args:
            path: Filesystem path to a ``.bm25.pkl`` file written by
                :meth:`~reporag.embedding.index_builder.BM25Index.save`.
                When ``None``, defaults to ``data/bm25_index.pkl``.
        """
        from reporag.embedding.index_builder import BM25Index

        resolved = Path(path) if path is not None else Path("data/bm25_index.pkl")
        logger.info("Loading BM25 index from %s", resolved)
        self._index = BM25Index.load(resolved)

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        name_boost: float = 2.0,
        *,
        file_path: str | None = None,
        symbol_type: str | None = None,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search the BM25 index for code chunks matching *query*.

        Args:
            query:       Natural-language or identifier query string.
            top_k:       Maximum results to return.  Defaults to
                         ``settings.bm25_search_top_k``.
            name_boost:  Multiplicative boost applied when the query appears
                         as a substring of a result's symbol name (case-
                         insensitive).  Set to ``1.0`` to disable boosting.
                         Defaults to ``2.0``.
            file_path:   Filter by exact file path or glob pattern
                         (e.g. ``"src/**/*.py"``).
            symbol_type: Filter by symbol type (e.g. ``"function"``,
                         ``"class"``).
            repo_id:     Filter to a specific repository.

        Returns:
            Up to *top_k* :class:`RetrievalResult` objects sorted by
            BM25 score (descending), with exact-name boost applied.

        Raises:
            RuntimeError: If no BM25 index has been loaded.

        Note:
            There is no ``language`` filter because the BM25 index does
            not store a ``language`` field in its metadata.
        """
        if self._index is None:
            raise RuntimeError(
                "No BM25 index loaded. Call load_index() or pass a "
                "BM25Index to the constructor."
            )

        if not query or not query.strip():
            return []

        effective_top_k = top_k if top_k is not None else settings.bm25_search_top_k

        # Always over-fetch (spec: top_k * 3) so post-filtering and
        # re-sorting after boost have enough candidates.
        fetch_k = effective_top_k * 3
        raw_results = self._index.search(query, top_k=fetch_k)

        if not raw_results:
            return []

        # ------------------------------------------------------------------
        # Apply name boost
        # ------------------------------------------------------------------
        # The spec checks `symbol_name in query` but our symbols are
        # qualified names like "auth.authenticate_user".  We check
        # `query in symbol` instead, so "authenticate_user" correctly
        # matches "auth.authenticate_user".
        query_lower = query.strip().lower()
        boosted: list[tuple[float, dict[str, Any]]] = []

        for hit in raw_results:
            score = hit["score"]
            meta = hit["metadata"]

            if name_boost > 1.0:
                symbol = meta.get("symbol")
                if symbol and query_lower in symbol.lower():
                    score *= name_boost

            boosted.append((score, meta))

        # Re-sort by boosted score descending
        boosted.sort(key=lambda pair: pair[0], reverse=True)

        # ------------------------------------------------------------------
        # Post-filter on metadata
        # ------------------------------------------------------------------
        filtered: list[tuple[float, dict[str, Any]]] = []
        for score, meta in boosted:
            if symbol_type is not None and meta.get("symbol_type") != symbol_type:
                continue
            if repo_id is not None and meta.get("repo_id") != repo_id:
                continue
            if file_path is not None:
                hit_path = meta.get("file_path") or ""
                if _is_glob(file_path):
                    if not fnmatch.fnmatchcase(hit_path, file_path):
                        continue
                elif hit_path != file_path:
                    continue
            filtered.append((score, meta))

        # ------------------------------------------------------------------
        # Convert to RetrievalResult
        # ------------------------------------------------------------------
        results: list[RetrievalResult] = []
        for score, meta in filtered[:effective_top_k]:
            results.append(
                RetrievalResult(
                    score=score,
                    file_path=meta.get("file_path") or "",
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    symbol_name=meta.get("symbol"),
                    chunk_text=meta.get("content") or "",
                    metadata=meta,
                )
            )

        return results

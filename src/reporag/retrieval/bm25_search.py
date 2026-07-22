"""BM25 sparse keyword search for code identifiers.

Queries a pre-built :class:`~reporag.embedding.index_builder.BM25Index` with
code-aware tokenization.  Excels at finding **exact identifier matches** that
vector search may miss -- searching for ``"authenticate_user"`` returns the
defining function as rank 1, not just files that happen to mention it.

Design
------
:class:`BM25Search` mirrors the public API surface of
:class:`~reporag.retrieval.vector_search.VectorSearch` so downstream
fusion (Issue 19) can treat both identically:

* same ``search(query, top_k, language, file_path, symbol_type)`` signature,
* same :class:`~reporag.retrieval.vector_search.RetrievalResult` return type,
* same ``top_k < 1`` -> ``ValueError`` contract.

On top of raw BM25 scoring, an **exact-name boost** (configurable, default
``2.0x``) is applied to any result whose ``symbol`` metadata
case-insensitively matches the full query string.  This is what guarantees
the defining function -- not just a mention -- is rank 1 for identifier
queries.

The BM25 index itself is loaded lazily: either from a pickle file on disk
(created by :meth:`~reporag.embedding.index_builder.HybridIndexBuilder.save_bm25`)
or injected directly as a :class:`~reporag.embedding.index_builder.BM25Index`
instance (useful for tests).

Usage
-----
::

    from reporag.retrieval.bm25_search import BM25Search

    searcher = BM25Search(index_path="data/my-repo.bm25.pkl")
    results = searcher.search("authenticate_user", top_k=10)
    for r in results:
        print(f"{r.score:.3f} | {r.file_path}:{r.start_line} | {r.symbol_name}")
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any

from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Compiled pattern that matches any fnmatch wildcard character.
_GLOB_CHARS_RE = re.compile(r"[*?\[\]]")


def _is_glob(pattern: str) -> bool:
    """Return True if *pattern* contains any fnmatch wildcard character."""
    return bool(_GLOB_CHARS_RE.search(pattern))


def _normalize_symbol(name: str | None) -> str:
    """Lower-case and strip a symbol name for case-insensitive comparison."""
    if not name:
        return ""
    return name.strip().lower()


class BM25Search:
    """Performs BM25 sparse keyword search over a pre-built code index.

    Queries the :class:`~reporag.embedding.index_builder.BM25Index` using the
    same code-aware tokenizer that was used during indexing, ensuring lexical
    consistency.  Supports post-search filtering and exact-name boosting.

    Args:
        index_path: Path to a persisted BM25 index pickle file (created by
            :meth:`~reporag.embedding.index_builder.HybridIndexBuilder.save_bm25`).
            The index is loaded lazily on first :meth:`search` call.
        bm25_index: An already-constructed
            :class:`~reporag.embedding.index_builder.BM25Index` instance.
            Takes precedence over *index_path* when both are provided.
            Primarily useful for dependency injection in tests.
        exact_name_boost: Multiplicative boost factor applied to results whose
            ``symbol`` metadata case-insensitively matches the full query
            string.  Set to ``1.0`` to disable boosting.  Defaults to ``2.0``.
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        *,
        bm25_index: Any | None = None,
        exact_name_boost: float = 2.0,
    ) -> None:
        """Initialize BM25Search with an index source and boost configuration."""
        self._index_path = Path(index_path) if index_path else None
        self._index = bm25_index
        self._exact_name_boost = exact_name_boost

    # ------------------------------------------------------------------
    # Lazy index loading
    # ------------------------------------------------------------------

    @property
    def index(self) -> Any:
        """The BM25 index, loaded lazily from disk if not injected."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> Any:
        """Load the BM25 index from the configured path.

        Raises:
            FileNotFoundError: If no index path was provided or the file
                does not exist.
        """
        if self._index_path is None:
            raise FileNotFoundError(
                "No BM25 index available: provide either index_path or "
                "bm25_index to the BM25Search constructor."
            )
        if not self._index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {self._index_path}")

        from reporag.embedding.index_builder import BM25Index

        logger.info("Loading BM25 index from %s", self._index_path)
        return BM25Index.load(self._index_path)

    def load_index(self) -> None:
        """Eagerly load the BM25 index (forces the lazy property).

        Provided for compatibility with the usage example in the issue spec::

            searcher = BM25Search()
            searcher.load_index()
        """
        _ = self.index

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
    ) -> list[RetrievalResult]:
        """Search the BM25 index for code matching *query*.

        The query is tokenized with the same code-aware tokenizer used
        during indexing, ensuring consistent lexical matching.  Results
        are optionally boosted for exact symbol name matches and filtered
        by language, file path, and symbol type.

        Args:
            query: Natural language or code identifier query string.
            top_k: Maximum number of results to return.
            language: Filter results to a specific programming language
                (e.g. ``"python"``).
            file_path: Filter by exact file path, or a glob pattern
                (e.g. ``"src/**/*.py"``).  Mirrors the
                :class:`~reporag.retrieval.vector_search.VectorSearch`
                file_path filter semantics.
            symbol_type: Filter by symbol type (e.g. ``"function"``,
                ``"class"``).

        Returns:
            List of :class:`~reporag.retrieval.vector_search.RetrievalResult`
            objects sorted by score descending, containing at most *top_k*
            items.  Zero-scoring results (no lexical overlap) are excluded.

        Raises:
            ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        # Determine how many raw results to fetch from BM25. When filters
        # are active we over-fetch so enough results survive post-filtering.
        has_filters = bool(language or file_path or symbol_type)
        fetch_limit = max(100, top_k * 5) if has_filters else top_k

        raw_results = self.index.search(query, top_k=fetch_limit)

        # Convert raw BM25 results to RetrievalResult objects
        results = self._to_retrieval_results(raw_results)

        # Apply exact-name boost
        if self._exact_name_boost != 1.0:
            results = self._apply_exact_name_boost(results, query)

        # Apply post-search filters
        results = self._apply_filters(
            results,
            language=language,
            file_path=file_path,
            symbol_type=symbol_type,
        )

        # Sort by score descending (boost may have changed ordering)
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_retrieval_results(
        raw_results: list[dict[str, Any]],
    ) -> list[RetrievalResult]:
        """Convert raw BM25Index search results to RetrievalResult objects.

        The raw format from :meth:`BM25Index.search` is::

            {"id": doc_id, "score": float, "metadata": {...}}

        where ``metadata`` follows the schema set by
        :meth:`HybridIndexBuilder.upsert_code_chunks`:
        ``file_path``, ``symbol``, ``symbol_type``, ``chunk_kind``,
        ``repo_id``, ``start_line``, ``end_line``, ``content``.
        """
        results: list[RetrievalResult] = []
        for hit in raw_results:
            meta = hit.get("metadata", {})
            results.append(
                RetrievalResult(
                    score=hit["score"],
                    file_path=meta.get("file_path", ""),
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    symbol_name=meta.get("symbol"),
                    chunk_text=meta.get("content", ""),
                    metadata=meta,
                )
            )
        return results

    def _apply_exact_name_boost(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        """Boost results whose symbol name matches the query exactly.

        The comparison is case-insensitive and uses the raw query string
        (before tokenization) against the ``symbol`` metadata field.
        This ensures that ``"authenticate_user"`` boosts the
        ``authenticate_user`` definition above chunks that merely
        *mention* the identifier.

        A new list of :class:`RetrievalResult` objects is returned so the
        originals are not mutated.
        """
        query_normalized = _normalize_symbol(query)
        if not query_normalized:
            return results

        boosted: list[RetrievalResult] = []
        for r in results:
            symbol_normalized = _normalize_symbol(r.symbol_name)
            # Match against either the full qualified name or the bare symbol
            # name.  "auth.authenticate_user" should boost when the query is
            # "authenticate_user".
            bare_name = (
                symbol_normalized.rsplit(".", 1)[-1] if symbol_normalized else ""
            )
            if query_normalized in (symbol_normalized, bare_name):
                boosted.append(
                    RetrievalResult(
                        score=r.score * self._exact_name_boost,
                        file_path=r.file_path,
                        start_line=r.start_line,
                        end_line=r.end_line,
                        symbol_name=r.symbol_name,
                        chunk_text=r.chunk_text,
                        metadata=r.metadata,
                    )
                )
            else:
                boosted.append(r)
        return boosted

    @staticmethod
    def _apply_filters(
        results: list[RetrievalResult],
        *,
        language: str | None,
        file_path: str | None,
        symbol_type: str | None,
    ) -> list[RetrievalResult]:
        """Apply post-search metadata filters to narrow down results.

        Supports exact file path matching, glob patterns (e.g.
        ``"src/**/*.py"``), language filtering, and symbol type filtering.
        """
        if not any((language, file_path, symbol_type)):
            return results

        # Determine file path filter mode
        glob_pattern: str | None = None
        exact_file_path: str | None = None
        if file_path:
            if _is_glob(file_path):
                glob_pattern = file_path
            else:
                exact_file_path = file_path

        filtered: list[RetrievalResult] = []
        for r in results:
            if language and r.metadata.get("language") != language:
                continue
            if exact_file_path and r.file_path != exact_file_path:
                continue
            if glob_pattern and not fnmatch.fnmatchcase(r.file_path, glob_pattern):
                continue
            if symbol_type and r.metadata.get("symbol_type") != symbol_type:
                continue
            filtered.append(r)
        return filtered

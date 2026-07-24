"""BM25 sparse keyword search.

Queries the BM25 index with code-aware tokenization. Excels at finding
exact identifier matches that vector search may miss.

Why
---
Dense vectors blur related-but-different symbols together (``get_user`` and
``fetch_user`` land close in embedding space). BM25 is a pure lexical
overlap score, so when someone searches for a specific identifier
(``get_user_by_id``) the chunk that actually contains those tokens wins,
regardless of semantic drift. See
:mod:`reporag.embedding.index_builder` for why the two retrieval paths are
built side by side (Issue 15) and how ``tokenize_code`` normalizes naming
styles (``getUserByID`` / ``get_user_by_id`` / ``GetUserById`` all tokenize
to the same ``{get, user, by, id}`` set).

Design
------
:class:`BM25Search` is deliberately a thin query-side wrapper around an
already-built :class:`~reporag.embedding.index_builder.BM25Index` -- all
the indexing/tokenization logic lives there so the two sides can never
drift apart. This module only:

* re-tokenizes the query with the **same** tokenizer the index was built
  with (``self.index.tokenizer``), never a hardcoded one, so query-side and
  index-side tokenization can't silently diverge if the tokenizer is ever
  swapped out in tests or configuration,
* asks the index for a wide-enough candidate pool (wider than *top_k* when
  filters or boosting are in play, since both can change which items
  survive/rank first),
* applies an **exact-identifier boost**: if the tokenized query exactly
  equals the tokenized ``symbol`` of a candidate, its score is multiplied by
  ``exact_match_boost``. This is what makes a literal identifier query
  return the *defining* chunk as top-1 instead of some unrelated chunk that
  merely mentions the name more often in prose/comments,
* applies the same ``language`` / ``file_path`` / ``symbol_type`` filter
  vocabulary as :class:`~reporag.retrieval.vector_search.VectorSearch`, and
  returns the exact same :class:`~reporag.retrieval.vector_search.RetrievalResult`
  dataclass, so callers (and the future RRF fusion step in
  :mod:`reporag.retrieval.fusion`, Issue 19) can treat both retrieval paths
  interchangeably.

Known limitation -- ``language`` filtering
-------------------------------------------
:meth:`~reporag.embedding.index_builder.HybridIndexBuilder.upsert_code_chunks`
does **not** put a ``language`` key into the BM25 metadata dict it builds
(only ``file_path``, ``symbol``, ``symbol_type``, ``chunk_kind``,
``repo_id``, ``start_line``, ``end_line``, ``content``). A BM25-only corpus
therefore has no language signal to filter on. Rather than silently ignore
the ``language`` argument (which would be misleading -- callers would think
it's applied), :meth:`BM25Search.search` treats a document with no
``language`` in its metadata as a **non-match** whenever a language filter
is requested, so passing ``language=...`` against metadata built purely by
``upsert_code_chunks`` returns an empty result set rather than unfiltered
results. If a caller's metadata *does* carry a ``language`` key (e.g. added
manually, or by a future indexing change), filtering works normally. This
is covered by ``TestLanguageFilter`` in the test suite so a future change to
the metadata schema is caught either way.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reporag.embedding.index_builder import BM25Index

from reporag.retrieval.vector_search import RetrievalResult, _is_glob

logger = logging.getLogger(__name__)


class BM25Search:
    """Performs BM25 sparse keyword search over a code corpus.

    Args:
        bm25_index: A pre-built :class:`~reporag.embedding.index_builder.BM25Index`
            to query. Mutually exclusive with *index_path*. When both are
            omitted, an empty index is created lazily (``search`` then
            simply returns ``[]`` until documents are added to it).
        index_path: Path to a ``BM25Index.save``-d index to load lazily on
            first use. Mutually exclusive with *bm25_index*.
        exact_match_boost: Multiplier applied to a candidate's raw BM25
            score when the tokenized query exactly equals the tokenized
            ``symbol`` of that candidate (e.g. a query for
            ``get_user_by_id`` against a chunk whose ``symbol`` is
            ``get_user_by_id``, or an equivalent ``getUserByID`` /
            ``GetUserById`` spelling). Must be ``>= 1.0`` -- a value below 1
            would *penalize* exact matches, which defeats the purpose.
            Defaults to ``2.0``.
        candidate_multiplier: How many extra candidates (relative to
            *top_k*) to pull from the underlying index before filtering and
            boosting are applied, so that a candidate outside the naive
            top-k window can still surface after boosting, or survive a
            filter that drops earlier-ranked items. Defaults to ``5``.

    Raises:
        ValueError: If both *bm25_index* and *index_path* are given, or if
            *exact_match_boost* is less than ``1.0``.
    """

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        *,
        index_path: str | Path | None = None,
        exact_match_boost: float = 2.0,
        candidate_multiplier: int = 5,
    ) -> None:
        """Initialize BM25Search with an index (direct, lazy-loaded, or empty)."""
        if bm25_index is not None and index_path is not None:
            raise ValueError(
                "Pass either bm25_index or index_path, not both "
                f"(got bm25_index={bm25_index!r}, index_path={index_path!r})."
            )
        if exact_match_boost < 1.0:
            raise ValueError(
                f"exact_match_boost must be >= 1.0, got {exact_match_boost!r}."
            )

        if candidate_multiplier < 1:
            raise ValueError(
                "candidate_multiplier must be >= 1, " f"got {candidate_multiplier!r}."
            )
        self._index = bm25_index
        self._index_path = index_path
        self.exact_match_boost = exact_match_boost
        self._candidate_multiplier = candidate_multiplier

    @property
    def index(self) -> BM25Index:
        """Get the underlying BM25Index, loading/constructing it lazily."""
        if self._index is None:
            from reporag.embedding.index_builder import BM25Index

            if self._index_path is None:
                logger.info("No BM25Index or index_path given; starting empty.")
                self._index = BM25Index()
            else:
                logger.info("Loading BM25Index from %s", self._index_path)
                self._index = BM25Index.load(self._index_path)
        return self._index

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(raw: dict[str, Any]) -> RetrievalResult:
        """Convert one raw ``BM25Index.search`` result dict to a RetrievalResult."""
        meta: dict[str, Any] = raw.get("metadata") or {}
        return RetrievalResult(
            score=float(raw["score"]),
            file_path=meta.get("file_path") or "",
            start_line=meta.get("start_line"),
            end_line=meta.get("end_line"),
            symbol_name=meta.get("symbol"),
            chunk_text=meta.get("content") or "",
            metadata=meta,
        )

    @staticmethod
    def _is_exact_identifier_match(
        query_tokens: tuple[str, ...],
        symbol: str | None,
        tokenizer: Any,
    ) -> bool:
        """Return True if *symbol*, tokenized, exactly equals *query_tokens*.

        Both empty-token cases are treated as "no match" -- an empty query
        or a candidate with no/blank ``symbol`` can never count as an exact
        identifier hit, however coincidentally the (empty) token lists would
        otherwise compare equal.
        """
        if not query_tokens or not symbol:
            return False
        symbol_tokens = tuple(tokenizer(symbol))
        return bool(symbol_tokens) and symbol_tokens == query_tokens

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
        boost_exact_match: bool = True,
    ) -> list[RetrievalResult]:
        """Perform BM25 keyword search, returning up to top_k results.

        Args:
            query: Keyword or identifier query string. Tokenized with the
                exact same code-aware tokenizer the index was built with.
            top_k: Maximum number of results to return. Must be ``>= 1``.
            language: Filter to a specific language. See the module
                docstring's "Known limitation" section -- this only works
                when candidate metadata actually carries a ``language`` key.
            file_path: Filter by exact file path, or a glob pattern (e.g.
                ``"src/**/*.py"``); glob detection mirrors
                :class:`~reporag.retrieval.vector_search.VectorSearch`.
            symbol_type: Filter by symbol type (e.g. ``"function"``,
                ``"class"``).
            boost_exact_match: Apply :attr:`exact_match_boost` to candidates
                whose ``symbol`` tokenizes to exactly the query's tokens.
                Defaults to ``True``.

        Returns:
            Up to *top_k* :class:`~reporag.retrieval.vector_search.RetrievalResult`
            objects, sorted by (possibly boosted) score descending. Results
            with a BM25 score of exactly zero are never returned, since the
            underlying index already drops those as pure noise. An empty or
            whitespace-only query, or a query with no lexical overlap with
            anything indexed, returns ``[]`` rather than raising.

        Raises:
            ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        has_filters = bool(language or file_path or symbol_type)
        # Filters and boosting can both promote/demote items relative to the
        # index's raw ranking, so pull a wider candidate pool than top_k
        # whenever either is in play -- otherwise a post-boost or
        # post-filter winner sitting just outside a narrow top_k window
        # would never be seen at all.
        candidate_k = top_k
        if has_filters or boost_exact_match:
            candidate_k = max(top_k * self._candidate_multiplier, 100)

        raw_results = self.index.search(query, top_k=candidate_k)
        results = [self._to_result(r) for r in raw_results]

        if boost_exact_match and query.strip():
            tokenizer = self.index.tokenizer
            query_tokens = tuple(tokenizer(query))
            if query_tokens:
                for r in results:
                    if self._is_exact_identifier_match(
                        query_tokens, r.symbol_name, tokenizer
                    ):
                        r.score *= self.exact_match_boost

        if language is not None:
            results = [r for r in results if r.metadata.get("language") == language]

        if file_path is not None:
            if _is_glob(file_path):
                results = [
                    r
                    for r in results
                    if r.file_path and fnmatch.fnmatchcase(r.file_path, file_path)
                ]
            else:
                results = [r for r in results if r.file_path == file_path]

        if symbol_type is not None:
            results = [
                r for r in results if r.metadata.get("symbol_type") == symbol_type
            ]

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

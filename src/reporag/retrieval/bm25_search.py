"""BM25 sparse keyword search.

Queries the BM25 index with code-aware tokenization. Excels at finding
exact identifier matches that vector search may miss.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporag.embedding.index_builder import BM25Index

from reporag.config import settings
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Compiled pattern that matches any fnmatch wildcard character.
_GLOB_CHARS_RE = re.compile(r"[*?\[\]]")


def _is_glob(pattern: str) -> bool:
    """Return True if *pattern* contains any fnmatch wildcard character."""
    return bool(_GLOB_CHARS_RE.search(pattern))


def _get_language_from_path(file_path: str) -> str | None:
    """Infer the programming language of a file from its extension."""
    ext = Path(file_path).suffix.lower()
    return settings.extension_map.get(ext)


class BM25Search:
    """Performs sparse keyword search over the code corpus using Okapi BM25.

    Loads a pre-built BM25 index from disk and tokenizes query strings using
    the same code-aware tokenizer that was used during indexing.
    Supports boosting exact name matches of class/function definitions and
    optional filters.

    Args:
        index_path: Default path to the serialized BM25 index. If omitted,
            defaults to a standard database file location or configuration.
        default_boost: Configurable multiplier applied to the BM25 score of
            exact symbol definition matches. Defaults to 2.0.
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        default_boost: float = 2.0,
    ) -> None:
        self.index_path = index_path or Path("data/bm25.pkl")
        self.default_boost = default_boost
        self._index: BM25Index | None = None

    def load_index(self, path: str | Path | None = None) -> None:
        """Load a BM25Index serialized file from disk.

        Args:
            path: Path to the index file. If omitted, uses self.index_path.

        Raises:
            FileNotFoundError: If the index file does not exist on disk.
        """
        load_path = Path(path or self.index_path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"BM25 index file not found at: {load_path.resolve()}"
            )

        from reporag.embedding.index_builder import BM25Index

        logger.info("Loading BM25 index from %s", load_path)
        self._index = BM25Index.load(load_path)

    def search(
        self,
        query: str,
        top_k: int = 10,
        boost_factor: float | None = None,
        language: str | None = None,
        file_path: str | None = None,
        symbol_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Perform BM25 keyword search, ranking matches by exactness and token overlap.

        Args:
            query: The keyword query string.
            top_k: Maximum number of search results to return.
            boost_factor: Configurable score multiplier for exact symbol definition
                matches. Defaults to self.default_boost.
            language: Filter results to a specific programming language.
            file_path: Filter by exact file path, or a glob pattern (e.g.
                ``"src/**/*.py"``).
            symbol_type: Filter by symbol type (e.g. ``"function"``, ``"class"``).

        Returns:
            A list of RetrievalResult objects sorted by score descending.

        Raises:
            RuntimeError: If search is called before loading the index.
            ValueError: If top_k is less than 1.
        """
        if self._index is None:
            raise RuntimeError("BM25 index is not loaded. Call load_index() first.")

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        clean_query = query.strip()
        if not clean_query:
            return []

        boost = boost_factor if boost_factor is not None else self.default_boost

        # Tokenize the query using the code-aware tokenizer from the index.
        # This ensures query tokens exactly match indexed document tokens.
        query_tokens = self._index.tokenizer(clean_query)
        if not query_tokens:
            return []

        # Extract all identifier words in the raw query string for exact symbol match boosting
        query_identifiers = {
            t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", clean_query)
        }

        # Query all documents with a score > 0 to apply post-filtering and boosting
        raw_results = self._index.search(clean_query, top_k=len(self._index))

        results: list[RetrievalResult] = []
        for item in raw_results:
            metadata = item.get("metadata") or {}
            doc_file = metadata.get("file_path") or ""

            # 1. Apply language filter (inferred from extension)
            if language:
                inferred_lang = _get_language_from_path(doc_file)
                if inferred_lang != language:
                    continue

            # 2. Apply file path filter
            if file_path:
                if _is_glob(file_path):
                    if not fnmatch.fnmatchcase(doc_file, file_path):
                        continue
                elif doc_file != file_path:
                    continue

            # 3. Apply symbol type filter
            if symbol_type and metadata.get("symbol_type") != symbol_type:
                continue

            # 4. Boost exact function/class/method definitions if the base symbol
            # name matches any token/identifier in the query.
            score = item["score"]
            symbol = metadata.get("symbol")
            if symbol and boost != 1.0:
                base_symbol = symbol.split(".")[-1]
                chunk_kind = metadata.get("chunk_kind")
                meta_symbol_type = metadata.get("symbol_type")

                # Definition check
                is_definition = (chunk_kind == "definition") or (
                    meta_symbol_type in ("function", "class", "method")
                )

                if is_definition and (base_symbol.lower() in query_identifiers):
                    score *= boost

            results.append(
                RetrievalResult(
                    score=score,
                    file_path=doc_file,
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                    symbol_name=symbol,
                    chunk_text=metadata.get("content") or "",
                    metadata=metadata,
                )
            )

        # Sort the filtered/boosted candidates by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

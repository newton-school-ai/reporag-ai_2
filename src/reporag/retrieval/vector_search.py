"""Vector semantic search.

Queries Qdrant with an embedded query vector. Returns top-k results with
scores and metadata payloads. Supports filtering by language, file path,
and symbol type.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from reporag.config import settings
from reporag.embedding.code_embedder import CodeEmbedder
from reporag.embedding.doc_embedder import DocEmbedder

logger = logging.getLogger(__name__)


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


class VectorSearch:
    """Performs vector semantic search over code and documentation collections in Qdrant.

    Queries both collections using query embeddings, combines the results,
    deduplicates candidates, and supports filtering by language, symbol type, and
    file path (exact or glob pattern).
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
    ) -> None:
        """Initialize VectorSearch with optional custom client and config overrides."""
        self._client = client
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_code = collection_code or settings.qdrant_collection_code
        self.collection_docs = collection_docs or settings.qdrant_collection_docs

        self.code_embedder = code_embedder or CodeEmbedder()
        self.doc_embedder = doc_embedder or DocEmbedder()

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

        Args:
            query: The natural language search query.
            top_k: Number of results to return.
            language: Optional programming language filter (e.g. 'python').
            file_path: Optional file path filter (can be exact or a glob pattern).
            symbol_type: Optional symbol type filter (e.g. 'class', 'function').
            score_threshold: Optional similarity score threshold. Defaults to 0.0
                (filters out orthogonal and negative matches).

        Returns:
            A list of RetrievalResult objects sorted by cosine similarity score descending.
        """
        # Embed the query vector for code (768-dim) and docs (384-dim)
        query_vector_code = self.code_embedder.embed(query).tolist()
        query_vector_docs = self.doc_embedder.embed(query).tolist()

        # Build Qdrant filter conditions
        must_code = []
        must_docs = []

        if language:
            cond = FieldCondition(key="language", match=MatchValue(value=language))
            must_code.append(cond)
            must_docs.append(cond)

        if symbol_type:
            must_code.append(
                FieldCondition(key="symbol_type", match=MatchValue(value=symbol_type))
            )
            must_docs.append(
                FieldCondition(
                    key="metadata.symbol_type", match=MatchValue(value=symbol_type)
                )
            )

        # Check if file_path is a glob pattern
        is_glob = False
        if file_path:
            is_glob = any(c in file_path for c in ["*", "?", "[", "]"])
            if not is_glob:
                cond = FieldCondition(
                    key="file_path", match=MatchValue(value=file_path)
                )
                must_code.append(cond)
                must_docs.append(cond)

        # If we have a glob, query more candidates to allow post-filtering in Python
        qdrant_limit = max(100, top_k * 5) if is_glob else top_k

        code_results: list[RetrievalResult] = []
        doc_results: list[RetrievalResult] = []

        # 1. Search code collection
        try:
            filter_code = Filter(must=must_code) if must_code else None
            res_code = self.client.search(
                collection_name=self.collection_code,
                query_vector=query_vector_code,
                query_filter=filter_code,
                limit=qdrant_limit,
                score_threshold=score_threshold,
            )
            for point in res_code:
                p = point.payload or {}
                code_results.append(
                    RetrievalResult(
                        score=point.score,
                        file_path=p.get("file_path", ""),
                        start_line=p.get("start_line"),
                        end_line=p.get("end_line"),
                        symbol_name=p.get("qualified_name")
                        or p.get("symbol")
                        or p.get("parent_symbol"),
                        chunk_text=p.get("content", ""),
                        metadata=p,
                    )
                )
        except Exception as e:
            logger.warning(
                "Error searching code collection '%s': %s", self.collection_code, e
            )

        # 2. Search doc collection
        try:
            filter_docs = Filter(must=must_docs) if must_docs else None
            res_docs = self.client.search(
                collection_name=self.collection_docs,
                query_vector=query_vector_docs,
                query_filter=filter_docs,
                limit=qdrant_limit,
                score_threshold=score_threshold,
            )
            for point in res_docs:
                p = point.payload or {}
                doc_results.append(
                    RetrievalResult(
                        score=point.score,
                        file_path=p.get("file_path", ""),
                        start_line=p.get("start_line"),
                        end_line=p.get("end_line"),
                        symbol_name=p.get("symbol_id"),
                        chunk_text=p.get("text") or p.get("content", ""),
                        metadata=p,
                    )
                )
        except Exception as e:
            logger.warning(
                "Error searching docs collection '%s': %s", self.collection_docs, e
            )

        # 3. Apply score threshold post-filter in Python (safeguard for fakes/mocks)
        if score_threshold is not None:
            code_results = [r for r in code_results if r.score > score_threshold]
            doc_results = [r for r in doc_results if r.score > score_threshold]

        # 4. Apply glob filter in Python if needed
        if is_glob and file_path:
            code_results = [
                r for r in code_results if fnmatch.fnmatch(r.file_path, file_path)
            ]
            doc_results = [
                r for r in doc_results if fnmatch.fnmatch(r.file_path, file_path)
            ]

        # 5. Merge and deduplicate results
        # Duplicate chunks are matching on file_path, start_line, and end_line
        merged = code_results + doc_results
        seen: dict[tuple[str, int | None, int | None], RetrievalResult] = {}
        for r in merged:
            key = (r.file_path, r.start_line, r.end_line)
            if key in seen:
                if r.score > seen[key].score:
                    seen[key] = r
            else:
                seen[key] = r

        deduped = list(seen.values())

        # 6. Sort by score descending and return top_k
        deduped.sort(key=lambda x: x.score, reverse=True)
        return deduped[:top_k]

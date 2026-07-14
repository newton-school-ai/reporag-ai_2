"""Hybrid index builder.

Creates and populates both a Qdrant vector collection and a BM25 sparse
index. Includes a code-aware tokenizer that splits camelCase and snake_case
identifiers for better keyword matching.
"""

import logging
import os
import pickle
import re
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from reporag.config import settings

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds and maintains hybrid retrieval indices (Qdrant + BM25)."""

    def __init__(self, qdrant_url: str | None = None) -> None:
        url = qdrant_url or settings.qdrant_url
        if url == ":memory:":
            self.qdrant = QdrantClient(location=":memory:")
        else:
            self.qdrant = QdrantClient(url=url)
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus: list[list[str]] = []
        self.bm25_path = "data/bm25_index.pkl"

    def code_tokenize(self, text: str) -> list[str]:
        """Tokenize code text, splitting on operators, camelCase, and snake_case."""
        parts = re.split(r"[^a-zA-Z0-9]+", text)
        tokens = []
        for part in parts:
            if not part:
                continue
            # Regex to match camelCase, PascalCase, abbreviations, and numbers
            pattern = r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b|[0-9])|[0-9]+"
            sub_tokens = re.finditer(pattern, part)
            found = [m.group(0).lower() for m in sub_tokens]
            if found:
                tokens.extend(found)
            else:
                tokens.append(part.lower())
        return tokens

    def _ensure_collections(self) -> None:
        """Create Qdrant collections if they do not exist."""
        if not self.qdrant.collection_exists(settings.qdrant_collection_code):
            logger.info("Creating collection: %s", settings.qdrant_collection_code)
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection_code,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        if not self.qdrant.collection_exists(settings.qdrant_collection_docs):
            logger.info("Creating collection: %s", settings.qdrant_collection_docs)
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection_docs,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def build_vector_index(
        self,
        chunks: Sequence[Any],
        code_embeddings: Any,
        doc_embeddings: Sequence[Any] | None = None,
    ) -> None:
        """Upsert code and doc embeddings to Qdrant."""
        self._ensure_collections()

        points_code = []
        # Convert code embeddings to list if it's a numpy array
        code_embeddings_list = (
            code_embeddings.tolist()
            if hasattr(code_embeddings, "tolist")
            else code_embeddings
        )

        for chunk, embedding in zip(chunks, code_embeddings_list, strict=True):
            payload = {
                "file_path": getattr(chunk, "file_path", ""),
                "start_line": getattr(chunk, "start_line", 0),
                "end_line": getattr(chunk, "end_line", 0),
                "qualified_name": getattr(chunk, "qualified_name", None),
                "parent_symbol": getattr(chunk, "parent_symbol", None),
                "language": getattr(chunk, "language", ""),
            }
            # Deterministic UUID for incremental updates
            content_hash = hash(getattr(chunk, "content", ""))
            unique_str = f"{payload['file_path']}:{payload['start_line']}:{payload['end_line']}:{content_hash}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            points_code.append(
                PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        if points_code:
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection_code, points=points_code
            )

        if doc_embeddings:
            points_docs = []
            for doc in doc_embeddings:
                payload = {
                    "symbol_id": getattr(doc, "symbol_id", ""),
                    "text": getattr(doc, "text", ""),
                }
                unique_str = f"{payload['symbol_id']}:{hash(payload['text'])}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

                # Convert vector to list if it's a numpy array
                vec = doc.vector
                vec_list = vec.tolist() if hasattr(vec, "tolist") else vec
                points_docs.append(
                    PointStruct(id=point_id, vector=vec_list, payload=payload)
                )

            if points_docs:
                self.qdrant.upsert(
                    collection_name=settings.qdrant_collection_docs, points=points_docs
                )

    def load_bm25(self) -> None:
        """Load the existing BM25 corpus from disk."""
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25_corpus = pickle.load(f)
            if self.bm25_corpus:
                self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []

    def save_bm25(self) -> None:
        """Save the BM25 corpus to disk for incremental updates."""
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25_corpus, f)

    def build_bm25_index(self, chunks: Sequence[Any]) -> None:
        """Tokenize chunks and append to BM25 index."""
        self.load_bm25()

        for chunk in chunks:
            content = getattr(chunk, "content", "")
            tokens = self.code_tokenize(content)
            self.bm25_corpus.append(tokens)

        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
            self.save_bm25()

    def vector_count(self) -> int:
        """Return total points in code collection."""
        if not self.qdrant.collection_exists(settings.qdrant_collection_code):
            return 0
        return self.qdrant.count(settings.qdrant_collection_code).count

    def doc_vector_count(self) -> int:
        """Return total points in docs collection."""
        if not self.qdrant.collection_exists(settings.qdrant_collection_docs):
            return 0
        return self.qdrant.count(settings.qdrant_collection_docs).count

    def bm25_doc_count(self) -> int:
        """Return number of documents in BM25 index."""
        return len(self.bm25_corpus)

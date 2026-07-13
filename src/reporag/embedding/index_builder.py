"""Hybrid index builder.

Creates and populates both a Qdrant vector collection and a BM25 sparse
index. Includes a code-aware tokenizer that splits camelCase and snake_case
identifiers for better keyword matching.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
import re
import uuid
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from reporag.config import settings
from reporag.embedding.doc_embedder import DocEmbedding
from reporag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def tokenize_code(text: str) -> list[str]:
    """Tokenize source code, splitting camelCase, snake_case, and operators.

    This tokenizer splits compound identifiers into individual words (e.g.
    ``getUserById`` -> ``['get', 'user', 'by', 'id']``), preserves operators,
    and returns a list of lowercased string tokens suitable for BM25 indexing.
    """
    if not text:
        return []

    # Split text into sequences of word characters/digits and multi-char operators.
    # Operator group matches greedily up to 3 chars (e.g. '!=', '>=', '**', '//'),
    # before falling back to single punctuation/bracket tokens.
    raw_tokens = re.findall(
        r"[a-zA-Z0-9_]+|[+\-*/%=!<>&|^~]{1,3}|[()\[\]{}:;.,?@#\"'\\]", text
    )

    final_tokens: list[str] = []
    for token in raw_tokens:
        if re.match(r"^[a-zA-Z0-9_]+$", token):
            # Split snake_case parts
            parts = token.split("_")
            for part in parts:
                if not part:
                    continue
                # Split camelCase using regex:
                # 1. Lowercase/digit followed by Uppercase -> separate
                s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
                # 2. Uppercase acronym followed by Uppercase word boundary -> separate
                s2 = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", s1)

                for sub_part in s2.split():
                    final_tokens.append(sub_part.lower())
        else:
            # It's an operator or punctuation
            final_tokens.append(token)

    return final_tokens


class IndexBuilder:
    """Builder for vector (Qdrant) and sparse (BM25) indexes.

    Coordinates creation, payload mapping, and incremental updates of Qdrant
    collections and BM25 persistent files.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        bm25_path: str | None = None,
    ) -> None:
        """Initialize connection to Qdrant and the BM25 persistent storage."""
        # Use settings fallback for Qdrant URL
        url = qdrant_url or settings.qdrant_url
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            if not url.startswith(("http://", "https://")):
                url = f"http://{url}"
            self.client = QdrantClient(url=url)

        self.bm25_path = bm25_path or "data/bm25_index.pkl"

    def build_vector_index(
        self,
        chunks: Sequence[Chunk],
        code_embeddings: np.ndarray,
        doc_embeddings: Sequence[DocEmbedding],
    ) -> None:
        """Create and populate Qdrant collections for code and doc embeddings.

        Handles collection creation, schema verification, payload indexing,
        and incremental updates (deleting existing points for updated files
        before upserting new ones).
        """
        # Ensure collections exist
        self._ensure_collection(settings.qdrant_collection_code, 768)
        self._ensure_collection(settings.qdrant_collection_docs, 384)

        # Build symbol-to-file path mapping from chunks for doc embeddings fallback
        symbol_to_file: dict[str, str] = {}
        for chunk in chunks:
            if chunk.qualified_name:
                symbol_to_file[chunk.qualified_name] = chunk.file_path

        # Determine all files being updated to perform incremental deletion
        updated_files: set[str] = set()
        for chunk in chunks:
            updated_files.add(chunk.file_path)
        for doc_emb in doc_embeddings:
            file_path = doc_emb.file_path or symbol_to_file.get(doc_emb.symbol_id or "")
            if file_path:
                updated_files.add(file_path)

        # Deleting old points for updated files to support incremental update
        for file_path in updated_files:
            logger.info("Deleting existing points for file %s", file_path)
            delete_filter = Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path))
                ]
            )
            self.client.delete(
                collection_name=settings.qdrant_collection_code,
                points_selector=FilterSelector(filter=delete_filter),
            )
            self.client.delete(
                collection_name=settings.qdrant_collection_docs,
                points_selector=FilterSelector(filter=delete_filter),
            )

        # Process and upsert code embeddings
        if len(chunks) > 0:
            if len(chunks) != len(code_embeddings):
                raise ValueError(
                    f"Number of chunks ({len(chunks)}) must match "
                    f"code embeddings count ({len(code_embeddings)})"
                )

            code_points: list[PointStruct] = []
            for chunk, embedding in zip(chunks, code_embeddings, strict=True):
                point_id = self._generate_chunk_uuid(chunk)
                payload = {
                    "file_path": chunk.file_path,
                    "file": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "lines": [chunk.start_line, chunk.end_line],
                    "symbol_name": chunk.qualified_name or "",
                    "symbol": chunk.qualified_name or "",
                    "chunk_text": chunk.content,
                    "content": chunk.content,
                    "language": chunk.language,
                    "chunk_kind": chunk.chunk_kind,
                    "chunk_index": chunk.chunk_index,
                    "part": chunk.part,
                    "total_parts": chunk.total_parts,
                    "is_continuation": chunk.is_continuation,
                }
                code_points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            if code_points:
                self.client.upsert(
                    collection_name=settings.qdrant_collection_code,
                    points=code_points,
                )

        # Process and upsert doc embeddings
        if len(doc_embeddings) > 0:
            doc_points: list[PointStruct] = []
            for doc_emb in doc_embeddings:
                point_id = self._generate_doc_uuid(doc_emb)
                file_path = (
                    doc_emb.file_path
                    or symbol_to_file.get(doc_emb.symbol_id or "")
                    or ""
                )
                payload = {
                    "symbol_name": doc_emb.symbol_id or "",
                    "symbol": doc_emb.symbol_id or "",
                    "chunk_text": doc_emb.text,
                    "content": doc_emb.text,
                    "file_path": file_path,
                    "file": file_path,
                    "start_line": doc_emb.start_line,
                    "end_line": doc_emb.end_line,
                    "lines": (
                        [doc_emb.start_line, doc_emb.end_line]
                        if doc_emb.start_line and doc_emb.end_line
                        else []
                    ),
                    "doc_type": doc_emb.doc_type,
                    "source": doc_emb.doc_type,
                }
                doc_points.append(
                    PointStruct(
                        id=point_id,
                        vector=doc_emb.vector.tolist(),
                        payload=payload,
                    )
                )

            if doc_points:
                self.client.upsert(
                    collection_name=settings.qdrant_collection_docs,
                    points=doc_points,
                )

    def build_bm25_index(self, chunks: Sequence[Chunk]) -> None:
        """Build and persist the BM25 index over the provided chunks.

        Supports incremental updates: merges new/updated chunks into the
        existing index corpus and fits a new BM25Okapi model.
        """
        # Load existing index state
        existing_chunks: list[Chunk] = []
        existing_tokens: list[list[str]] = []
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        existing_chunks = data.get("chunks", [])
                        existing_tokens = data.get("tokens", [])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load existing BM25 index from %s (%s). Rebuilding.",
                    self.bm25_path,
                    exc,
                )

        # Extract file paths being updated
        updated_files = {chunk.file_path for chunk in chunks}

        # Filter out existing chunks that belong to files being updated
        merged_chunks: list[Chunk] = []
        merged_tokens: list[list[str]] = []

        for ec, et in zip(existing_chunks, existing_tokens, strict=True):
            if ec.file_path not in updated_files:
                merged_chunks.append(ec)
                merged_tokens.append(et)

        # Add the new chunks and their tokens
        for chunk in chunks:
            merged_chunks.append(chunk)
            tokens = tokenize_code(chunk.content)
            merged_tokens.append(tokens)

        if not merged_chunks:
            # If everything was deleted, delete the file if it exists and return
            if os.path.exists(self.bm25_path):
                with contextlib.suppress(OSError):
                    os.remove(self.bm25_path)
            return

        # Fit a new BM25Okapi model
        bm25_model = BM25Okapi(merged_tokens)

        # Save index state
        os.makedirs(os.path.dirname(os.path.abspath(self.bm25_path)), exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25_model,
                    "chunks": merged_chunks,
                    "tokens": merged_tokens,
                },
                f,
            )

    def vector_count(self, collection_name: str | None = None) -> int:
        """Return the count of active points in the vector collections."""
        if collection_name:
            return self.client.count(collection_name).count

        # Default to sum of both collections
        total = 0
        with contextlib.suppress(Exception):
            total += self.client.count(settings.qdrant_collection_code).count
        with contextlib.suppress(Exception):
            total += self.client.count(settings.qdrant_collection_docs).count
        return total

    def bm25_doc_count(self) -> int:
        """Return the number of documents in the BM25 index."""
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    return len(data.get("chunks", []))
            except Exception:  # noqa: BLE001
                pass
        return 0

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create Qdrant collection and payload schemas if missing."""
        from qdrant_client.models import PayloadSchemaType

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            # Create keyword payload indexes for filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="file_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="file",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="language",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="symbol",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="symbol_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    @staticmethod
    def _generate_chunk_uuid(chunk: Chunk) -> str:
        """Generate a deterministic UUID based on chunk content/location."""
        unique_str = f"code:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.content}"
        hasher = hashlib.md5(unique_str.encode("utf-8"))
        return str(uuid.UUID(bytes=hasher.digest()))

    @staticmethod
    def _generate_doc_uuid(doc_emb: DocEmbedding) -> str:
        """Generate a deterministic UUID based on doc embedding content."""
        symbol_id = doc_emb.symbol_id or ""
        unique_str = f"doc:{symbol_id}:{doc_emb.text}"
        hasher = hashlib.md5(unique_str.encode("utf-8"))
        return str(uuid.UUID(bytes=hasher.digest()))

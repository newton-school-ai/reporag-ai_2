"""Unit tests for the IndexBuilder pipeline (Issue 15).

All Qdrant tests run against an in-memory instance, ensuring that no running
Qdrant Docker service is required.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from reporag.config import settings
from reporag.embedding.doc_embedder import DocEmbedding
from reporag.embedding.index_builder import IndexBuilder, tokenize_code
from reporag.ingestion.chunker import Chunk

# ===========================================================================
# Helper methods and test fixtures
# ===========================================================================


@pytest.fixture
def temp_bm25_path(tmp_path: pytest.TempPathFactory) -> str:
    """Return a temporary file path for BM25 index serialization."""
    return str(tmp_path / "bm25_index.pkl")


@pytest.fixture
def mock_chunks() -> list[Chunk]:
    """Return a sequence of mock Chunk objects."""
    return [
        Chunk(
            content="def authenticate_user(token):\n    pass",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=2,
            qualified_name="auth.authenticate_user",
        ),
        Chunk(
            content="class RequestParser:\n    def parse_body(self):\n        pass",
            file_path="src/parser.py",
            language="python",
            start_line=1,
            end_line=3,
            qualified_name="parser.RequestParser",
        ),
    ]


@pytest.fixture
def mock_code_embeddings() -> np.ndarray:
    """Return mock code embeddings corresponding to mock_chunks."""
    # 2 mock chunks, size 768
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((2, 768)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms < 1e-12, 1.0, norms)


@pytest.fixture
def mock_doc_embeddings() -> list[DocEmbedding]:
    """Return a list of mock DocEmbedding objects."""
    rng = np.random.default_rng(24)
    emb1 = rng.standard_normal(384).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    emb2 = rng.standard_normal(384).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    return [
        DocEmbedding(
            symbol_id="auth.authenticate_user",
            text="Verify user identity using standard token authentication mechanisms.",
            vector=emb1,
            doc_type="docstring",
            file_path="src/auth.py",
            start_line=1,
            end_line=2,
        ),
        DocEmbedding(
            symbol_id="parser.RequestParser",
            text="Extract and validate parameters from request body.",
            vector=emb2,
            doc_type="docstring",
            file_path="src/parser.py",
            start_line=1,
            end_line=3,
        ),
    ]


# ===========================================================================
# Code-aware Tokenizer tests
# ===========================================================================


def test_tokenize_code_empty_and_none() -> None:
    """Test tokenization of empty or None values."""
    assert tokenize_code("") == []
    assert tokenize_code(None) == []  # type: ignore[arg-type]


def test_tokenize_code_camel_case() -> None:
    """Test camelCase and PascalCase splitting."""
    assert tokenize_code("getUserById") == ["get", "user", "by", "id"]
    assert tokenize_code("HTMLParser") == ["html", "parser"]
    assert tokenize_code("simpleHTTPRequest") == ["simple", "http", "request"]


def test_tokenize_code_snake_case() -> None:
    """Test snake_case splitting."""
    assert tokenize_code("get_user_by_id") == ["get", "user", "by", "id"]
    assert tokenize_code("__init__") == ["init"]


def test_tokenize_code_mixed_identifiers() -> None:
    """Test mixed casing conventions (snake_case + camelCase)."""
    assert tokenize_code("get_user_ById") == ["get", "user", "by", "id"]
    assert tokenize_code("loadXML_File") == ["load", "xml", "file"]


def test_tokenize_code_operators() -> None:
    """Test splitting on operators and punctuation."""
    assert tokenize_code("x = a + b") == ["x", "=", "a", "+", "b"]
    assert tokenize_code("if (x != y):") == ["if", "(", "x", "!=", "y", ")", ":"]


# ===========================================================================
# IndexBuilder initialization & basic vector index checks
# ===========================================================================


def test_index_builder_init(temp_bm25_path: str) -> None:
    """Test initialization of IndexBuilder client connection."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)
    assert builder.bm25_path == temp_bm25_path
    assert builder.vector_count() == 0


def test_build_vector_index_creates_collections(
    temp_bm25_path: str,
    mock_chunks: list[Chunk],
    mock_code_embeddings: np.ndarray,
    mock_doc_embeddings: list[DocEmbedding],
) -> None:
    """Test that build_vector_index creates collection schemas and upserts points."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)

    builder.build_vector_index(mock_chunks, mock_code_embeddings, mock_doc_embeddings)

    # Check collections are created
    collections = builder.client.get_collections().collections
    names = {c.name for c in collections}
    assert settings.qdrant_collection_code in names
    assert settings.qdrant_collection_docs in names

    # Check correct vector sizes are used and points are count-verifiable
    assert builder.vector_count(settings.qdrant_collection_code) == 2
    assert builder.vector_count(settings.qdrant_collection_docs) == 2
    assert builder.vector_count() == 4


def test_build_vector_index_mismatched_lengths(
    temp_bm25_path: str,
    mock_chunks: list[Chunk],
    mock_doc_embeddings: list[DocEmbedding],
) -> None:
    """Test that build_vector_index raises ValueError on mismatched dimensions."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)
    bad_embeddings = np.zeros((5, 768), dtype=np.float32)

    with pytest.raises(ValueError, match="must match code embeddings count"):
        builder.build_vector_index(mock_chunks, bad_embeddings, mock_doc_embeddings)


# ===========================================================================
# Vector index incremental updates tests
# ===========================================================================


def test_vector_index_incremental_updates(
    temp_bm25_path: str,
    mock_chunks: list[Chunk],
    mock_code_embeddings: np.ndarray,
    mock_doc_embeddings: list[DocEmbedding],
) -> None:
    """Test that vector indexing supports incremental file updates correctly."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)

    # 1. Build initial index
    builder.build_vector_index(mock_chunks, mock_code_embeddings, mock_doc_embeddings)
    assert builder.vector_count(settings.qdrant_collection_code) == 2

    # 2. Update chunk metadata for parser.py (with different content/range)
    updated_chunks = [
        Chunk(
            content="class RequestParser:\n    def parse_body_v2(self):\n        pass",
            file_path="src/parser.py",
            language="python",
            start_line=1,
            end_line=4,
            qualified_name="parser.RequestParser",
        )
    ]
    rng = np.random.default_rng(100)
    updated_code_embeddings = rng.standard_normal((1, 768)).astype(np.float32)
    updated_code_embeddings /= np.linalg.norm(updated_code_embeddings)

    updated_doc_embeddings = [
        DocEmbedding(
            symbol_id="parser.RequestParser",
            text="Extract and validate parameters v2.",
            vector=rng.standard_normal(384).astype(np.float32),
            doc_type="docstring",
            file_path="src/parser.py",
            start_line=1,
            end_line=4,
        )
    ]

    # Re-run build_vector_index with ONLY the updated parser.py details
    builder.build_vector_index(
        updated_chunks, updated_code_embeddings, updated_doc_embeddings
    )

    # Total points should still be 2 in each collection (1 for auth.py + 1 updated parser.py)
    assert builder.vector_count(settings.qdrant_collection_code) == 2
    assert builder.vector_count(settings.qdrant_collection_docs) == 2

    # Verify that the old parser chunk was deleted and the new one was inserted
    code_scroll = builder.client.scroll(
        collection_name=settings.qdrant_collection_code,
        limit=10,
        with_payload=True,
    )
    payloads = [p.payload for p in code_scroll[0]]
    texts = {p["chunk_text"] for p in payloads if p}

    assert "def authenticate_user(token):\n    pass" in texts
    assert "class RequestParser:\n    def parse_body_v2(self):\n        pass" in texts
    assert "class RequestParser:\n    def parse_body(self):\n        pass" not in texts


# ===========================================================================
# BM25 sparse index tests (build, persistence, incremental updates)
# ===========================================================================


def test_build_bm25_index_basic(
    temp_bm25_path: str,
    mock_chunks: list[Chunk],
) -> None:
    """Test that build_bm25_index compiles the index and persists it to file."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)

    builder.build_bm25_index(mock_chunks)
    assert builder.bm25_doc_count() == 2
    assert os.path.exists(temp_bm25_path)

    # Load file and verify contents
    with open(temp_bm25_path, "rb") as f:
        data = pickle.load(f)
        assert "bm25" in data
        assert len(data["chunks"]) == 2
        assert len(data["tokens"]) == 2


def test_bm25_index_incremental_updates(
    temp_bm25_path: str,
    mock_chunks: list[Chunk],
) -> None:
    """Test that BM25 index updates incrementally (replacing old file chunks)."""
    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)

    # 1. Build initial index
    builder.build_bm25_index(mock_chunks)
    assert builder.bm25_doc_count() == 2

    # 2. Re-index src/parser.py with an updated chunk (and add a new file src/helper.py)
    new_chunks = [
        Chunk(
            content="class RequestParser:\n    def updated_parse(self):\n        pass",
            file_path="src/parser.py",
            language="python",
            start_line=1,
            end_line=3,
            qualified_name="parser.RequestParser",
        ),
        Chunk(
            content="def print_helper(msg):\n    print(msg)",
            file_path="src/helper.py",
            language="python",
            start_line=1,
            end_line=2,
            qualified_name="helper.print_helper",
        ),
    ]

    builder.build_bm25_index(new_chunks)

    # Total doc count should be 3:
    # 1 from src/auth.py (retained)
    # 1 updated from src/parser.py (replaced)
    # 1 new from src/helper.py (added)
    assert builder.bm25_doc_count() == 3

    with open(temp_bm25_path, "rb") as f:
        data = pickle.load(f)
        contents = [c.content for c in data["chunks"]]

        assert "def authenticate_user(token):\n    pass" in contents
        assert (
            "class RequestParser:\n    def updated_parse(self):\n        pass"
            in contents
        )
        assert "def print_helper(msg):\n    print(msg)" in contents
        assert (
            "class RequestParser:\n    def parse_body(self):\n        pass"
            not in contents
        )


# ===========================================================================
# Acceptance Criteria: Semantic & Keyword Search Verification
# ===========================================================================


def test_search_authenticate_returns_auth_chunks(
    temp_bm25_path: str,
    mock_code_embeddings: np.ndarray,
    mock_doc_embeddings: list[DocEmbedding],
) -> None:
    """Unit test: search 'authenticate' returns auth-related chunks in both indices.

    We use a 3-chunk corpus (auth, parser, helper) so that BM25 IDF is non-zero
    for 'authenticate' (which only appears in the auth chunk), giving a positive
    relevance score for the correct result.
    """
    # Three chunks so IDF for 'authenticate' is positive (appears in 1 of 3 docs)
    chunks = [
        Chunk(
            content="def authenticate_user(token):\n    pass",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=2,
            qualified_name="auth.authenticate_user",
        ),
        Chunk(
            content="class RequestParser:\n    def parse_body(self):\n        pass",
            file_path="src/parser.py",
            language="python",
            start_line=1,
            end_line=3,
            qualified_name="parser.RequestParser",
        ),
        Chunk(
            content="def send_email(recipient, subject):\n    pass",
            file_path="src/email_utils.py",
            language="python",
            start_line=1,
            end_line=2,
            qualified_name="email_utils.send_email",
        ),
    ]
    rng = np.random.default_rng(42)
    code_embeddings = rng.standard_normal((3, 768)).astype(np.float32)
    code_embeddings /= np.linalg.norm(code_embeddings, axis=1, keepdims=True)

    builder = IndexBuilder(qdrant_url=":memory:", bm25_path=temp_bm25_path)
    builder.build_vector_index(chunks, code_embeddings, mock_doc_embeddings)
    builder.build_bm25_index(chunks)

    # 1. Test BM25 keyword matching
    with open(temp_bm25_path, "rb") as f:
        data = pickle.load(f)
        bm25_model = data["bm25"]
        stored_chunks = data["chunks"]

        query_tokens = tokenize_code("authenticate")
        scores = bm25_model.get_scores(query_tokens)

        # The highest scoring chunk must be the authenticate_user chunk
        best_idx = int(np.argmax(scores))
        best_chunk = stored_chunks[best_idx]
        assert "authenticate" in best_chunk.content
        assert best_chunk.file_path == "src/auth.py"
        # With >=3 docs, IDF for a term in 1 doc is positive
        assert scores[best_idx] > 0.0

    # 2. Test Vector search (using in-memory QdrantClient)
    # auth.py embedding is index 0; query with it to retrieve the most similar chunk
    query_vector = code_embeddings[0].tolist()

    res = builder.client.search(
        collection_name=settings.qdrant_collection_code,
        query_vector=query_vector,
        limit=1,
    )
    assert len(res) == 1
    payload = res[0].payload
    assert payload is not None
    assert "authenticate" in payload["chunk_text"]
    assert payload["file_path"] == "src/auth.py"

"""Unit tests for the IndexBuilder pipeline."""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

from reporag.config import settings
from reporag.embedding.index_builder import IndexBuilder


@dataclass
class _FakeChunk:
    content: str
    file_path: str = "test.py"
    start_line: int = 1
    end_line: int = 10
    qualified_name: str | None = None
    parent_symbol: str | None = None
    language: str = "python"


@dataclass
class _FakeDocEmbedding:
    symbol_id: str
    text: str
    vector: np.ndarray


@pytest.fixture
def temp_bm25_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "bm25_index.pkl")


@pytest.fixture
def builder(temp_bm25_path):
    b = IndexBuilder(qdrant_url=":memory:")
    b.bm25_path = temp_bm25_path
    return b


def test_code_tokenize(builder):
    tokens = builder.code_tokenize("getUserById")
    assert tokens == ["get", "user", "by", "id"]

    tokens = builder.code_tokenize("snake_case_var")
    assert tokens == ["snake", "case", "var"]

    tokens = builder.code_tokenize("HTTPResponse200")
    assert tokens == ["http", "response", "200"]


def test_build_vector_index(builder):
    chunks = [
        _FakeChunk(
            content="def authenticate_user(): pass", qualified_name="authenticate_user"
        ),
        _FakeChunk(content="class Database: pass", qualified_name="Database"),
    ]

    # Fake 768-dim embeddings for code
    code_embeddings = np.random.randn(2, 768).astype(np.float32)

    # Fake 384-dim embeddings for docs
    doc_embeddings = [
        _FakeDocEmbedding(
            symbol_id="authenticate_user",
            text="Authenticates a user",
            vector=np.random.randn(384).astype(np.float32),
        ),
    ]

    builder.build_vector_index(chunks, code_embeddings, doc_embeddings)

    assert builder.vector_count() == 2
    assert builder.doc_vector_count() == 1

    # Test incremental updates (no duplicates)
    builder.build_vector_index(chunks, code_embeddings, doc_embeddings)
    assert builder.vector_count() == 2
    assert builder.doc_vector_count() == 1


def test_build_bm25_index(builder):
    chunks = [
        _FakeChunk(
            content="def authenticate_user(): pass", qualified_name="authenticate_user"
        ),
        _FakeChunk(content="class Database: pass", qualified_name="Database"),
        _FakeChunk(content="def other_func(): pass", qualified_name="other_func"),
        _FakeChunk(content="class SomethingElse: pass", qualified_name="SomethingElse"),
    ]

    builder.build_bm25_index(chunks)
    assert builder.bm25_doc_count() == 4

    # Check that BM25Okapi is initialized
    assert builder.bm25 is not None

    # Check if we can search for authenticate
    tokenized_query = builder.code_tokenize("authenticate")
    scores = builder.bm25.get_scores(tokenized_query)
    # The first chunk should have a higher score than the second chunk
    assert scores[0] > scores[1]


def test_search_authenticate(builder):
    """Integration style test for searching authenticate."""
    chunks = [
        _FakeChunk(
            content="def authenticate_user(): pass", qualified_name="authenticate_user"
        ),
        _FakeChunk(content="class Database: pass", qualified_name="Database"),
        _FakeChunk(content="def other_func(): pass", qualified_name="other_func"),
        _FakeChunk(content="class SomethingElse: pass", qualified_name="SomethingElse"),
    ]

    # Fake 768-dim embeddings for code
    # Make the first one match the query vector
    query_vector = np.random.randn(768).astype(np.float32)
    code_embeddings = np.array(
        [
            query_vector,
            np.random.randn(768).astype(np.float32),
            np.random.randn(768).astype(np.float32),
            np.random.randn(768).astype(np.float32),
        ]
    )

    builder.build_vector_index(chunks, code_embeddings, None)
    builder.build_bm25_index(chunks)

    # Test vector search
    results = builder.qdrant.search(
        collection_name=settings.qdrant_collection_code,
        query_vector=query_vector.tolist(),
        limit=1,
    )
    assert len(results) == 1
    assert results[0].payload["qualified_name"] == "authenticate_user"

    # Test BM25 search
    tokenized_query = builder.code_tokenize("authenticate")
    top_chunks = builder.bm25.get_top_n(tokenized_query, chunks, n=1)
    assert top_chunks[0].qualified_name == "authenticate_user"

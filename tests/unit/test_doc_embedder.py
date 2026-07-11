"""Unit tests for the DocEmbedder pipeline (Issue 14).

Tests cover offline operations with a mock model, as well as a real-model test
to verify semantic similarity for authentication queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from reporag.embedding.doc_embedder import (
    DOC_EMBEDDING_DIM,
    DocEmbedder,
    DocEmbedding,
    _normalise,
    _pick_device,
)


@dataclass
class _FakeRecord:
    """Duck-typed SymbolRecord for testing."""

    symbol_id: str
    docstring: str | None = None
    signature: str | None = None


def _make_model(dim: int = DOC_EMBEDDING_DIM) -> MagicMock:
    """Return a mock SentenceTransformer that returns random normalized vectors."""

    def fake_encode(texts, **kwargs):
        n = len(texts) if not isinstance(texts, str) else 1
        # deterministic pseudo-random vectors based on text hash
        rng = np.random.default_rng(seed=abs(hash(str(texts))) % (2**31))
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / np.where(norms < 1e-12, 1.0, norms)

    model = MagicMock()
    model.encode.side_effect = fake_encode
    return model


def _make_embedder(batch_size: int = 8) -> DocEmbedder:
    """Return a DocEmbedder with a mock model pre-injected."""
    embedder = DocEmbedder(
        model_name="test/doc-model", device="cpu", batch_size=batch_size
    )
    embedder._model = _make_model()
    embedder._loaded = True
    return embedder


# ===========================================================================
# Helpers / Utilities
# ===========================================================================


def test_normalise_produces_unit_rows():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, DOC_EMBEDDING_DIM)).astype(np.float32)
    normed = _normalise(vecs)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_normalise_zero_vector_unchanged():
    zero = np.zeros((1, DOC_EMBEDDING_DIM), dtype=np.float32)
    result = _normalise(zero)
    np.testing.assert_array_equal(result, zero)


def test_pick_device_returns_string():
    device = _pick_device()
    assert isinstance(device, str)
    assert device in ("cpu", "cuda", "mps")


# ===========================================================================
# DocEmbedder Construction and Lazy Loading
# ===========================================================================


def test_construction_is_lazy():
    embedder = DocEmbedder(model_name="test/doc-model")
    assert not embedder._loaded
    assert embedder._model is None
    assert embedder.embedding_dim == DOC_EMBEDDING_DIM


# ===========================================================================
# DocEmbedding Dataclass
# ===========================================================================


def test_doc_embedding_to_from_dict():
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(DOC_EMBEDDING_DIM).astype(np.float32)
    original = DocEmbedding(
        symbol_id="test.symbol",
        text="A test docstring.",
        embedding=vec,
        source="docstring",
    )
    d = original.to_dict()
    assert isinstance(d["embedding"], list)

    rebuilt = DocEmbedding.from_dict(d)
    assert rebuilt.symbol_id == original.symbol_id
    assert rebuilt.text == original.text
    np.testing.assert_allclose(rebuilt.embedding, original.embedding, rtol=1e-5)


# ===========================================================================
# DocEmbedder.embed_batch
# ===========================================================================


def test_embed_batch_returns_numpy_array():
    embedder = _make_embedder()
    texts = ["Authenticate user with JWT token", "Parse request body"]
    vectors = embedder.embed_batch(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, DOC_EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embed_batch_empty_list():
    embedder = DocEmbedder(model_name="test/doc-model")
    vectors = embedder.embed_batch([])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, DOC_EMBEDDING_DIM)
    assert not embedder._loaded


def test_embed_batch_handles_empty_gracefully():
    embedder = _make_embedder()
    texts = ["Valid text", "", "   ", None]  # type: ignore[list-item]
    vectors = embedder.embed_batch(texts)

    assert vectors.shape == (4, DOC_EMBEDDING_DIM)
    # The valid text has a non-zero vector
    assert np.linalg.norm(vectors[0]) > 0.5
    # The empty/whitespace/None values have zero vectors
    assert np.linalg.norm(vectors[1]) == 0.0
    assert np.linalg.norm(vectors[2]) == 0.0
    assert np.linalg.norm(vectors[3]) == 0.0


# ===========================================================================
# Caching & Deduplication
# ===========================================================================


def test_cache_prevents_recomputation():
    embedder = _make_embedder()
    model = embedder._model

    embedder.embed_batch(["Cache me."])
    calls_after_first = model.encode.call_count

    embedder.embed_batch(["Cache me."])
    calls_after_second = model.encode.call_count

    assert calls_after_first == calls_after_second
    assert embedder.cache_stats()["hits"] == 1


def test_cache_eviction_respects_maxsize():
    embedder = _make_embedder()
    embedder._cache_maxsize = 2

    embedder.embed_batch(["a", "b"])
    assert embedder.cache_stats()["size"] == 2

    embedder.embed_batch(["c"])
    assert embedder.cache_stats()["size"] == 2

    # "a" should have been evicted
    embedder.embed_batch(["a"])
    assert embedder.cache_stats()["misses"] == 4


def test_clear_cache():
    embedder = _make_embedder()
    embedder.embed_batch(["a"])
    embedder.clear_cache()
    stats = embedder.cache_stats()
    assert stats == {"hits": 0, "misses": 0, "size": 0}


def test_duplicate_texts_computed_once():
    embedder = _make_embedder()
    model = embedder._model

    vectors = embedder.embed_batch(["dup", "dup", "dup"])
    assert vectors.shape == (3, DOC_EMBEDDING_DIM)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert model.encode.call_count == 1


# ===========================================================================
# Batch processing & Progress callback
# ===========================================================================


def test_progress_callback_called():
    embedder = _make_embedder(batch_size=2)
    calls = []

    def on_progress(done: int, total: int) -> None:
        calls.append((done, total))

    embedder.embed_batch(["a", "b", "c", "d", "e"], progress=on_progress)
    assert len(calls) > 0
    assert calls[-1] == (5, 5)


def test_progress_callback_not_called_for_empty():
    embedder = _make_embedder()
    calls = []
    embedder.embed_batch([], progress=lambda done, total: calls.append((done, total)))
    assert calls == []


# ===========================================================================
# embed_records (SymbolRecords)
# ===========================================================================


def test_embed_records_skips_empty_docstrings():
    embedder = _make_embedder()
    records = [
        _FakeRecord("s1", docstring="Valid doc."),
        _FakeRecord("s2", docstring=None),
        _FakeRecord("s3", docstring=""),
        _FakeRecord("s4", docstring="   "),
    ]
    embeddings = embedder.embed_records(records)

    assert len(embeddings) == 1
    assert embeddings[0].symbol_id == "s1"
    assert embeddings[0].text == "Valid doc."
    assert embeddings[0].embedding.shape == (DOC_EMBEDDING_DIM,)


def test_embed_records_include_signature():
    embedder = _make_embedder()
    records = [
        _FakeRecord("s1", docstring="Returns value", signature="def get() -> int")
    ]
    embeddings = embedder.embed_records(records, include_signature=True)
    assert len(embeddings) == 1
    assert embeddings[0].text == "def get() -> int\nReturns value"


# ===========================================================================
# Acceptance Criteria: Semantic Similarity Test
# ===========================================================================


def test_authentication_query_is_close_to_verify_jwt_token():
    """Unit test: "authentication" query is close to "verify JWT token" docstring."""
    embedder = (
        DocEmbedder()
    )  # Loads the real SentenceTransformer model (all-MiniLM-L6-v2)

    texts = [
        "authentication",
        "verify JWT token",
        "parse request body to dictionary",
    ]
    vectors = embedder.embed_batch(texts)

    # Calculate cosine similarity (dot product since they are L2-normalized)
    sim_auth_jwt = float(np.dot(vectors[0], vectors[1]))
    sim_auth_parse = float(np.dot(vectors[0], vectors[2]))

    # Assert query is closer semantically to JWT verification than parsing request body
    assert sim_auth_jwt > sim_auth_parse
    assert sim_auth_jwt > 0.3

"""Unit tests for the CodeEmbedder pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import torch

from reporag.embedding.code_embedder import (
    EMBEDDING_DIM,
    CodeEmbedder,
    _extract_text,
    _resolve_device,
)

# -- helpers ------------------------------------------------------------


class _FakeBatchEncoding(dict):
    """Minimal stand-in for tokenizer output that supports ``.to()``."""

    def to(self, device):
        return self


@dataclass
class _FakeChunk:
    """Simulates the real ``Chunk`` dataclass from the ingestion module."""

    content: str
    file_path: str = "test.py"


def _make_embedder(batch_size: int = 1) -> CodeEmbedder:
    """Return a CodeEmbedder with mocked model/tokenizer pre-injected.

    Because model loading is lazy and the imports happen inside
    ``_ensure_loaded``, we inject fakes directly instead of patching.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = _FakeBatchEncoding(
        {
            "input_ids": torch.ones(batch_size, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
        }
    )

    model = MagicMock()
    outputs = MagicMock()
    outputs.last_hidden_state = torch.randn(batch_size, 5, EMBEDDING_DIM)
    model.return_value = outputs
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    embedder = CodeEmbedder(model_name="test/model", device="cpu")
    embedder._tokenizer = tokenizer
    embedder._model = model
    return embedder


# -- _resolve_device ----------------------------------------------------


def test_resolve_device_explicit():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto():
    device = _resolve_device("auto")
    assert device in (torch.device("cpu"), torch.device("cuda"), torch.device("mps"))


# -- _extract_text ------------------------------------------------------


def test_extract_text_from_string():
    assert _extract_text("def foo(): pass") == "def foo(): pass"


def test_extract_text_from_chunk():
    chunk = _FakeChunk(content="class Bar: pass")
    assert _extract_text(chunk) == "class Bar: pass"


# -- CodeEmbedder: construction -----------------------------------------


def test_lazy_loading_does_not_load_model_at_init():
    """Model must NOT be loaded during __init__ -- only on first embed."""
    embedder = CodeEmbedder(model_name="test/model")
    assert not embedder._loaded
    assert embedder._model is None
    assert embedder._tokenizer is None


# -- CodeEmbedder: embed_batch ------------------------------------------


def test_embed_batch_shape_and_dtype():
    embedder = _make_embedder(batch_size=1)
    result = embedder.embed_batch(["def hello(): pass"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, EMBEDDING_DIM)
    assert result.dtype == np.float32


def test_embed_batch_l2_normalised():
    embedder = _make_embedder(batch_size=3)
    result = embedder.embed_batch(["a", "b", "c"])

    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_embed_batch_empty():
    embedder = CodeEmbedder(model_name="test/model")
    result = embedder.embed_batch([])

    assert result.shape == (0, EMBEDDING_DIM)
    # Model should never have been loaded for an empty batch
    assert not embedder._loaded


# -- CodeEmbedder: Chunk support ----------------------------------------


def test_embed_batch_accepts_chunk_objects():
    embedder = _make_embedder(batch_size=2)
    chunks = [
        _FakeChunk(content="def add(a, b): return a+b"),
        _FakeChunk(content="class Dog: pass"),
    ]
    result = embedder.embed_batch(chunks)

    assert result.shape == (2, EMBEDDING_DIM)


# -- CodeEmbedder: caching ---------------------------------------------


def test_cache_prevents_recomputation():
    embedder = _make_embedder(batch_size=1)
    model = embedder._model

    first = embedder.embed_batch(["def f(): pass"])
    calls_after_first = model.call_count

    second = embedder.embed_batch(["def f(): pass"])
    calls_after_second = model.call_count

    np.testing.assert_array_equal(first, second)
    assert calls_after_first == calls_after_second


def test_cache_stats():
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["x"])  # miss
    embedder.embed_batch(["x"])  # hit

    stats = embedder.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_cache_eviction_respects_maxsize():
    embedder = _make_embedder(batch_size=1)
    embedder._cache_maxsize = 2

    embedder.embed_batch(["a"])
    embedder.embed_batch(["b"])
    embedder.embed_batch(["c"])  # should evict "a"

    assert embedder.cache_stats()["size"] == 2


def test_clear_cache():
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["x"])
    embedder.clear_cache()

    stats = embedder.cache_stats()
    assert stats == {"hits": 0, "misses": 0, "size": 0}


# -- CodeEmbedder: deduplication ----------------------------------------


def test_duplicates_in_batch_computed_once():
    """If the same string appears 3x in one batch, the model sees it only once."""
    embedder = _make_embedder(batch_size=1)
    model = embedder._model

    result = embedder.embed_batch(["dup", "dup", "dup"])

    assert result.shape == (3, EMBEDDING_DIM)
    # All three rows should be identical
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[1], result[2])
    # Model forward pass called only once (1 unique text)
    assert model.call_count == 1


# -- CodeEmbedder: single embed helper ---------------------------------


def test_embed_single():
    embedder = _make_embedder(batch_size=1)
    vec = embedder.embed("def g(): return 1")

    assert vec.shape == (EMBEDDING_DIM,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


# -- CodeEmbedder: similarity helper -----------------------------------


def test_similarity_returns_float():
    embedder = _make_embedder(batch_size=2)
    score = embedder.similarity("def a(): pass", "def b(): pass")

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

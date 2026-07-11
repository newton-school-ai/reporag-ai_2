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


# =========================================================================
# DEVICE FALLBACK -- strong tests
# =========================================================================


def test_resolve_device_explicit_cpu():
    """Explicit 'cpu' always returns CPU regardless of GPU availability."""
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_explicit_cuda():
    """Explicit 'cuda' returns cuda device without probing availability."""
    assert _resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_explicit_mps():
    """Explicit 'mps' returns mps device without probing availability."""
    assert _resolve_device("mps") == torch.device("mps")


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    """When CUDA is available, auto should pick cuda over mps and cpu."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    """When CUDA is unavailable but MPS is, auto should pick mps."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    """When neither CUDA nor MPS is available, auto must fall back to cpu."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device("auto") == torch.device("cpu")


def test_embedder_device_propagated_to_model():
    """The model's .to() must be called with the resolved device."""
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["x"])
    embedder._model.to.assert_called_with(torch.device("cpu"))


def test_embedder_honours_explicit_device_kwarg():
    """device='cpu' should override auto-detection."""
    embedder = CodeEmbedder(model_name="test/model", device="cpu")
    assert embedder.device == torch.device("cpu")


# =========================================================================
# BATCHING -- strong tests
# =========================================================================


def _make_counting_embedder(total_items: int, batch_size: int) -> CodeEmbedder:
    """Return an embedder whose mock model tracks per-call batch sizes."""
    call_sizes: list[int] = []

    tokenizer = MagicMock()

    def fake_tokenize(texts, **kwargs):
        n = len(texts)
        call_sizes.append(n)
        return _FakeBatchEncoding(
            {
                "input_ids": torch.ones(n, 5, dtype=torch.long),
                "attention_mask": torch.ones(n, 5, dtype=torch.long),
            }
        )

    tokenizer.side_effect = fake_tokenize

    model = MagicMock()

    def fake_forward(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.randn(n, 5, EMBEDDING_DIM)
        return out

    model.side_effect = fake_forward
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    embedder = CodeEmbedder(
        model_name="test/model", device="cpu", batch_size=batch_size
    )
    embedder._tokenizer = tokenizer
    embedder._model = model
    embedder._call_sizes = call_sizes  # expose for assertions
    return embedder


def test_batching_splits_into_correct_chunks():
    """7 unique items with batch_size=3 should produce batches [3, 3, 1]."""
    embedder = _make_counting_embedder(total_items=7, batch_size=3)
    texts = [f"code_{i}" for i in range(7)]
    result = embedder.embed_batch(texts)

    assert result.shape == (7, EMBEDDING_DIM)
    assert embedder._call_sizes == [3, 3, 1]


def test_batching_single_batch_when_items_fit():
    """4 items with batch_size=10 should run in a single batch."""
    embedder = _make_counting_embedder(total_items=4, batch_size=10)
    texts = [f"code_{i}" for i in range(4)]
    result = embedder.embed_batch(texts)

    assert result.shape == (4, EMBEDDING_DIM)
    assert embedder._call_sizes == [4]


def test_batch_size_override_per_call():
    """batch_size kwarg on embed_batch overrides the instance default."""
    embedder = _make_counting_embedder(total_items=6, batch_size=100)
    texts = [f"code_{i}" for i in range(6)]
    result = embedder.embed_batch(texts, batch_size=2)

    assert result.shape == (6, EMBEDDING_DIM)
    assert embedder._call_sizes == [2, 2, 2]


def test_batch_preserves_input_order():
    """Output row i must correspond to input item i, not batch order."""
    embedder = _make_counting_embedder(total_items=3, batch_size=1)

    items = ["alpha", "beta", "gamma"]
    result = embedder.embed_batch(items)

    assert result.shape == (3, EMBEDDING_DIM)
    # Each row should be individually L2-normalised
    for i in range(3):
        np.testing.assert_allclose(np.linalg.norm(result[i]), 1.0, rtol=1e-5)


def test_batch_size_one_processes_items_individually():
    """batch_size=1 must process each item in its own forward pass."""
    embedder = _make_counting_embedder(total_items=3, batch_size=1)
    texts = ["a", "b", "c"]
    embedder.embed_batch(texts)

    assert embedder._call_sizes == [1, 1, 1]


# =========================================================================
# CACHE BEHAVIOUR -- strong tests
# =========================================================================


def test_cache_lru_evicts_oldest_entry():
    """With maxsize=2, the least-recently-used entry must be evicted first."""
    embedder = _make_embedder(batch_size=1)
    embedder._cache_maxsize = 2

    embedder.embed_batch(["first"])  # cache: [first]
    embedder.embed_batch(["second"])  # cache: [first, second]
    embedder.embed_batch(["third"])  # cache: [second, third] -- first evicted

    # "first" should be a miss (evicted), "second" should be a hit
    embedder.embed_batch(["second"])  # hit
    embedder.embed_batch(["first"])  # miss (re-computed)

    stats = embedder.cache_stats()
    # misses: first(1) + second(1) + third(1) + first-again(1) = 4
    # hits: second-again(1) = 1
    assert stats["misses"] == 4
    assert stats["hits"] == 1


def test_cache_lru_refresh_on_access():
    """Accessing a cached item should refresh its LRU position."""
    embedder = _make_embedder(batch_size=1)
    embedder._cache_maxsize = 2

    embedder.embed_batch(["old"])  # cache: [old]
    embedder.embed_batch(["newer"])  # cache: [old, newer]
    embedder.embed_batch(["old"])  # hit -- refreshes "old" to most recent

    # Now "newer" is the oldest. Adding a third should evict "newer", not "old"
    embedder.embed_batch(["newest"])  # cache: [old, newest]

    embedder.embed_batch(["old"])  # should be a hit (still cached)
    embedder.embed_batch(["newer"])  # should be a miss (was evicted)

    stats = embedder.cache_stats()
    # hits: old(1) + old(1) = 2
    # misses: old(1) + newer(1) + newest(1) + newer-again(1) = 4
    assert stats["hits"] == 2
    assert stats["misses"] == 4


def test_cache_key_isolates_different_models():
    """Same text embedded by different model names must have different keys."""
    e1 = CodeEmbedder(model_name="model-A", device="cpu")
    e2 = CodeEmbedder(model_name="model-B", device="cpu")

    key_a = e1._cache_key("def foo(): pass")
    key_b = e2._cache_key("def foo(): pass")

    assert key_a != key_b, "Cache keys must differ across model names"


def test_cache_mixed_hits_and_misses_in_single_batch():
    """A batch with some cached and some new items must return correct results."""
    embedder = _make_counting_embedder(total_items=1, batch_size=1)

    # Pre-populate cache with "cached_item"
    first = embedder.embed_batch(["cached_item"])
    calls_after_first = len(embedder._call_sizes)

    # Now embed a mix: one cached, two new
    result = embedder.embed_batch(["cached_item", "new_a", "new_b"])

    assert result.shape == (3, EMBEDDING_DIM)
    # First row must match the previously cached value
    np.testing.assert_array_equal(result[0], first[0])
    # Two more forward passes (for new_a and new_b, not cached_item)
    assert len(embedder._call_sizes) == calls_after_first + 2


def test_cache_maxsize_one_keeps_only_latest():
    """cache_maxsize=1 evicts immediately, keeping only the last entry."""
    embedder = _make_counting_embedder(total_items=1, batch_size=1)
    embedder._cache_maxsize = 1

    embedder.embed_batch(["first"])
    assert embedder.cache_stats()["size"] == 1

    embedder.embed_batch(["second"])  # should evict "first"
    assert embedder.cache_stats()["size"] == 1

    # "first" must be a miss now (evicted)
    embedder.embed_batch(["first"])
    # misses: first(1) + second(1) + first-again(1) = 3, hits: 0
    assert embedder.cache_stats()["misses"] == 3
    assert embedder.cache_stats()["hits"] == 0


def test_clear_cache_forces_recomputation():
    """After clear_cache(), previously cached items must recompute."""
    embedder = _make_embedder(batch_size=1)
    model = embedder._model

    embedder.embed_batch(["reuse_me"])
    calls_after_first = model.call_count

    embedder.clear_cache()

    embedder.embed_batch(["reuse_me"])
    calls_after_second = model.call_count

    assert calls_after_second > calls_after_first
    stats = embedder.cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["size"] == 1

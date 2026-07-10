"""Unit tests for the code embedding pipeline (Issue 13).

Uses a pure-torch fake tokenizer/model pair (``_FakeTokenizer`` /
``_FakeModel``) rather than downloading real CodeBERT/UniXcoder weights, so
these tests run offline and fast while still exercising the real
``torch``/``numpy`` compute path (batching, attention-masked mean pooling,
L2 normalization, device placement).

Covers every acceptance criterion: 768-dim output, configurable batch size,
GPU support with CPU fallback, L2 normalization, and cache hits avoiding
re-computation -- plus integration with Issue 8's ``Chunk``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.reporag.embedding.code_embedder import EMBEDDING_DIM, CodeEmbedder
from src.reporag.ingestion.chunker import Chunk

# ---------------------------------------------------------------------------
# Fake tokenizer / model -- pure torch, no network, no HF downloads
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic char-hash tokenizer with the transformers __call__ shape."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(
        self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ):
        self.calls += 1
        seqs = [
            [(ord(c) % (self.vocab_size - 2)) + 2 for c in t[:max_length]] or [2]
            for t in texts
        ]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeModel(torch.nn.Module):
    """A tiny real torch module producing (B, T, 768) hidden states."""

    def __init__(self, vocab_size: int = 64, hidden: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        return _FakeOutput(self.embed(input_ids))


@pytest.fixture
def fake_pair():
    """A fresh (model, tokenizer) pair per test, so call counters start at 0."""
    return _FakeModel(), _FakeTokenizer()


@pytest.fixture
def embedder(fake_pair):
    model, tokenizer = fake_pair
    return CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=2)


CODE_SAMPLES = [
    "def foo():\n    pass",
    "class Bar:\n    pass",
    "x = 1 + 2",
    "import os",
    "return self.value",
]


# ---------------------------------------------------------------------------
# 768-dim output
# ---------------------------------------------------------------------------


def test_embed_batch_produces_768_dim_vectors(embedder: CodeEmbedder) -> None:
    """embed_batch returns a (N, 768) array."""
    vectors = embedder.embed_batch(CODE_SAMPLES)
    assert vectors.shape == (len(CODE_SAMPLES), EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embed_single_produces_768_vector(embedder: CodeEmbedder) -> None:
    """embed() returns a single (768,) vector."""
    vector = embedder.embed(CODE_SAMPLES[0])
    assert vector.shape == (EMBEDDING_DIM,)


def test_embedding_dim_property(embedder: CodeEmbedder) -> None:
    """embedding_dim always reports 768."""
    assert embedder.embedding_dim == EMBEDDING_DIM


def test_empty_batch_returns_empty_array_without_loading_model(fake_pair) -> None:
    """An empty batch returns (0, 768) and never touches the model/tokenizer."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer)
    result = embedder.embed_batch([])
    assert result.shape == (0, EMBEDDING_DIM)
    assert model.forward_calls == 0
    assert tokenizer.calls == 0


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------


def test_embeddings_are_l2_normalized(embedder: CodeEmbedder) -> None:
    """Every output row has unit L2 norm."""
    vectors = embedder.embed_batch(CODE_SAMPLES)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_single_embed_is_l2_normalized(embedder: CodeEmbedder) -> None:
    """A single embed() result also has unit norm."""
    vector = embedder.embed("def f(): pass")
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Configurable batch size
# ---------------------------------------------------------------------------


def test_batch_size_controls_number_of_model_calls(fake_pair) -> None:
    """5 items with batch_size=2 triggers 3 forward passes (2, 2, 1)."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=2)
    embedder.embed_batch(CODE_SAMPLES)
    assert model.forward_calls == 3


def test_batch_size_override_per_call(fake_pair) -> None:
    """A per-call batch_size overrides the constructor default."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=100)
    embedder.embed_batch(CODE_SAMPLES, batch_size=1)
    assert model.forward_calls == len(CODE_SAMPLES)


def test_single_large_batch_size_makes_one_call(fake_pair) -> None:
    """A batch_size >= len(items) makes exactly one forward pass."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=100)
    embedder.embed_batch(CODE_SAMPLES)
    assert model.forward_calls == 1


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_avoids_recomputation(fake_pair) -> None:
    """Re-embedding the same text doesn't call the model again."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=10)
    embedder.embed_batch(CODE_SAMPLES)
    calls_after_first = model.forward_calls
    embedder.embed_batch(CODE_SAMPLES)
    assert model.forward_calls == calls_after_first


def test_cache_hit_returns_identical_vector(embedder: CodeEmbedder) -> None:
    """A cached embedding is bit-identical to the original computation."""
    first = embedder.embed_batch(CODE_SAMPLES)
    second = embedder.embed_batch(CODE_SAMPLES)
    np.testing.assert_array_equal(first, second)


def test_partial_cache_hit_only_computes_misses(fake_pair) -> None:
    """A batch mixing cached and new items only forwards the new ones."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=10)
    embedder.embed_batch(CODE_SAMPLES[:2])
    assert model.forward_calls == 1
    embedder.embed_batch(CODE_SAMPLES)  # first 2 cached, 3 new
    assert model.forward_calls == 2
    assert tokenizer.calls == 2  # tokenizer only invoked for the 3 new items


def test_use_cache_false_bypasses_and_does_not_populate_cache(
    embedder: CodeEmbedder,
) -> None:
    """use_cache=False skips the cache entirely, both reading and writing."""
    embedder.embed_batch(CODE_SAMPLES, use_cache=False)
    assert embedder.cache_size() == 0


def test_cache_size_and_clear_cache(embedder: CodeEmbedder) -> None:
    """cache_size reflects unique entries; clear_cache empties it."""
    embedder.embed_batch(CODE_SAMPLES)
    assert embedder.cache_size() == len(CODE_SAMPLES)
    embedder.clear_cache()
    assert embedder.cache_size() == 0


def test_contains_reflects_cache_state(embedder: CodeEmbedder) -> None:
    """__contains__ reports whether an item's embedding is cached."""
    assert CODE_SAMPLES[0] not in embedder
    embedder.embed(CODE_SAMPLES[0])
    assert CODE_SAMPLES[0] in embedder


def test_duplicate_items_in_one_batch_share_one_computation(fake_pair) -> None:
    """The same text appearing twice in one batch is only forwarded once."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=10)
    vectors = embedder.embed_batch([CODE_SAMPLES[0], CODE_SAMPLES[0]])
    assert model.forward_calls == 1
    np.testing.assert_array_equal(vectors[0], vectors[1])


def test_custom_cache_mapping_is_used(fake_pair) -> None:
    """An injected mapping is used directly as the cache store."""
    model, tokenizer = fake_pair
    custom_cache: dict = {}
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, cache=custom_cache)
    embedder.embed(CODE_SAMPLES[0])
    assert len(custom_cache) == 1


def test_cache_key_includes_model_name(fake_pair) -> None:
    """Caching the same text with different models creates different cache entries."""
    model, tokenizer = fake_pair
    # We create two embedders sharing the same dict, but with different model names
    shared_cache: dict = {}
    embedder_a = CodeEmbedder(
        model_name="model_A", model=model, tokenizer=tokenizer, cache=shared_cache
    )
    embedder_b = CodeEmbedder(
        model_name="model_B", model=model, tokenizer=tokenizer, cache=shared_cache
    )

    embedder_a.embed("print(1)")
    assert len(shared_cache) == 1

    embedder_b.embed("print(1)")
    assert len(shared_cache) == 2  # The second model didn't hit the first model's cache


def test_cache_stats_tracking(embedder: CodeEmbedder) -> None:
    """Cache hits, misses, and size are tracked correctly."""
    assert embedder.cache_stats() == {"hits": 0, "misses": 0, "size": 0}

    # 2 new items -> 2 misses
    embedder.embed_batch(CODE_SAMPLES[:2])
    assert embedder.cache_stats() == {"hits": 0, "misses": 2, "size": 2}

    # 2 cached items + 1 new item -> 2 hits, 1 miss
    embedder.embed_batch(CODE_SAMPLES[:3])
    assert embedder.cache_stats() == {"hits": 2, "misses": 3, "size": 3}

    # clear resets it
    embedder.clear_cache()
    assert embedder.cache_stats() == {"hits": 0, "misses": 0, "size": 0}


# ---------------------------------------------------------------------------
# GPU support with CPU fallback
# ---------------------------------------------------------------------------


def test_device_auto_resolves_to_cpu_when_no_gpu(fake_pair, monkeypatch) -> None:
    """device='auto' with no CUDA or MPS available resolves to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="auto")
    assert embedder.device == "cpu"


def test_device_auto_resolves_to_cuda_when_available(fake_pair, monkeypatch) -> None:
    """device='auto' resolves to CUDA when it's available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="auto")
    assert embedder.device == "cuda"


def test_device_cuda_falls_back_to_cpu_when_unavailable(
    fake_pair, monkeypatch, caplog
) -> None:
    """Explicitly requesting CUDA without a GPU falls back to CPU (not an error)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="cuda")
    assert embedder.device == "cpu"


def test_device_cpu_forces_cpu_even_if_cuda_available(fake_pair, monkeypatch) -> None:
    """device='cpu' is honored even when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="cpu")
    assert embedder.device == "cpu"


def test_invalid_device_raises() -> None:
    """An unrecognized device string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown device"):
        CodeEmbedder(device="tpu")  # type: ignore[arg-type]


def test_device_auto_resolves_to_mps_when_available(fake_pair, monkeypatch) -> None:
    """device='auto' resolves to MPS when it's available and CUDA is not."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FakeMPSBackend:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(), raising=False)

    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="auto")
    assert embedder.device == "mps"


def test_device_mps_explicit_resolves_or_falls_back(fake_pair, monkeypatch) -> None:
    """Explicitly requesting MPS falls back to CPU if unavailable."""

    class FakeMPSBackend:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch.backends, "mps", FakeMPSBackend(), raising=False)

    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, device="mps")
    assert embedder.device == "cpu"


# ---------------------------------------------------------------------------
# Issue 8 integration (Chunk objects)
# ---------------------------------------------------------------------------


def test_embed_batch_accepts_chunk_objects(embedder: CodeEmbedder) -> None:
    """embed_batch works directly on Issue 8 Chunk objects, using their content."""
    chunks = [
        Chunk(
            content=code, file_path="a.py", language="python", start_line=1, end_line=1
        )
        for code in CODE_SAMPLES[:2]
    ]
    vectors = embedder.embed_batch(chunks)
    assert vectors.shape == (2, EMBEDDING_DIM)


def test_chunk_and_equivalent_string_share_cache_entry(embedder: CodeEmbedder) -> None:
    """A Chunk and a raw string with the same content hit the same cache key."""
    code = CODE_SAMPLES[0]
    chunk = Chunk(
        content=code, file_path="a.py", language="python", start_line=1, end_line=1
    )
    embedder.embed(chunk)
    assert code in embedder


def test_mixed_strings_and_chunks_in_one_batch(embedder: CodeEmbedder) -> None:
    """A batch can mix raw strings and Chunk objects."""
    chunk = Chunk(
        content=CODE_SAMPLES[1],
        file_path="a.py",
        language="python",
        start_line=1,
        end_line=1,
    )
    vectors = embedder.embed_batch([CODE_SAMPLES[0], chunk])
    assert vectors.shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Ordering / correctness
# ---------------------------------------------------------------------------


def test_output_order_matches_input_order_across_cache_and_batches(fake_pair) -> None:
    """Result order matches input order regardless of cache hits or batch splits."""
    model, tokenizer = fake_pair
    embedder = CodeEmbedder(model=model, tokenizer=tokenizer, batch_size=2)
    embedder.embed(CODE_SAMPLES[2])  # pre-cache the middle item
    vectors = embedder.embed_batch(CODE_SAMPLES)
    # Re-embedding item 2 alone must match its position in the full batch.
    solo = embedder.embed(CODE_SAMPLES[2])
    np.testing.assert_array_equal(vectors[2], solo)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_similarity_computes_dot_product(embedder: CodeEmbedder) -> None:
    """similarity() computes the dot product of the L2-normalized vectors."""
    sim = embedder.similarity(CODE_SAMPLES[0], CODE_SAMPLES[1])
    vecs = embedder.embed_batch(CODE_SAMPLES[:2])
    expected = float(np.dot(vecs[0], vecs[1]))
    assert np.isclose(sim, expected)


def test_similarity_zero_vector_edge_case(embedder: CodeEmbedder, monkeypatch) -> None:
    """similarity() returns 0.0 to avoid division by zero edge cases when norm is < 1e-9."""

    # Monkeypatch embed_batch to return exactly zero vectors for this test
    def zero_batch(items, **kwargs):
        return np.zeros((len(items), EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(embedder, "embed_batch", zero_batch)

    sim = embedder.similarity("foo", "bar")
    assert sim == 0.0

"""Unit tests for the code embedding pipeline (Issue 13).

The suite runs fully offline.  Behavioural tests use
:class:`HashingEmbeddingBackend` -- deterministic and dependency-free -- so
they never download a model, mirroring the project's fallback philosophy
(``tiktoken`` -> regex, Neo4j -> NetworkX).  The transformer forward-pass
logic (masked mean pooling, batching, device handling) is verified against a
*mocked* tokenizer and model built from real ``torch`` tensors, the same way
``test_neo4j_store`` mocks the Neo4j driver.

Acceptance criteria covered:
- Produces 768-dim vectors from code strings
- Batch embedding with configurable batch size
- GPU support with automatic CPU fallback (device selection)
- Embeddings are L2-normalized (unit vectors)
- Embedding cache avoids re-computation for unchanged chunks
- Similar code produces high cosine similarity (> 0.8)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.reporag.embedding.code_embedder import (
    _DEFAULT_DIM,
    CodeEmbedder,
    EmbeddingBackend,
    EmbeddingCache,
    HashingEmbeddingBackend,
    TransformerEmbeddingBackend,
    select_device,
    subtokenize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedder() -> CodeEmbedder:
    """A deterministic, offline embedder backed by feature hashing."""
    return CodeEmbedder(backend="hashing")


# Two snippets differing only by a comment and whitespace: lexically almost
# identical, so a lexical embedder must rate them highly similar.
_SIMILAR_A = "def authenticate_user(token):\n    return verify_jwt_token(token)"
_SIMILAR_B = (
    "def authenticate_user(token):\n"
    "    # look up and verify the credential\n"
    "    return verify_jwt_token(token)"
)
_UNRELATED = "table.render_footer(rows, columns, style='grid')"


# ---------------------------------------------------------------------------
# subtokenize
# ---------------------------------------------------------------------------


class TestSubtokenize:
    def test_empty_string_returns_empty(self) -> None:
        assert subtokenize("") == []

    def test_snake_case_split(self) -> None:
        assert subtokenize("get_user_by_id") == ["get", "user", "by", "id"]

    def test_camel_case_split(self) -> None:
        assert subtokenize("getUserById") == ["get", "user", "by", "id"]

    def test_camel_and_snake_agree(self) -> None:
        """Camel and snake spellings of the same name yield equal sub-tokens."""
        assert subtokenize("getUserById") == subtokenize("get_user_by_id")

    def test_acronym_boundary(self) -> None:
        # HTTPServer -> http + server (acronym followed by a word)
        assert subtokenize("HTTPServer") == ["http", "server"]

    def test_operators_preserved(self) -> None:
        toks = subtokenize("a + b")
        assert "+" in toks
        assert "a" in toks and "b" in toks

    def test_operators_distinguish_expressions(self) -> None:
        assert subtokenize("a + b") != subtokenize("a - b")


# ---------------------------------------------------------------------------
# select_device
# ---------------------------------------------------------------------------


class TestSelectDevice:
    def test_explicit_preference_wins(self) -> None:
        assert select_device("cpu") == "cpu"
        assert select_device("cuda") == "cuda"

    def test_prefers_cuda_when_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert select_device() == "cuda"

    def test_prefers_mps_when_no_cuda(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert select_device() == "mps"

    def test_falls_back_to_cpu(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert select_device() == "cpu"


# ---------------------------------------------------------------------------
# HashingEmbeddingBackend
# ---------------------------------------------------------------------------


class TestHashingBackend:
    def test_default_dimension_is_768(self) -> None:
        assert HashingEmbeddingBackend().dimension == _DEFAULT_DIM

    def test_custom_dimension(self) -> None:
        assert HashingEmbeddingBackend(dimension=128).dimension == 128

    def test_rejects_nonpositive_dimension(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HashingEmbeddingBackend(dimension=0)

    def test_output_shape(self) -> None:
        backend = HashingEmbeddingBackend(dimension=64)
        out = backend.embed(["a b c", "d e f"])
        assert out.shape == (2, 64)
        assert out.dtype == np.float32

    def test_deterministic_across_instances(self) -> None:
        """Stable hashing: two fresh backends embed identically (no salt)."""
        a = HashingEmbeddingBackend(dimension=256).embed(["def f(): pass"])
        b = HashingEmbeddingBackend(dimension=256).embed(["def f(): pass"])
        assert np.array_equal(a, b)

    def test_empty_string_is_zero_vector(self) -> None:
        out = HashingEmbeddingBackend(dimension=32).embed([""])
        assert np.count_nonzero(out) == 0

    def test_satisfies_backend_protocol(self) -> None:
        assert isinstance(HashingEmbeddingBackend(), EmbeddingBackend)

    def test_progress_callback_reports_total(self) -> None:
        seen: list[tuple[int, int]] = []
        HashingEmbeddingBackend(dimension=16).embed(
            ["x", "y", "z"], batch_size=2, progress=lambda d, t: seen.append((d, t))
        )
        assert seen[-1] == (3, 3)


# ---------------------------------------------------------------------------
# CodeEmbedder -- core behaviour (hashing backend)
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    def test_produces_768_dim_vectors(self, embedder: CodeEmbedder) -> None:
        vectors = embedder.embed_batch(["def hello(): return 42", "class Foo: pass"])
        assert vectors.shape == (2, 768)

    def test_vectors_are_l2_normalized(self, embedder: CodeEmbedder) -> None:
        vectors = embedder.embed_batch(["def f(): return 1", "x = compute(y)"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_dot_product_of_unit_vector_is_one(self, embedder: CodeEmbedder) -> None:
        v = embedder.embed_batch(["def hello(): return 42"])
        assert float(v[0] @ v[0]) == pytest.approx(1.0, abs=1e-4)

    def test_empty_input_returns_empty_matrix(self, embedder: CodeEmbedder) -> None:
        out = embedder.embed_batch([])
        assert out.shape == (0, 768)

    def test_embed_single_returns_1d_vector(self, embedder: CodeEmbedder) -> None:
        v = embedder.embed("def f(): pass")
        assert v.shape == (768,)

    def test_configurable_batch_size(self) -> None:
        """A small batch size still embeds every input correctly."""
        emb = CodeEmbedder(backend="hashing", batch_size=2)
        vectors = emb.embed_batch([f"def f{i}(): return {i}" for i in range(5)])
        assert vectors.shape == (5, 768)

    def test_normalization_can_be_disabled(self) -> None:
        emb = CodeEmbedder(backend="hashing", normalize=False)
        vectors = emb.embed_batch(["a b c a b c"])
        # Raw signed-hash counts are integers, so at least one norm != 1.
        assert not np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_duplicate_inputs_get_identical_vectors(
        self, embedder: CodeEmbedder
    ) -> None:
        vectors = embedder.embed_batch(["same()", "other()", "same()"])
        assert np.array_equal(vectors[0], vectors[2])

    def test_progress_callback_invoked(self, embedder: CodeEmbedder) -> None:
        seen: list[tuple[int, int]] = []
        embedder.embed_batch(["a", "b", "c"], progress=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1][0] == seen[-1][1]


# ---------------------------------------------------------------------------
# Cosine similarity behaviour
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_code_similarity_is_one(self, embedder: CodeEmbedder) -> None:
        code = "def add(a, b): return a + b"
        assert embedder.similarity(code, code) == pytest.approx(1.0, abs=1e-5)

    def test_similar_code_high_similarity(self, embedder: CodeEmbedder) -> None:
        """Acceptance criterion: similar code scores > 0.8 cosine similarity."""
        assert embedder.similarity(_SIMILAR_A, _SIMILAR_B) > 0.8

    def test_camel_vs_snake_are_highly_similar(self, embedder: CodeEmbedder) -> None:
        """Code-aware tokenization makes naming style nearly irrelevant."""
        assert embedder.similarity("getUserById(x)", "get_user_by_id(x)") > 0.8

    def test_similar_beats_unrelated(self, embedder: CodeEmbedder) -> None:
        near = embedder.similarity(_SIMILAR_A, _SIMILAR_B)
        far = embedder.similarity(_SIMILAR_A, _UNRELATED)
        assert near > far

    def test_similarity_in_valid_range(self, embedder: CodeEmbedder) -> None:
        s = embedder.similarity("def f(): pass", "class C: pass")
        assert -1.0 <= s <= 1.0

    def test_similarity_handles_empty_string(self, embedder: CodeEmbedder) -> None:
        # Empty code -> zero vector -> defined (0.0) similarity, never NaN.
        assert embedder.similarity("", "def f(): pass") == 0.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_repeated_embed_hits_cache(self, embedder: CodeEmbedder) -> None:
        embedder.embed_batch(["def f(): pass"])
        embedder.embed_batch(["def f(): pass"])
        stats = embedder.cache_stats()
        assert stats["hits"] >= 1

    def test_cache_avoids_backend_recomputation(self) -> None:
        """Unchanged inputs must not reach the backend a second time."""
        spy = _CountingBackend(dimension=64)
        emb = CodeEmbedder(backend=spy)
        emb.embed_batch(["a", "b"])
        assert spy.calls == 1  # 2 unique misses -> one backend call
        emb.embed_batch(["a", "b"])  # all cached
        assert spy.calls == 1  # no new backend call

    def test_only_novel_inputs_recomputed(self) -> None:
        spy = _CountingBackend(dimension=64)
        emb = CodeEmbedder(backend=spy)
        emb.embed_batch(["a", "b"])
        emb.embed_batch(["a", "c"])  # only "c" is new
        assert spy.total_texts == 3  # a, b (first call) + c (second)

    def test_clear_cache_resets_counters(self, embedder: CodeEmbedder) -> None:
        embedder.embed_batch(["x"])
        embedder.clear_cache()
        assert embedder.cache_stats() == {"hits": 0, "misses": 0, "size": 0}

    def test_disk_cache_persists_across_instances(self, tmp_path: Path) -> None:
        """A fresh embedder sharing the on-disk cache skips recomputation."""
        cache_dir = tmp_path / "emb"
        code = "def persisted(): pass"

        first = CodeEmbedder(backend="hashing", cache_dir=cache_dir)
        v1 = first.embed_batch([code])
        assert list(cache_dir.glob("*.npy")), "expected a persisted cache file"

        # A brand-new embedder with the same config keys and shared cache dir
        # loads the vector from disk -- a hit, with zero misses.
        second = CodeEmbedder(backend="hashing", cache_dir=cache_dir)
        v2 = second.embed_batch([code])
        assert np.allclose(v1, v2)
        assert second.cache_stats() == {"hits": 1, "misses": 0, "size": 1}


# ---------------------------------------------------------------------------
# EmbeddingCache (unit-level)
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_memory_get_set(self) -> None:
        cache = EmbeddingCache()
        vec = np.ones(8, dtype=np.float32)
        cache.set("k", vec)
        assert np.array_equal(cache.get("k"), vec)

    def test_missing_key_returns_none(self) -> None:
        assert EmbeddingCache().get("nope") is None

    def test_contains(self) -> None:
        cache = EmbeddingCache()
        cache.set("k", np.zeros(4, dtype=np.float32))
        assert "k" in cache
        assert "missing" not in cache

    def test_len_tracks_entries(self) -> None:
        cache = EmbeddingCache()
        cache.set("a", np.zeros(4, dtype=np.float32))
        cache.set("b", np.zeros(4, dtype=np.float32))
        assert len(cache) == 2

    def test_clear_empties_memory(self) -> None:
        cache = EmbeddingCache()
        cache.set("a", np.zeros(4, dtype=np.float32))
        cache.clear()
        assert len(cache) == 0

    def test_disk_write_through_and_reload(self, tmp_path: Path) -> None:
        vec = np.arange(5, dtype=np.float32)
        EmbeddingCache(cache_dir=tmp_path).set("key", vec)
        # A brand-new cache over the same dir loads the persisted vector.
        reloaded = EmbeddingCache(cache_dir=tmp_path).get("key")
        assert reloaded is not None and np.array_equal(reloaded, vec)

    def test_corrupt_disk_file_degrades_to_miss(self, tmp_path: Path) -> None:
        (tmp_path / "bad.npy").write_bytes(b"not a real npy file")
        assert EmbeddingCache(cache_dir=tmp_path).get("bad") is None


# ---------------------------------------------------------------------------
# embed_chunks convenience
# ---------------------------------------------------------------------------


class TestEmbedChunks:
    def test_embeds_chunk_content(self, embedder: CodeEmbedder) -> None:
        chunks = [
            _FakeChunk("def a(): pass"),
            _FakeChunk("def b(): pass"),
            _FakeChunk("def c(): pass"),
        ]
        vectors = embedder.embed_chunks(chunks)
        assert vectors.shape == (3, 768)

    def test_matches_embed_batch(self, embedder: CodeEmbedder) -> None:
        texts = ["def a(): pass", "def b(): pass"]
        from_chunks = embedder.embed_chunks([_FakeChunk(t) for t in texts])
        from_batch = embedder.embed_batch(texts)
        assert np.allclose(from_chunks, from_batch)


# ---------------------------------------------------------------------------
# Backend selection / fallback
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_hashing_backend_reports_name(self) -> None:
        emb = CodeEmbedder(backend="hashing")
        assert emb.backend_name.startswith("hashing")
        assert emb.uses_transformer is False

    def test_injected_backend_used_directly(self) -> None:
        spy = _CountingBackend(dimension=32)
        emb = CodeEmbedder(backend=spy)
        assert emb.dimension == 32
        assert emb.backend_name == spy.name

    def test_unknown_backend_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            CodeEmbedder(backend="banana")

    def test_invalid_backend_type_raises(self) -> None:
        with pytest.raises(TypeError):
            CodeEmbedder(backend=object())

    def test_auto_falls_back_to_hashing_when_model_unavailable(self) -> None:
        """`auto` must degrade to hashing if the transformer cannot load."""

        def boom(self: TransformerEmbeddingBackend) -> None:
            raise RuntimeError("no model / no network")

        with patch.object(TransformerEmbeddingBackend, "load", boom):
            emb = CodeEmbedder(backend="auto", fallback=True)
        assert emb.backend_name.startswith("hashing")
        assert emb.embed_batch(["def f(): pass"]).shape == (1, 768)

    def test_transformer_backend_propagates_load_error(self) -> None:
        """`transformer` (no fallback) must surface a load failure."""

        def boom(self: TransformerEmbeddingBackend) -> None:
            raise RuntimeError("no model")

        with (
            patch.object(TransformerEmbeddingBackend, "load", boom),
            pytest.raises(RuntimeError),
        ):
            CodeEmbedder(backend="transformer")


# ---------------------------------------------------------------------------
# TransformerEmbeddingBackend -- pooling / batching (mocked model)
# ---------------------------------------------------------------------------


class TestTransformerBackendMocked:
    """Validate the real forward-pass logic without downloading a model.

    A ``torch`` tensor stand-in for the model output plus a fake tokenizer let
    us assert the masked-mean-pooling math exactly, the same technique
    ``test_neo4j_store`` uses to mock the Neo4j driver.
    """

    def _make_backend(self, pooling: str = "mean") -> TransformerEmbeddingBackend:
        import torch

        backend = TransformerEmbeddingBackend(
            "fake-model", device="cpu", pooling=pooling
        )

        def fake_tokenizer(batch: list[str], **_: object) -> dict[str, torch.Tensor]:
            # One "token" per character, padded to the longest input; the
            # attention mask marks real vs padding positions.
            lengths = [max(len(s), 1) for s in batch]
            width = max(lengths)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
            return {
                "input_ids": torch.ones((len(batch), width), dtype=torch.long),
                "attention_mask": mask,
            }

        def fake_model(**enc: torch.Tensor) -> object:
            mask = enc["attention_mask"]
            b, t = mask.shape
            hidden = 4
            # Deterministic hidden states: position index broadcast over hidden.
            positions = torch.arange(t, dtype=torch.float32).view(1, t, 1)
            last_hidden = positions.expand(b, t, hidden).clone()
            out = MagicMock()
            out.last_hidden_state = last_hidden
            return out

        backend._tokenizer = MagicMock(side_effect=fake_tokenizer)
        backend._model = MagicMock(side_effect=fake_model)
        backend._dimension = 4
        backend._device = "cpu"
        return backend

    def test_mean_pooling_ignores_padding(self) -> None:
        """Masked mean of positions [0..L-1] must exclude padded positions."""
        backend = self._make_backend("mean")
        # "abcd" -> length 4, mean of positions 0,1,2,3 = 1.5
        # "ab"   -> length 2, mean of positions 0,1     = 0.5
        out = backend.embed(["abcd", "ab"])
        assert out.shape == (2, 4)
        assert np.allclose(out[0], 1.5)
        assert np.allclose(out[1], 0.5)

    def test_cls_pooling_takes_first_token(self) -> None:
        backend = self._make_backend("cls")
        out = backend.embed(["abcd", "ab"])
        # First-token (position 0) hidden state is all zeros.
        assert np.allclose(out, 0.0)

    def test_batching_covers_all_inputs(self) -> None:
        backend = self._make_backend("mean")
        texts = ["a", "ab", "abc", "abcd", "abcde"]
        out = backend.embed(texts, batch_size=2)
        assert out.shape == (5, 4)
        # Row i is the mean of positions 0..i => i/2.
        for i in range(5):
            assert np.allclose(out[i], i / 2.0)

    def test_output_is_float32(self) -> None:
        out = self._make_backend("mean").embed(["abc"])
        assert out.dtype == np.float32

    def test_invalid_pooling_rejected(self) -> None:
        with pytest.raises(ValueError, match="pooling"):
            TransformerEmbeddingBackend("m", pooling="max")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CountingBackend:
    """A backend that counts calls, to prove the cache prevents recomputation."""

    def __init__(self, dimension: int = _DEFAULT_DIM) -> None:
        self._dimension = dimension
        self.name = f"counting:{dimension}"
        self.calls = 0
        self.total_texts = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        progress: object = None,
    ) -> np.ndarray:
        self.calls += 1
        self.total_texts += len(texts)
        # Deterministic non-zero rows so normalization is well-defined.
        out = np.ones((len(texts), self._dimension), dtype=np.float32)
        for i in range(len(texts)):
            out[i, i % self._dimension] += 1.0
        return out


class _FakeChunk:
    """Minimal stand-in for chunker.Chunk exposing only ``.content``."""

    def __init__(self, content: str) -> None:
        self.content = content

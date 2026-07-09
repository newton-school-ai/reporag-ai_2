"""Code embedding pipeline: CodeBERT / UniXcoder.

Embeds code chunks into dense vectors using a code-specific transformer
(``microsoft/unixcoder-base`` by default) and exposes them as L2-normalized
numpy arrays ready for cosine-similarity search.

Design
------
The public :class:`CodeEmbedder` is a thin orchestrator over a pluggable
*backend* plus a content-addressed *cache*.  Two backends implement the same
:class:`EmbeddingBackend` protocol:

* :class:`TransformerEmbeddingBackend` -- loads a Hugging Face encoder
  (CodeBERT / UniXcoder / any ``AutoModel``), runs batched inference on the
  best available device (CUDA -> MPS -> CPU), and mean-pools the last hidden
  state into one vector per input.  This is the production path.
* :class:`HashingEmbeddingBackend` -- a dependency-free, fully deterministic
  feature-hashing embedder.  It requires no model download, runs anywhere,
  and still gives *similar code high cosine similarity* (shared sub-tokens map
  to shared dimensions).  It is the offline / CI fallback and the default
  target for unit tests.

The ``"auto"`` backend eagerly loads the transformer and, on *any* failure
(missing ``torch``/``transformers``, no network, model not cached), logs a
warning and falls back to the hashing backend.  This mirrors the project's
established graceful-degradation pattern (``tiktoken`` -> regex heuristic,
Neo4j -> NetworkX).

Usage::

    from src.reporag.embedding.code_embedder import CodeEmbedder

    embedder = CodeEmbedder(model_name="microsoft/unixcoder-base")
    vectors = embedder.embed_batch(["def hello(): return 42", "class Foo: pass"])
    vectors.shape        # (2, 768)
    vectors[0] @ vectors[0]  # ~1.0  (unit vectors)

    # Cosine similarity of two snippets:
    embedder.similarity("def add(a, b): return a + b",
                        "def add(x, y): return x + y")  # -> high

    # Persist embeddings across runs so re-indexing is incremental:
    embedder = CodeEmbedder(cache_dir=".cache/embeddings")
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.reporag.config import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Embedding dimension of the default UniXcoder / CodeBERT encoders.
_DEFAULT_DIM = 768

#: Maximum number of tokens fed to the transformer per input (longer inputs
#: are truncated).  512 matches the position-embedding limit of RoBERTa-based
#: code encoders such as CodeBERT and UniXcoder.
_DEFAULT_MAX_LENGTH = 512

#: Default number of inputs processed per forward pass.
_DEFAULT_BATCH_SIZE = 32

#: Pooling strategies supported when reducing token vectors to one vector.
_VALID_POOLING = ("mean", "cls")


# ---------------------------------------------------------------------------
# Sub-tokenizer (shared by the hashing backend and useful on its own)
# ---------------------------------------------------------------------------

# Splits source into identifiers, numbers, and single punctuation/operator
# characters.  Whitespace is dropped.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

# Splits an identifier into its constituent words: snake_case parts and
# camelCase / PascalCase / ALLCAPS runs.
_IDENT_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def subtokenize(code: str) -> list[str]:
    """Split *code* into lower-cased sub-tokens for lexical embedding.

    - **Why it exists**: Identifier-aware tokenization is what lets
      ``getUserById`` and ``get_user_by_id`` share the sub-tokens
      ``get``, ``user``, ``by``, ``id`` -- the property that makes lexical
      similarity meaningful for code.  It backs the hashing embedder and is
      exported for reuse by the BM25 tokenizer (Issue 15).
    - **Algorithm**: Scan with :data:`_TOKEN_RE` to yield identifiers,
      integer literals, and standalone operator/punctuation characters.
      Each identifier is further split on underscores and camelCase
      boundaries via :data:`_IDENT_PART_RE`; every part is lower-cased.
      Operators and numbers pass through unchanged so ``a + b`` and
      ``a - b`` differ.
    - **Edge cases**: Empty or whitespace-only input returns ``[]``.
    """
    parts: list[str] = []
    for tok in _TOKEN_RE.findall(code):
        first = tok[0]
        if first.isalpha() or first == "_":
            for piece in _IDENT_PART_RE.findall(tok):
                parts.append(piece.lower())
        else:
            parts.append(tok)
    return parts


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(preferred: str | None = None) -> str:
    """Return the best available torch device string.

    Preference order is CUDA GPU -> Apple-Silicon MPS -> CPU.  An explicit
    *preferred* value is returned verbatim (trusting the caller).  Importing
    torch is deferred to call time so the hashing backend never pays for it.

    Args:
        preferred: An explicit device (``"cuda"``, ``"mps"``, ``"cpu"``) to
            use instead of auto-detection.

    Returns:
        A device string suitable for ``tensor.to(device)``.  Falls back to
        ``"cpu"`` if torch is not importable.
    """
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Structural interface every embedding backend must satisfy.

    A backend turns a list of strings into a ``(N, dimension)`` float32 array
    of *raw* (un-normalized) embeddings.  L2 normalization and caching are the
    orchestrator's responsibility, so backends stay minimal and easy to test.
    """

    name: str

    @property
    def dimension(self) -> int:
        """Length of the vectors this backend produces."""
        ...  # pragma: no cover

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Return raw embeddings of shape ``(len(texts), dimension)``."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Hashing backend (deterministic, dependency-free fallback)
# ---------------------------------------------------------------------------


class HashingEmbeddingBackend:
    """Deterministic feature-hashing embedder (the "hashing trick").

    - **Why it exists**: Guarantees the pipeline runs with zero model
      downloads -- in offline CI, air-gapped environments, or when
      ``torch``/``transformers`` are unavailable -- while still producing
      vectors where similar code has high cosine similarity.  Shared
      sub-tokens hash to shared dimensions, so lexically similar snippets
      overlap heavily.
    - **Algorithm**: For each input, :func:`subtokenize` yields sub-tokens.
      Each token is hashed with BLAKE2b (a *stable* cross-process hash --
      Python's built-in ``hash()`` is salted and must not be used here) to an
      index in ``[0, dimension)`` and a sign bit; the signed count is
      accumulated into that dimension.  The signed hashing keeps the
      expected dot-product an unbiased estimate of shared-token overlap.
    - **Edge cases**: An empty string yields the zero vector (normalized to
      zero downstream, never NaN).
    """

    def __init__(self, dimension: int = _DEFAULT_DIM) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.name = f"hashing:{dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Hash each input into a ``(N, dimension)`` float32 matrix."""
        total = len(texts)
        mat = np.zeros((total, self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in subtokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dimension
                sign = 1.0 if (digest[4] & 1) else -1.0
                mat[i, idx] += sign
            if progress is not None and (i + 1) % batch_size == 0:
                progress(i + 1, total)
        if progress is not None and total:
            progress(total, total)
        return mat

    def __repr__(self) -> str:
        return f"HashingEmbeddingBackend(dimension={self._dimension})"


# ---------------------------------------------------------------------------
# Transformer backend (production path)
# ---------------------------------------------------------------------------


class TransformerEmbeddingBackend:
    """Hugging Face encoder backend (CodeBERT / UniXcoder / any ``AutoModel``).

    - **Why it exists**: Code-specific encoders capture programming-language
      semantics (control flow, API usage, naming conventions) that a lexical
      hash cannot.  This is the production-quality embedding path.
    - **Algorithm**: Lazily loads the tokenizer and model on first use, moves
      the model to the best device, and runs batched inference under
      ``torch.no_grad()``.  Token vectors are reduced to a single vector per
      input by masked mean pooling (default) or the ``[CLS]`` token.
    - **Correctness choice**: Mean pooling with the attention mask ignores
      padding tokens, which gives markedly better retrieval than raw ``[CLS]``
      for encoders that were not fine-tuned for sentence embeddings -- the
      common case for CodeBERT/UniXcoder checkpoints.

    Args:
        model_name: Hugging Face model id or local path.
        device: Explicit torch device; ``None`` auto-selects (CUDA/MPS/CPU).
        max_length: Truncation length in tokens.
        pooling: ``"mean"`` (masked mean) or ``"cls"`` (first token).
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        pooling: str = "mean",
    ) -> None:
        if pooling not in _VALID_POOLING:
            raise ValueError(
                f"pooling must be one of {_VALID_POOLING}, got {pooling!r}"
            )
        self.model_name = model_name
        self.name = f"transformer:{model_name}:{pooling}"
        self._device = device
        self._max_length = max_length
        self._pooling = pooling
        self._tokenizer: Any = None
        self._model: Any = None
        self._dimension: int | None = None

    def load(self) -> None:
        """Load tokenizer and model, moving the model to the target device.

        Idempotent: a second call is a no-op.  Any import, download, or load
        failure propagates unchanged so callers (e.g. the ``"auto"`` factory)
        can decide whether to fall back.
        """
        if self._model is not None:
            return
        import torch  # noqa: F401  (validates torch availability early)
        from transformers import AutoModel, AutoTokenizer

        self._device = select_device(self._device)
        logger.info(
            "Loading transformer '%s' on device '%s'",
            self.model_name,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self._device)
        model.eval()
        self._model = model
        self._dimension = int(model.config.hidden_size)

    @property
    def dimension(self) -> int:
        """Hidden size of the loaded model (triggers a load if needed)."""
        if self._dimension is None:
            self.load()
        assert self._dimension is not None
        return self._dimension

    @property
    def device(self) -> str | None:
        """The resolved device, or ``None`` before the model is loaded."""
        return self._device

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Run batched inference and return raw pooled embeddings."""
        import torch

        self.load()
        total = len(texts)
        if total == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out_batches: list[np.ndarray] = []
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
            pooled = self._pool(output.last_hidden_state, encoded["attention_mask"])
            out_batches.append(pooled.detach().to("cpu", torch.float32).numpy())
            if progress is not None:
                progress(min(start + batch_size, total), total)

        return np.vstack(out_batches)

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        """Reduce ``(B, T, H)`` token vectors to ``(B, H)`` per the strategy.

        - **Algorithm**: For ``"cls"`` return the first token's vector.  For
          ``"mean"`` multiply by the attention mask, sum over the sequence
          axis, and divide by the number of real tokens (clamped away from
          zero) so padding never dilutes the mean.
        """
        if self._pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"TransformerEmbeddingBackend(model={self.model_name!r}, "
            f"pooling={self._pooling!r}, loaded={loaded})"
        )


# ---------------------------------------------------------------------------
# Embedding cache (content-addressed; in-memory with optional disk persistence)
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Content-addressed cache mapping a text key to its embedding vector.

    - **Why it exists**: Re-indexing a repository re-embeds thousands of
      chunks, most of which are unchanged.  Keying on content lets unchanged
      chunks hit the cache and skip the (expensive) model forward pass.
    - **Algorithm**: An in-memory ``dict`` provides O(1) hits within a run.
      When *cache_dir* is set the cache is write-through: each vector is also
      persisted as ``<key>.npy`` so hits survive across processes and runs.
    - **Correctness choice**: Disk reads/writes never raise into the caller;
      a corrupt or unreadable cache file degrades to a miss (recompute) rather
      than crashing the pipeline.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._mem: dict[str, np.ndarray] = {}
        self._dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached vector for *key*, or ``None`` on a miss."""
        cached = self._mem.get(key)
        if cached is not None:
            return cached
        if self._dir is not None:
            path = self._dir / f"{key}.npy"
            if path.exists():
                try:
                    arr = np.load(path)
                except (OSError, ValueError) as exc:
                    logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
                    return None
                self._mem[key] = arr
                return arr
        return None

    def set(self, key: str, vector: np.ndarray) -> None:
        """Store *vector* for *key* in memory and (if enabled) on disk."""
        self._mem[key] = vector
        if self._dir is not None:
            try:
                np.save(self._dir / f"{key}.npy", vector)
            except OSError as exc:
                logger.warning("Could not persist embedding to disk: %s", exc)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._mem)

    def clear(self) -> None:
        """Drop all in-memory entries (on-disk files are left untouched)."""
        self._mem.clear()

    def __repr__(self) -> str:
        loc = str(self._dir) if self._dir is not None else "memory"
        return f"EmbeddingCache(entries={len(self._mem)}, store={loc})"


# ---------------------------------------------------------------------------
# CodeEmbedder (public orchestrator)
# ---------------------------------------------------------------------------


class CodeEmbedder:
    """Embed code strings into L2-normalized vectors for cosine search.

    Combines a pluggable :class:`EmbeddingBackend` with an
    :class:`EmbeddingCache` and centralizes L2 normalization.  A single
    instance is meant to be built once per pipeline run and reused.

    Args:
        model_name: Hugging Face model id for the transformer backend.
            Defaults to ``settings.code_embedding_model``.
        backend: ``"auto"`` (transformer, hashing fallback), ``"transformer"``
            (require the model, raise on failure), ``"hashing"`` (always the
            deterministic fallback), or a ready-made :class:`EmbeddingBackend`
            instance (useful for tests).
        device: Explicit torch device; ``None`` auto-selects.
        batch_size: Inputs per forward pass.
        max_length: Transformer truncation length in tokens.
        pooling: ``"mean"`` or ``"cls"`` pooling for the transformer backend.
        dimension: Fallback / hashing embedding size (default 768, matching
            UniXcoder).  The real dimension of a loaded transformer overrides
            this.
        normalize: When ``True`` (default) every returned vector is L2-scaled
            to unit length so a dot product equals cosine similarity.
        cache_dir: Directory for a persistent write-through cache; ``None``
            keeps the cache in memory only.
        cache: Inject a pre-built :class:`EmbeddingCache` (overrides
            *cache_dir*).
        fallback: When ``True`` (default) ``backend="auto"`` degrades to the
            hashing backend if the transformer cannot be loaded.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str | EmbeddingBackend = "auto",
        device: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_length: int = _DEFAULT_MAX_LENGTH,
        pooling: str = "mean",
        dimension: int = _DEFAULT_DIM,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
        cache: EmbeddingCache | None = None,
        fallback: bool = True,
    ) -> None:
        self.model_name = model_name or settings.code_embedding_model
        self.batch_size = batch_size
        self.normalize = normalize
        self._pooling = pooling
        self._dim = dimension
        self._cache = cache if cache is not None else EmbeddingCache(cache_dir)
        self._hits = 0
        self._misses = 0
        self._backend = self._build_backend(
            backend,
            device=device,
            max_length=max_length,
            pooling=pooling,
            fallback=fallback,
        )
        # Adopt the backend's true dimension when it is known up front
        # (hashing always, transformer only once loaded).  A lazy injected
        # backend may defer until its first embed call.
        with contextlib.suppress(Exception):
            self._dim = self._backend.dimension

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _build_backend(
        self,
        backend: str | EmbeddingBackend,
        *,
        device: str | None,
        max_length: int,
        pooling: str,
        fallback: bool,
    ) -> EmbeddingBackend:
        """Resolve the *backend* argument into a concrete backend instance.

        - **Why it exists**: Encapsulates the ``auto`` graceful-degradation
          policy so the constructor stays declarative.
        - **Algorithm**: A ready-made backend instance is used as-is.  For
          ``"hashing"`` build the deterministic backend.  For
          ``"transformer"``/``"auto"`` construct and *eagerly load* the
          transformer so failures surface immediately; under ``"auto"`` any
          failure is caught and swapped for the hashing backend.
        - **Correctness choice**: Eager load (like ``Neo4jGraphStore``
          verifying connectivity in its constructor) means an ``"auto"``
          embedder is fully usable and its ``backend_name`` truthfully
          reflects the path actually chosen.
        """
        if not isinstance(backend, str):
            if not isinstance(backend, EmbeddingBackend):
                raise TypeError(
                    "backend must be a string or an EmbeddingBackend, "
                    f"got {type(backend).__name__}"
                )
            return backend

        choice = backend.lower()
        if choice == "hashing":
            return HashingEmbeddingBackend(dimension=self._dim)

        if choice in ("transformer", "auto"):
            try:
                impl = TransformerEmbeddingBackend(
                    self.model_name,
                    device=device,
                    max_length=max_length,
                    pooling=pooling,
                )
                impl.load()
                return impl
            except Exception as exc:  # noqa: BLE001 - many failure modes
                if choice == "auto" and fallback:
                    logger.warning(
                        "CodeEmbedder: transformer backend unavailable (%s); "
                        "falling back to deterministic hashing embedder",
                        exc,
                    )
                    return HashingEmbeddingBackend(dimension=self._dim)
                raise

        raise ValueError(
            f"Unknown backend {backend!r}. Use 'auto', 'transformer', "
            "'hashing', or an EmbeddingBackend instance."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string, returning a ``(dimension,)`` vector."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed many strings into a ``(len(texts), dimension)`` array.

        - **Why it exists**: The batch entry point that every caller uses.
          It layers caching over the backend so unchanged inputs never hit the
          model twice, within or across runs.
        - **Algorithm**: Look up each input in the cache.  Compute embeddings
          only for the *unique* misses in one backend call, L2-normalize them,
          store them, then reassemble the output in the original order
          (duplicates share a single computed vector).
        - **Edge cases**: Empty input returns an empty ``(0, dimension)``
          array without touching the backend.
        - **Correctness choice**: De-duplicating misses means embedding N
          identical strings costs exactly one forward pass.

        Args:
            texts: The strings to embed.
            progress: Optional ``progress(done, total)`` callback reporting
                over the number of *uncached* inputs.

        Returns:
            A float32 array; each row is the (optionally L2-normalized)
            embedding of the corresponding input.
        """
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        keys = [self._content_key(t) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)

        # Partition into hits and unique misses (preserving first-seen order).
        unique_miss_texts: list[str] = []
        key_to_miss_pos: dict[str, int] = {}
        for i, (text, key) in enumerate(zip(texts, keys, strict=True)):
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                self._hits += 1
                continue
            self._misses += 1
            if key not in key_to_miss_pos:
                key_to_miss_pos[key] = len(unique_miss_texts)
                unique_miss_texts.append(text)

        # Compute and cache the unique misses in a single backend call.
        if unique_miss_texts:
            raw = self._backend.embed(
                unique_miss_texts, batch_size=self.batch_size, progress=progress
            )
            vectors = (
                self._l2_normalize(raw)
                if self.normalize
                else raw.astype(np.float32, copy=False)
            )
            self._dim = vectors.shape[1]
            for key, pos in key_to_miss_pos.items():
                self._cache.set(key, vectors[pos])

        # Fill in the miss slots from the now-populated cache.
        for i, key in enumerate(keys):
            if results[i] is None:
                results[i] = self._cache.get(key)

        return np.vstack(results).astype(np.float32, copy=False)

    def embed_chunks(
        self,
        chunks: Iterable[Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed the ``.content`` of each chunk, aligned to input order.

        Accepts any object exposing a ``content`` attribute -- notably
        :class:`~src.reporag.ingestion.chunker.Chunk` -- so the embedder stays
        decoupled from the chunker's concrete type.
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed_batch(texts, progress=progress)

    def similarity(self, a: str, b: str) -> float:
        """Return the cosine similarity of two code strings in ``[-1, 1]``."""
        vectors = self.embed_batch([a, b])
        va, vb = vectors[0], vectors[1]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """The embedding dimension produced by this embedder."""
        return self._dim

    @property
    def backend_name(self) -> str:
        """Identifier of the backend actually in use (e.g. after fallback)."""
        return self._backend.name

    @property
    def uses_transformer(self) -> bool:
        """``True`` when a transformer model backs this embedder."""
        return isinstance(self._backend, TransformerEmbeddingBackend)

    def cache_stats(self) -> dict[str, int]:
        """Return cache ``{"hits", "misses", "size"}`` counters."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

    def clear_cache(self) -> None:
        """Reset the in-memory cache and its hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _content_key(self, text: str) -> str:
        """Derive a stable cache key from the model config and *text*.

        The key folds in ``model_name``, pooling, and the backend name so a
        vector produced by one configuration is never served to another (e.g.
        a hashing vector must not satisfy a transformer request).  The
        mutable, discovered ``self._dim`` is intentionally excluded so keys
        stay stable across the first embed call.
        """
        hasher = hashlib.sha256()
        for part in (self.model_name, self._pooling, self._backend.name, text):
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row; zero rows stay zero (no divide-by-zero)."""
        matrix = matrix.astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"CodeEmbedder(model={self.model_name!r}, backend={self.backend_name!r}, "
            f"dimension={self._dim}, normalize={self.normalize})"
        )

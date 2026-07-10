"""Code embedding pipeline.

Embeds code chunks using CodeBERT or UniXcoder. Produces 768-dim
L2-normalized vectors. Supports batch embedding with GPU acceleration and
automatic CPU fallback, and caches embeddings so unchanged code is never
re-embedded.

Dependency on Issue 8
----------------------
:meth:`CodeEmbedder.embed` / :meth:`CodeEmbedder.embed_batch` accept either
raw code strings or Issue 8's
:class:`~src.reporag.ingestion.chunker.Chunk` objects directly (duck-typed
on ``chunk.content``), so this sits immediately downstream of
:class:`~src.reporag.ingestion.chunker.SemanticChunker` with no glue code:

    chunks = SemanticChunker().chunk_source(source, file_path="a.py")
    vectors = CodeEmbedder().embed_batch(chunks)  # (len(chunks), 768)

Model loading is lazy: constructing a :class:`CodeEmbedder` does no network
I/O or GPU work. The model/tokenizer are only pulled from Hugging Face (or
moved onto the resolved device) the first time an embedding is actually
computed -- so building an embedder is cheap even if it's never used, and a
model/tokenizer can be injected directly for testing without touching the
network at all.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from src.reporag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
"""Output vector dimensionality (CodeBERT / UniXcoder base hidden size)."""

Device = Literal["auto", "cuda", "mps", "cpu"]
CodeInput = Union[str, "Chunk"]


def _extract_text(item: CodeInput) -> str:
    """Return the code text for *item*, whether it's a raw string or a Chunk.

    Duck-typed on a ``content`` attribute rather than importing
    :class:`~src.reporag.ingestion.chunker.Chunk` at runtime, so this module
    doesn't need Issue 8 installed just to embed plain strings.
    """
    content = getattr(item, "content", None)
    return content if content is not None else str(item)


class CodeEmbedder:
    """Embeds code strings/chunks into L2-normalized 768-dim vectors.

    Args:
        model_name: Hugging Face model id (defaults to
            ``settings.code_embedding_model``, i.e. ``microsoft/unixcoder-base``).
        device: ``"auto"`` (default) picks CUDA if available, else CPU;
            ``"cuda"`` requests GPU explicitly but still falls back to CPU
            with a warning if none is available; ``"cpu"`` forces CPU.
        batch_size: Default batch size for :meth:`embed_batch` (override
            per-call).
        max_length: Max token length passed to the tokenizer; longer inputs
            are truncated.
        model: Pre-built ``transformers`` model (inject for tests, or to
            reuse a model already loaded elsewhere). Skips
            ``from_pretrained`` entirely when combined with *tokenizer*.
        tokenizer: Pre-built ``transformers`` tokenizer, paired with *model*.
        cache: A mutable mapping used as the embedding cache (defaults to a
            plain in-memory ``dict``). Inject a disk- or Redis-backed
            mapping to persist the cache across runs.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: Device = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        model: Any | None = None,
        tokenizer: Any | None = None,
        cache: MutableMapping[str, np.ndarray] | None = None,
    ) -> None:
        """Configure the embedder; no model loading or GPU work happens here."""
        if model_name is None:
            from src.reporag.config import settings

            model_name = settings.code_embedding_model
        self._model_name = model_name
        self._requested_device = device
        self._device = self._resolve_device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = model
        self._tokenizer = tokenizer
        self._model_ready = False
        self._cache: MutableMapping[str, np.ndarray] = (
            cache if cache is not None else {}
        )
        self._hits = 0
        self._misses = 0

    def _cache_key(self, text: str) -> str:
        """Deterministic cache key incorporating the model name and text."""
        return hashlib.sha256(f"{self._model_name}|{text}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Device resolution (GPU support with CPU fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Device) -> str:
        """Resolve ``"auto"``/``"cuda"``/``"cpu"`` to a concrete torch device string.

        Preference order for ``"auto"``: CUDA -> MPS -> CPU.
        ``"cuda"`` requests that fail fall back to CPU (with a warning).
        """
        import torch

        if device == "cpu":
            return "cpu"
        if device in ("cuda", "auto"):
            if torch.cuda.is_available():
                return "cuda"
            if device == "cuda":
                logger.warning("CUDA requested but not available; falling back to CPU")
            # If auto, check MPS before falling to CPU
            if (
                device == "auto"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
            return "cpu"
        if device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available; falling back to CPU")
            return "cpu"

        raise ValueError(
            f"Unknown device {device!r}; expected 'auto', 'cuda', 'mps', or 'cpu'"
        )

    @property
    def device(self) -> str:
        """The resolved device embeddings are computed on (``'cuda'`` or ``'cpu'``)."""
        return self._device

    @property
    def embedding_dim(self) -> int:
        """Output vector dimensionality (always :data:`EMBEDDING_DIM`)."""
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model/tokenizer from Hugging Face if not injected, then place on device.

        Runs at most once: subsequent calls are a no-op check of
        ``_model_ready``.
        """
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading code embedding model '%s'", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)

        if not self._model_ready:
            self._model.to(self._device)
            self._model.eval()
            self._model_ready = True

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed(self, code: CodeInput, *, use_cache: bool = True) -> np.ndarray:
        """Embed a single code string or Chunk, returning a ``(768,)`` L2-normalized vector."""
        return self.embed_batch([code], use_cache=use_cache)[0]

    def similarity(self, a: CodeInput, b: CodeInput) -> float:
        """Calculate cosine similarity between two code inputs.

        Since vectors are L2 normalized, this is a dot product. Returns 0.0
        if either vector is fully zeroed out.
        """
        vecs = self.embed_batch([a, b])
        norm_a = np.linalg.norm(vecs[0])
        norm_b = np.linalg.norm(vecs[1])
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(vecs[0], vecs[1]))

    def embed_batch(
        self,
        items: Sequence[CodeInput],
        *,
        batch_size: int | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Embed a batch of code strings/Chunks.

        - **Why it exists**: The primary entry point -- accepts a mix of raw
          strings and Issue 8 ``Chunk`` objects, batches whatever isn't
          already cached, and returns embeddings in the same order as
          *items* regardless of cache hits/misses or batch boundaries.
        - **Algorithm**: Extracts text for every item, splits into
          cache-hit / cache-miss groups, runs the model only over the
          misses in chunks of *batch_size*, and stitches the results back
          into their original positions.
        - **Edge cases**: An empty *items* returns an empty
          ``(0, 768)`` array without touching the model (so building an
          embedder and calling it with nothing never triggers a model
          load).

        Args:
            items: Code strings and/or ``Chunk`` objects, in any mix.
            batch_size: Overrides the constructor's default batch size for
                this call.
            use_cache: When ``False``, bypasses and does not populate the
                cache for this call.

        Returns:
            A ``(len(items), 768)`` ``float32`` array, L2-normalized
            row-wise, in the same order as *items*.
        """
        texts = [_extract_text(item) for item in items]
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache.get(key) if use_cache else None
            if cached is not None:
                results[i] = cached
                if use_cache:
                    self._hits += 1
            else:
                to_compute.append((i, text))
                if use_cache:
                    self._misses += 1

        if to_compute:
            # Deduplicate items to avoid redundant GPU computation within the same call.
            unique_texts: list[str] = []
            text_to_idx: dict[str, list[int]] = {}
            for original_idx, text in to_compute:
                if text not in text_to_idx:
                    text_to_idx[text] = []
                    unique_texts.append(text)
                text_to_idx[text].append(original_idx)

            self._ensure_loaded()
            effective_batch_size = batch_size or self._batch_size
            for start in range(0, len(unique_texts), effective_batch_size):
                batch_texts = unique_texts[start : start + effective_batch_size]
                vectors = self._compute_batch(batch_texts)
                for text, vector in zip(batch_texts, vectors, strict=True):
                    # Map the single computed vector back to all positions where this text appeared
                    for idx in text_to_idx[text]:
                        results[idx] = vector
                    if use_cache:
                        self._cache[self._cache_key(text)] = vector

        return np.stack(results).astype(np.float32)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Model forward pass + pooling + normalization
    # ------------------------------------------------------------------

    def _compute_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Run one forward pass over *texts*, returning L2-normalized (768,) vectors."""
        import torch

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._model(**encoded)

        pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        normalized = self._l2_normalize(pooled)
        return list(normalized.cpu().numpy().astype(np.float32))

    @staticmethod
    def _mean_pool(token_embeddings: Any, attention_mask: Any) -> Any:
        """Attention-mask-weighted mean pool over the token dimension.

        Padding tokens are excluded from the average (rather than diluting
        it with zero vectors), which is what makes this safe to use on
        batches of mixed-length inputs.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _l2_normalize(embeddings: Any) -> Any:
        """L2-normalize each row to unit length (guards against a zero-norm divide)."""
        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return embeddings / norms

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Remove every cached embedding and reset tracking statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_size(self) -> int:
        """Return the number of embeddings currently cached."""
        return len(self._cache)

    def cache_stats(self) -> dict[str, int]:
        """Return dictionary containing hits, misses, and cache size."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.cache_size(),
        }

    def __contains__(self, code: CodeInput) -> bool:
        """Return ``True`` if *code*'s embedding is already cached for the current model."""
        return self._cache_key(_extract_text(code)) in self._cache

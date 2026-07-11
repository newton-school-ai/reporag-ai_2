"""Code embedding pipeline.

Embeds code chunks using CodeBERT or UniXcoder. Produces 768-dim L2-normalized
vectors. Supports batch embedding with GPU acceleration and CPU fallback.

Model loading is deferred until the first embedding call, keeping construction
cheap and test-friendly.  A model and tokenizer can also be injected directly
for testing without network access.

Downstream integration
----------------------
:meth:`CodeEmbedder.embed_batch` accepts plain strings or
:class:`~src.reporag.ingestion.chunker.Chunk` objects (duck-typed on
``.content``), so it plugs directly into the chunking pipeline::

    chunks = SemanticChunker().chunk_file("src/foo.py")
    vectors = CodeEmbedder().embed_batch(chunks)   # (len(chunks), 768)
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from reporag.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
"""Hidden-size for CodeBERT / UniXcoder base models."""


def _resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available accelerator.

    Resolution order for ``"auto"``: CUDA -> MPS (Apple Silicon) -> CPU.

    An explicit ``"cuda"`` or ``"mps"`` request is honoured only when that
    backend is actually available; otherwise it falls back to CPU so the
    embedder never crashes on a machine without a GPU.
    """
    if preference in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_text(item: Any) -> str:
    """Return the code text for *item*, whether raw string or Chunk.

    Duck-typed on a ``content`` attribute so this module never needs to
    import the Chunk dataclass at runtime.
    """
    content = getattr(item, "content", None)
    return content if isinstance(content, str) else str(item)


class CodeEmbedder:
    """Embeds code strings/chunks into L2-normalised 768-dim vectors.

    Features
    --------
    * **Lazy loading** -- the model is downloaded/moved to device on the first
      ``embed_batch`` call, not at construction time.
    * **Attention-mask-weighted mean pooling** -- uses all token representations
      (not just ``[CLS]``) for higher-quality embeddings.
    * **GPU acceleration** -- auto-selects CUDA / MPS / CPU.
    * **LRU cache** -- content-addressed, bounded, avoids re-computation.
    * **Batch deduplication** -- duplicate inputs within a single batch are
      computed only once.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``settings.code_embedding_model``.
        device: ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        batch_size: Default mini-batch size for GPU inference.
        max_length: Maximum token length for the tokeniser.
        cache_maxsize: Upper bound on cached embeddings (0 disables cache).
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        cache_maxsize: int = 10_000,
    ) -> None:
        self.model_name: str = model_name or settings.code_embedding_model
        self.batch_size = batch_size
        self.max_length = max_length

        self._device = _resolve_device(device)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._loaded = False

        # Content-addressed LRU cache
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_maxsize = cache_maxsize
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The resolved compute device."""
        return self._device

    @property
    def embedding_dim(self) -> int:
        """Output vector dimensionality."""
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download (if needed) and move the model to the target device.

        Called automatically before the first forward pass.  Subsequent
        calls are a fast no-op.  A pre-injected model/tokenizer (via
        ``_tokenizer`` / ``_model``) is respected, making tests network-free.
        """
        if self._loaded:
            return

        if self._tokenizer is None or self._model is None:
            from transformers import AutoModel, AutoTokenizer

            logger.info(
                "Loading code embedding model '%s' on %s",
                self.model_name,
                self._device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed(self, code: Any) -> np.ndarray:
        """Embed a single code string or Chunk, returning a ``(768,)`` vector."""
        return self.embed_batch([code])[0]

    def embed_batch(
        self,
        items: Sequence[Any],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed a batch of code strings or Chunk objects.

        Accepts any mix of raw strings and objects with a ``.content``
        attribute (e.g. :class:`~src.reporag.ingestion.chunker.Chunk`).

        Within a single call, duplicate texts are computed only once and
        the result is shared across all positions.

        Args:
            items: Code strings and/or Chunk objects.
            batch_size: Override the instance default for this call.

        Returns:
            A ``(len(items), 768)`` float32 array, L2-normalised row-wise.
        """
        texts = [_extract_text(item) for item in items]
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        effective_bs = batch_size or self.batch_size
        results: list[np.ndarray | None] = [None] * len(texts)

        # --- Phase 1: resolve cache hits & collect unique misses ----------
        unique_miss_texts: list[str] = []
        miss_key_to_positions: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                self._cache.move_to_end(key)  # refresh LRU position
                self._hits += 1
            else:
                self._misses += 1
                if key not in miss_key_to_positions:
                    miss_key_to_positions[key] = []
                    unique_miss_texts.append(text)
                miss_key_to_positions[key].append(i)

        # --- Phase 2: batch-compute unique misses -------------------------
        if unique_miss_texts:
            self._ensure_loaded()
            for start in range(0, len(unique_miss_texts), effective_bs):
                batch_texts = unique_miss_texts[start : start + effective_bs]
                vectors = self._forward(batch_texts)

                for text, vec in zip(batch_texts, vectors, strict=True):
                    key = self._cache_key(text)
                    # Place into every position that needs this text
                    for pos in miss_key_to_positions[key]:
                        results[pos] = vec
                    # Update cache (with LRU eviction)
                    self._cache[key] = vec
                    if self._cache_maxsize and len(self._cache) > self._cache_maxsize:
                        self._cache.popitem(last=False)

        return np.stack(results, dtype=np.float32)  # type: ignore[arg-type]

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between two code inputs.

        Since embeddings are L2-normalised, this is equivalent to a dot
        product.  Returns 0.0 if either vector is zero.
        """
        vecs = self.embed_batch([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        return dot

    # ------------------------------------------------------------------
    # Model forward pass
    # ------------------------------------------------------------------

    def _forward(self, texts: list[str]) -> list[np.ndarray]:
        """Run one forward pass and return L2-normalised (768,) vectors."""
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._model(**encoded)

        pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        normalised = F.normalize(pooled, p=2, dim=1)
        return list(normalised.cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over the token dimension.

        Padding tokens are excluded from the average so variable-length
        inputs produce faithful representations.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """Content-addressed key incorporating model name for safety."""
        raw = f"{self.model_name}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_stats(self) -> dict[str, int]:
        """Return ``{"hits": ..., "misses": ..., "size": ...}`` counters."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Drop all cached embeddings and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __repr__(self) -> str:
        return (
            f"CodeEmbedder(model={self.model_name!r}, "
            f"device={self._device}, loaded={self._loaded})"
        )

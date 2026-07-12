"""Docstring and comment embedding pipeline.

Embeds docstrings, comments, and README sections using sentence-transformers.
Each embedding links back to its parent code symbol for cross-reference.

Model loading is deferred until the first embedding call, keeping construction
cheap and test-friendly.  A model and tokenizer can also be injected directly
for testing without network access.

Downstream integration
----------------------
:meth:`DocEmbedder.embed_batch` embeds raw natural-language strings::

    vectors = DocEmbedder().embed_batch(["Authenticate user", "Parse body"])
    # vectors.shape == (2, 384)

For symbol-linked embeddings, use :meth:`DocEmbedder.embed_symbols`::

    symbols = [DocSymbol("func_123", "Authenticate user with JWT token")]
    results = DocEmbedder().embed_symbols(symbols)
    # results[0].symbol_id == "func_123"
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from reporag.config import settings

logger = logging.getLogger(__name__)

DOC_EMBEDDING_DIM = 384
"""Hidden-size for sentence-transformers/all-MiniLM-L6-v2."""


@dataclass
class DocSymbol:
    """A documentation snippet linked to its parent code symbol.

    Attributes:
        symbol_id: Unique identifier of the parent code symbol.
        text: Natural-language text (docstring, comment, README section).
    """

    symbol_id: str
    text: str


@dataclass
class DocEmbeddingResult:
    """Result of embedding a :class:`DocSymbol`.

    Attributes:
        symbol_id: Unique identifier of the parent code symbol.
        text: The original text that was embedded.
        vector: The L2-normalised embedding vector, or a zero vector if
            *text* was empty/whitespace-only.
    """

    symbol_id: str
    text: str
    vector: np.ndarray


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


class DocEmbedder:
    """Embeds natural-language text into L2-normalised 384-dim vectors.

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
    * **Empty-string handling** -- empty/whitespace-only inputs produce a zero
      vector and are never sent through the model.
    * **Symbol linking** -- :meth:`embed_symbols` ties each embedding to a
      parent symbol ID.
    * **Progress callback** -- ``on_progress(completed, total)`` for long
      batches.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``settings.doc_embedding_model``.
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
        batch_size: int = 32,
        max_length: int = 256,
        cache_maxsize: int = 10_000,
    ) -> None:
        self.model_name: str = model_name or settings.doc_embedding_model
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
        return DOC_EMBEDDING_DIM

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
                "Loading doc embedding model '%s' on %s",
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

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string, returning a ``(384,)`` vector.

        Empty or whitespace-only text returns a zero vector.
        """
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed a batch of natural-language strings.

        Empty or whitespace-only strings produce zero vectors and are **not**
        sent through the model.

        Within a single call, duplicate texts are computed only once and
        the result is shared across all positions.

        Args:
            texts: Natural-language strings to embed.
            batch_size: Override the instance default for this call.
            on_progress: Optional callback ``(completed, total)`` invoked
                after each mini-batch finishes.

        Returns:
            A ``(len(texts), 384)`` float32 array, L2-normalised row-wise.
            Rows for empty inputs are zero vectors.
        """
        if not texts:
            return np.empty((0, DOC_EMBEDDING_DIM), dtype=np.float32)

        effective_bs = batch_size or self.batch_size
        total = len(texts)
        results: list[np.ndarray | None] = [None] * total

        # --- Phase 0: handle empty/whitespace-only texts ------------------
        zero_vec = np.zeros(DOC_EMBEDDING_DIM, dtype=np.float32)

        # --- Phase 1: resolve cache hits & collect unique misses ----------
        unique_miss_texts: list[str] = []
        miss_key_to_positions: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = zero_vec
                continue

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
        completed = total - len(
            [pos for positions in miss_key_to_positions.values() for pos in positions]
        )

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
                        completed += 1
                    # Update cache (with LRU eviction)
                    self._cache[key] = vec
                    if self._cache_maxsize and len(self._cache) > self._cache_maxsize:
                        self._cache.popitem(last=False)

                if on_progress is not None:
                    on_progress(completed, total)

        # Fire final progress if there were no misses but callback was given
        if on_progress is not None and not unique_miss_texts:
            on_progress(total, total)

        return np.stack(results, dtype=np.float32)  # type: ignore[arg-type]

    def embed_symbols(
        self,
        symbols: Sequence[DocSymbol],
        *,
        batch_size: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[DocEmbeddingResult]:
        """Embed documentation symbols and link each vector to its parent.

        Empty/whitespace-only ``text`` fields produce a zero vector (the
        symbol is still included in the output but is not sent through
        the model).

        Args:
            symbols: Documentation snippets with their parent symbol IDs.
            batch_size: Override the instance default for this call.
            on_progress: Optional callback ``(completed, total)`` invoked
                after each mini-batch finishes.

        Returns:
            A list of :class:`DocEmbeddingResult`, one per input symbol,
            preserving the input order.
        """
        texts = [s.text for s in symbols]
        vectors = self.embed_batch(
            texts, batch_size=batch_size, on_progress=on_progress
        )
        return [
            DocEmbeddingResult(
                symbol_id=sym.symbol_id,
                text=sym.text,
                vector=vec,
            )
            for sym, vec in zip(symbols, vectors, strict=True)
        ]

    def embed_records(
        self,
        records: Sequence[Any],
        *,
        batch_size: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        include_signature: bool = False,
    ) -> list[DocEmbeddingResult]:
        """Embed docstrings directly from symbol records, skipping empty ones.

        Duck-types the input items: looks for ``docstring``, ``signature``,
        and ``symbol_id`` attributes (so it accepts
        :class:`~src.reporag.graph.symbol_table.SymbolRecord`).

        Records missing a docstring, or with an empty/whitespace-only docstring,
        are safely filtered out and do not appear in the results.

        Args:
            records: A sequence of objects (like ``SymbolRecord``) with a
                ``docstring`` and ``symbol_id``.
            batch_size: Override the instance default for this call.
            on_progress: Optional callback ``(completed, total)`` invoked
                after each mini-batch finishes.
            include_signature: If True, the ``signature`` (if present) is
                prepended to the docstring text before embedding. This gives
                the model crucial semantic context.

        Returns:
            A list of :class:`DocEmbeddingResult`, one per valid record.
        """
        valid_records = []
        valid_texts = []
        for r in records:
            docstring = getattr(r, "docstring", None)
            if docstring and isinstance(docstring, str) and docstring.strip():
                text = docstring.strip()
                if include_signature:
                    sig = getattr(r, "signature", None)
                    if sig and isinstance(sig, str) and sig.strip():
                        text = f"{sig.strip()}\n{text}"
                valid_records.append(r)
                valid_texts.append(text)

        if not valid_records:
            return []

        vectors = self.embed_batch(
            valid_texts,
            batch_size=batch_size,
            on_progress=on_progress,
        )

        embeddings = []
        for record, text, vec in zip(valid_records, valid_texts, vectors, strict=True):
            embeddings.append(
                DocEmbeddingResult(
                    symbol_id=getattr(record, "symbol_id", ""),
                    text=text,
                    vector=vec,
                )
            )
        return embeddings

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two text inputs.

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
        """Run one forward pass and return L2-normalised (384,) vectors."""
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
            f"DocEmbedder(model={self.model_name!r}, "
            f"device={self._device}, loaded={self._loaded})"
        )

"""Docstring and comment embedding pipeline.

Embeds docstrings, comments, and README sections using
``sentence-transformers``.  Produces L2-normalised 384-dim vectors from
natural language text (``all-MiniLM-L6-v2`` default) and links every embedding
back to the ``symbol_id`` of its parent code symbol, making the vectors
queryable by the retrieval layer (Issue 16).

Key design points
-----------------
* **Lazy model loading** -- the SentenceTransformer is not imported or
  downloaded until the first embedding call, keeping construction cheap and
  test-friendly.  A pre-built model can be injected via ``_model`` for tests
  that must run offline.
* **Empty docstring handling** -- records with ``None`` or whitespace-only
  docstrings are skipped gracefully; we do not embed empty strings.
* **Progress callback** -- :meth:`embed_batch` accepts an optional
  ``progress`` callable (``(completed: int, total: int) -> None``) so callers
  can drive progress-bars.
* **Batch deduplication** -- identical texts within a single batch are
  computed once and fanned out.
* **LRU cache** -- content-addressed, bounded, avoids repeated inference for
  the same docstring.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reporag.config import settings

logger = logging.getLogger(__name__)

DOC_EMBEDDING_DIM = 384
"""Output vector dimensionality for the default sentence-transformer model."""


@dataclass
class DocEmbedding:
    """A single embedded natural-language fragment linked to a code symbol.

    Attributes:
        symbol_id: The stable registry key of the parent SymbolRecord.
        text: The raw natural-language text that was embedded.
        embedding: L2-normalised float32 vector of shape ``(384,)``.
        source: Human-readable label for what kind of text produced this.
    """

    symbol_id: str
    text: str
    embedding: np.ndarray
    source: str = "docstring"

    def __repr__(self) -> str:
        dim = self.embedding.shape[0] if self.embedding.ndim else "?"
        return (
            f"DocEmbedding(symbol_id={self.symbol_id!r}, "
            f"source={self.source!r}, dim={dim})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "symbol_id": self.symbol_id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocEmbedding:
        """Reconstruct a DocEmbedding from to_dict output."""
        return cls(
            symbol_id=str(data["symbol_id"]),
            text=str(data["text"]),
            embedding=np.array(data["embedding"], dtype=np.float32),
            source=str(data.get("source", "docstring")),
        )


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class DocEmbedder:
    """Embeds docstrings, comments, and README sections into dense vectors.

    Uses ``sentence-transformers`` (default: ``all-MiniLM-L6-v2``) which is
    optimised for semantic similarity of natural language.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 32,
        cache_maxsize: int = 10_000,
    ) -> None:
        self.model_name: str = model_name or settings.doc_embedding_model
        self.device = device
        self.batch_size = batch_size

        self._model: Any | None = None
        self._loaded = False

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_maxsize = cache_maxsize
        self._hits = 0
        self._misses = 0

    @property
    def embedding_dim(self) -> int:
        """Output vector dimensionality (384)."""
        return DOC_EMBEDDING_DIM

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer on first use."""
        if self._loaded:
            return

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            effective_device = self.device
            if effective_device == "auto":
                effective_device = _pick_device()

            logger.info(
                "Loading doc embedding model '%s' on %s",
                self.model_name,
                effective_device,
            )
            self._model = SentenceTransformer(self.model_name, device=effective_device)

        self._loaded = True

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed a batch of text strings, returning a 2D numpy array.

        Empty or whitespace-only inputs are handled gracefully by returning
        zero vectors without calling the model.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        effective_bs = batch_size or self.batch_size
        total = len(texts)

        results = np.zeros((total, self.embedding_dim), dtype=np.float32)

        non_empty_indices = []
        non_empty_texts = []
        for idx, text in enumerate(texts):
            if text is not None and isinstance(text, str) and text.strip():
                non_empty_indices.append(idx)
                non_empty_texts.append(text)

        if not non_empty_texts:
            return results

        embedding_map = {}
        unique_miss_texts = []
        seen_keys = {}

        for text in non_empty_texts:
            key = self._cache_key(text)
            if key in self._cache:
                embedding_map[text] = self._cache[key]
                self._cache.move_to_end(key)
                self._hits += 1
            elif key not in seen_keys:
                seen_keys[key] = text
                unique_miss_texts.append(text)
                self._misses += 1
            else:
                self._misses += 1

        if unique_miss_texts:
            self._ensure_loaded()
            completed_inputs = 0
            for start in range(0, len(unique_miss_texts), effective_bs):
                batch = unique_miss_texts[start : start + effective_bs]
                vectors = self._forward(batch)

                for text, vec in zip(batch, vectors, strict=True):
                    key = self._cache_key(text)
                    embedding_map[text] = vec
                    self._cache[key] = vec
                    if self._cache_maxsize and len(self._cache) > self._cache_maxsize:
                        self._cache.popitem(last=False)

                completed_inputs += len(batch)
                if progress is not None:
                    frac = completed_inputs / len(unique_miss_texts)
                    done = min(int(frac * total), total)
                    progress(done, total)

        if progress is not None:
            progress(total, total)

        for orig_idx, text in zip(non_empty_indices, non_empty_texts, strict=True):
            vec = embedding_map.get(text)
            if vec is not None:
                results[orig_idx] = vec

        return results

    def embed_records(
        self,
        records: Sequence[Any],
        *,
        batch_size: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        include_signature: bool = False,
    ) -> list[DocEmbedding]:
        """Embed docstrings from symbol records, skipping empty ones gracefully."""
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
            progress=progress,
        )

        embeddings = []
        for record, text, vec in zip(valid_records, valid_texts, vectors, strict=True):
            embeddings.append(
                DocEmbedding(
                    symbol_id=getattr(record, "symbol_id", ""),
                    text=text,
                    embedding=vec,
                    source="docstring",
                )
            )
        return embeddings

    def _forward(self, texts: list[str]) -> list[np.ndarray]:
        """Run one forward pass through the SentenceTransformer."""
        raw: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        raw = _normalise(raw.astype(np.float32))
        return list(raw)

    def _cache_key(self, text: str) -> str:
        """Content-addressed key incorporating model name for safety."""
        raw = f"{self.model_name}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_stats(self) -> dict[str, int]:
        """Return cache hits, misses, and current size."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Drop all cached embeddings and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


def _pick_device() -> str:
    """Return the best available device string for SentenceTransformer."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

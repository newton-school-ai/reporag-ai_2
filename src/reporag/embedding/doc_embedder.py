"""Docstring and comment embedding pipeline.

Embeds docstrings, comments, and README sections using sentence-transformers.
Each embedding links back to its parent code symbol for cross-reference.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.reporag.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocEmbedding:
    """An embedding corresponding to a specific parent symbol."""

    symbol_id: str
    vector: np.ndarray


class DocEmbedder:
    """Embeds natural language text (docstrings, comments) into vectors.

    Features
    --------
    * **Lazy loading** -- model is downloaded/moved to device on first embed.
    * **GPU acceleration** -- auto-selects CUDA / MPS / CPU.
    * **Batching & Filtering** -- Handles batches and filters out empty texts.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name or settings.doc_embedding_model
        self.batch_size = batch_size
        self._device_pref = device

        self._model: Any | None = None
        self._device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> torch.device:
        if self._device_pref in ("cuda", "auto") and torch.cuda.is_available():
            return torch.device("cuda")
        if self._device_pref in ("mps", "auto") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        """Load SentenceTransformer model on demand."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading doc embedding model '%s' on %s", self.model_name, self.device
        )
        self._model = SentenceTransformer(self.model_name, device=str(self.device))
        self._model.eval()

    def embed_batch(
        self,
        items: Sequence[Any],
        *,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[DocEmbedding]:
        """Embed a batch of doc items (objects with text/content and id/symbol_id).

        Skips any items with empty or whitespace-only content gracefully.

        Args:
            items: Sequence of items (duck-typed for `symbol_id` and `content`).
            batch_size: Override instance default batch size.
            progress_callback: Function called with `(processed, total_valid_items)`.

        Returns:
            List of DocEmbedding objects for valid items.
        """
        valid_items: list[tuple[str, str]] = []
        for item in items:
            # Duck-type content
            content = getattr(item, "content", None) or getattr(item, "text", None)
            # Duck-type symbol id
            symbol_id = getattr(item, "symbol_id", None) or getattr(item, "id", None)

            if isinstance(item, dict):
                content = content or item.get("content") or item.get("text")
                symbol_id = symbol_id or item.get("symbol_id") or item.get("id")

            if content and str(content).strip() and symbol_id is not None:
                valid_items.append((str(symbol_id), str(content).strip()))

        total_valid = len(valid_items)
        if total_valid == 0:
            if progress_callback:
                progress_callback(0, 0)
            return []

        self._ensure_loaded()

        effective_bs = batch_size or self.batch_size
        results: list[DocEmbedding] = []
        texts = [content for _, content in valid_items]
        symbol_ids = [sid for sid, _ in valid_items]

        for i in range(0, total_valid, effective_bs):
            batch_texts = texts[i : i + effective_bs]
            batch_ids = symbol_ids[i : i + effective_bs]

            embeddings = self._model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            for sym_id, vec in zip(batch_ids, embeddings, strict=True):
                results.append(DocEmbedding(symbol_id=sym_id, vector=vec))

            if progress_callback:
                progress_callback(len(results), total_valid)

        return results

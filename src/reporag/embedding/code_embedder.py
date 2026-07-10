"""Code embedding pipeline.

Embeds code chunks using CodeBERT or UniXcoder. Produces 768-dim L2-normalized
vectors. Supports batch embedding with GPU acceleration and CPU fallback.
"""

from __future__ import annotations

import collections
import logging

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from reporag.config import settings

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Embeds code snippets using Hugging Face models (CodeBERT or UniXcoder).

    Includes caching for identical code chunks, automatic device mapping
    (CUDA, MPS, CPU), and automatic L2 normalization.
    """

    def __init__(self, model_name: str | None = None, cache_size: int = 10000):
        """Initialize the CodeEmbedder.

        Args:
            model_name: The Hugging Face model identifier. Defaults to the one in settings.
            cache_size: Maximum number of embeddings to keep in memory.
        """
        self.model_name = model_name or settings.code_embedding_model
        self.device = self._get_device()

        logger.info(f"Loading CodeEmbedder ({self.model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.cache_size = cache_size
        self._cache: collections.OrderedDict[str, np.ndarray] = (
            collections.OrderedDict()
        )

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def embed_batch(self, code_strings: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed a batch of code strings.

        Args:
            code_strings: A list of code snippets to embed.
            batch_size: Number of snippets to process at once.

        Returns:
            A numpy array of shape (N, 768) containing L2-normalized vectors.
        """
        if not code_strings:
            return np.empty((0, 768), dtype=np.float32)

        embeddings = np.empty((len(code_strings), 768), dtype=np.float32)
        uncached_indices = []
        uncached_strings = []

        # Retrieve from cache where possible
        for i, code in enumerate(code_strings):
            if code in self._cache:
                embeddings[i] = self._cache[code]
                self._cache.move_to_end(code)
            else:
                uncached_indices.append(i)
                uncached_strings.append(code)

        if not uncached_strings:
            return embeddings

        # Process uncached in batches
        for i in range(0, len(uncached_strings), batch_size):
            batch_texts = uncached_strings[i : i + batch_size]
            batch_indices = uncached_indices[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Take [CLS] token representation (index 0)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                # L2 normalize
                normalized = functional.normalize(cls_embeddings, p=2, dim=1)

                batch_embeddings = normalized.cpu().numpy()

            # Save to output array and update cache
            for idx, text, emb in zip(
                batch_indices, batch_texts, batch_embeddings, strict=True
            ):
                embeddings[idx] = emb
                self._cache[text] = emb
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        return embeddings

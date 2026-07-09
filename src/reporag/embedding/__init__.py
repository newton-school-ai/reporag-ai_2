"""Embedding module.

Turns code and documentation into dense vectors for hybrid retrieval.  The
code path (Issue 13) is implemented by :class:`CodeEmbedder`; the docstring
and index-builder paths land in Issues 14 and 15.
"""

from src.reporag.embedding.code_embedder import (
    CodeEmbedder,
    EmbeddingBackend,
    EmbeddingCache,
    HashingEmbeddingBackend,
    TransformerEmbeddingBackend,
    select_device,
    subtokenize,
)

__all__ = [
    "CodeEmbedder",
    "EmbeddingBackend",
    "EmbeddingCache",
    "HashingEmbeddingBackend",
    "TransformerEmbeddingBackend",
    "select_device",
    "subtokenize",
]

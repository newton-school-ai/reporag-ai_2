"""Cross-encoder reranker.

Scores (query, chunk) pairs using a cross-encoder model.  The cross-encoder
sees both query and document together (a single self-attention forward pass
over the concatenated ``[CLS] query [SEP] document [SEP]`` sequence), so it
can model fine-grained token-level query-document interactions that a
bi-encoder (two separate encoders + cosine similarity) cannot.  The trade-off
is latency: a cross-encoder runs one forward pass per (query, doc) pair, so
it is only ever applied to the *short* fused candidate list produced by
:mod:`reporag.retrieval.fusion` -- never to the whole corpus.

Design
------
:class:`CrossEncoderReranker` mirrors the conventions of
:class:`~reporag.embedding.doc_embedder.DocEmbedder` so the two stay
consistent and easy to test:

* **Lazy model loading** -- the model is downloaded / moved to device on the
  first :meth:`rerank` call, not at construction time (cheap, test-friendly).
  A pre-injected ``_model`` (a duck-typed object exposing ``predict``) is
  respected, making tests network-free.
* **Device auto-selection** with graceful fallback (CUDA -> MPS -> CPU),
  reusing :func:`~reporag.embedding.doc_embedder._resolve_device` so there is
  exactly one device-resolution routine in the codebase.
* **Batched scoring** -- all ``(query, chunk_text)`` pairs are scored in one
  ``predict`` call (the HuggingFace ``CrossEncoder`` API accepts a list of
  pairs), keeping the per-candidate amortised cost low enough to meet the
  "<500 ms for 20 candidates" acceptance criterion on CPU.
* **Stable, deterministic ordering** -- ties in rerank score are broken by
  the candidate's position in the input list, so two reruns of the same call
  never silently reorder results.

The output is a new list of
:class:`~reporag.retrieval.vector_search.RetrievalResult` objects whose
``score`` is overwritten with the cross-encoder ``rerank_score``; everything
else (``file_path``, ``start_line``, ``chunk_text``, ``metadata``, ...) is
preserved untouched so the reranked list is a drop-in replacement for the
input.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from reporag.config import settings
from reporag.embedding.doc_embedder import _resolve_device
from reporag.retrieval.vector_search import RetrievalResult

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks fused candidates by scoring (query, chunk) pairs with a cross-encoder.

    Args:
        model: The Hugging Face model id (e.g.
            ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``) of the cross-encoder
            to load lazily, **or** a pre-built ``CrossEncoder`` instance (or
            any duck-typed object exposing ``.predict(pairs)`` returning a
            sequence of floats).  Passing an instance is the supported seam
            for tests -- it makes the reranker network-free and avoids the
            ~100 MB model download.  Defaults to ``settings.reranker_model``.
        device: ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
            Ignored when a pre-built *model* instance is supplied (the
            instance already owns its device).
        max_length: Maximum token length per ``(query, chunk)`` pair, passed
            through to ``CrossEncoder``.  Mirrors
            :class:`~reporag.embedding.doc_embedder.DocEmbedder`'s knob so
            both models behave predictably on long code chunks.  Default 256
            matches ``ms-marco-MiniLM-L-6-v2``.

    Raises:
        ValueError: If *max_length* is not positive.
    """

    def __init__(
        self,
        model: str | Any | None = None,
        *,
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        """Initialize the reranker with a model id, instance, or config default."""
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length!r}.")

        # ``model`` may be: a str id, a pre-built instance, or None (load the
        # configured default lazily).  We stash the raw value and resolve it
        # on first ``rerank`` call to keep construction side-effect-free.
        self._model_ref: str | Any | None = model
        self._model_name: str = (
            model if isinstance(model, str) else settings.reranker_model
        )
        self._resolved_model: Any | None = model if not isinstance(model, str) else None
        self._device = _resolve_device(device)
        self.max_length = max_length
        self._loaded = False

    @property
    def device(self) -> torch.device:
        """The resolved compute device."""
        return self._device

    @property
    def model_name(self) -> str:
        """The configured cross-encoder model id."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """True once the model has been downloaded / moved to device."""
        return self._loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Resolve and (if needed) move the model to the target device.

        Called automatically before the first rerank; subsequent calls are a
        fast no-op.  A pre-injected ``_resolved_model`` is respected, making
        tests network-free exactly like
        :meth:`~reporag.embedding.doc_embedder.DocEmbedder._ensure_loaded`.
        """
        if self._loaded:
            return

        if self._resolved_model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder reranker model '%s' on %s",
                self._model_name,
                self._device,
            )
            self._resolved_model = CrossEncoder(
                self._model_name,
                device=str(self._device),
                max_length=self.max_length,
            )

        self._loaded = True

    @property
    def model(self) -> Any:
        """The resolved cross-encoder model instance (loaded lazily on access)."""
        self._ensure_loaded()
        return self._resolved_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank *candidates* by cross-encoder relevance to *query*.

        Each candidate's ``chunk_text`` is paired with *query* and scored in
        a single batched ``model.predict`` call.  Results are sorted by
        ``rerank_score`` descending; ties are broken by the candidate's
        original position in *candidates* so the ordering is deterministic.
        The returned objects are **copies** of the inputs (the caller's list
        and result objects are never mutated); each copy's ``score`` field is
        set to the cross-encoder score and a ``"rerank_score"`` key is also
        added to ``metadata`` for callers that inspect metadata payloads.

        Args:
            query: The natural-language or code query to score against.
            candidates: The candidate results to rerank.  Typically the
                top-N (e.g. 20) of the RRF-fused ranking from
                :func:`~reporag.retrieval.fusion.reciprocal_rank_fusion`.
            top_k: If given, return at most this many reranked results.
                ``None`` (default) returns the full reranked list. Must be
                ``>= 1`` when provided.

        Returns:
            Up to *top_k* (or ``len(candidates)``) new
            :class:`~reporag.retrieval.vector_search.RetrievalResult`
            objects sorted by cross-encoder score descending.  An empty
            *candidates* list returns ``[]`` without loading the model.

        Raises:
            ValueError: If *top_k* (when given) is ``< 1``.
        """
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k!r}.")

        if not candidates:
            # Skip the (potentially expensive) model load entirely when there
            # is nothing to score -- keeps the empty path side-effect-free.
            return []

        self._ensure_loaded()

        pairs = [(query, c.chunk_text) for c in candidates]
        # ``CrossEncoder.predict`` returns a 1-D sequence of floats aligned
        # with the input pairs.  Duck-typed fakes only need to return
        # something indexable/iterable of the right length.
        raw_scores = self.model.predict(pairs)
        scores = [float(s) for s in raw_scores]

        # Decorate index pairs with their scores so we can sort without losing
        # the original positions we need for the deterministic tiebreak below.
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (-scores[pair[0]], pair[0]),
        )

        reranked: list[RetrievalResult] = []
        for original_index, candidate in ranked:
            score = scores[original_index]
            new_metadata = dict(candidate.metadata)
            new_metadata["rerank_score"] = score
            reranked.append(
                RetrievalResult(
                    score=score,
                    file_path=candidate.file_path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    symbol_name=candidate.symbol_name,
                    chunk_text=candidate.chunk_text,
                    metadata=new_metadata,
                )
            )

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model={self._model_name!r}, "
            f"device={self._device}, loaded={self._loaded})"
        )

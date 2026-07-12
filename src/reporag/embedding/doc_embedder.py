"""Docstring, comment, and README embedding pipeline.

Embeds *natural-language* documentation -- function/class docstrings, inline
comments, and README sections -- into 384-dim L2-normalised vectors using the
``sentence-transformers/all-MiniLM-L6-v2`` model.

Why a separate embedder from :mod:`~reporag.embedding.code_embedder`?
--------------------------------------------------------------------
Docstrings and comments describe *intent* in prose.  Embedding them with a
model tuned for natural-language similarity lets a query phrased in English
("how does auth work?") match documentation even when the code itself uses
different identifiers.  Each doc embedding is linked back to its parent code
symbol (via :attr:`Symbol.qualified_name`) so retrieval can surface the exact
function or class the prose describes.

Design
------
The low-level embedding engine deliberately mirrors
:class:`~reporag.embedding.code_embedder.CodeEmbedder` (lazy model loading,
attention-mask-weighted mean pooling, GPU/MPS/CPU fallback, a bounded LRU cache,
and in-batch deduplication).  ``all-MiniLM-L6-v2`` is a mean-pooling model, so
using ``transformers`` directly with mean pooling reproduces the
sentence-transformers output while giving us full control over batching,
caching, and device placement -- and keeping unit tests network-free via
injected fakes.

On top of that engine sits the documentation layer that makes this module
distinct:

* :meth:`DocEmbedder.embed_symbols`  -- docstrings, linked to their symbol.
* :meth:`DocEmbedder.embed_comments` -- inline comments (via stdlib
  :mod:`tokenize`, so ``#`` inside string literals is never misread), optionally
  linked to the enclosing symbol.
* :meth:`DocEmbedder.embed_readme`   -- markdown split into heading-scoped
  sections.

Downstream integration
----------------------
Every method returns :class:`DocEmbedding` records whose :meth:`~DocEmbedding.to_dict`
yields a JSON-serialisable payload ready for the Qdrant ``reporag_docs``
collection::

    from reporag.embedding.doc_embedder import DocEmbedder
    from reporag.ingestion.symbol_extractor import SymbolExtractor

    symbols = SymbolExtractor().extract_from_file("src/auth.py")
    docs = DocEmbedder().embed_symbols(symbols)
    payloads = [d.to_dict() for d in docs]   # -> Qdrant upsert
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import tokenize
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from reporag.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
"""Hidden-size of ``sentence-transformers/all-MiniLM-L6-v2``."""

DocType = Literal["docstring", "comment", "readme"]
"""Provenance of a :class:`DocEmbedding` -- where the prose came from."""

# Progress is reported as ``callback(completed_items, total_items)``.
ProgressCallback = Callable[[int, int], None]

# Collapses any run of whitespace (including newlines) to a single space.
_WHITESPACE_RE = re.compile(r"\s+")

# A markdown ATX heading: 1-6 leading '#', a space, then the heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

# Opening/closing markdown code fence (``` or ~~~), possibly indented.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


# ---------------------------------------------------------------------------
# Module-level helpers (pure, unit-testable without a model)
# ---------------------------------------------------------------------------


def _resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available accelerator.

    Resolution order for ``"auto"``: CUDA -> MPS (Apple Silicon) -> CPU.  An
    explicit ``"cuda"`` / ``"mps"`` request is honoured only when that backend
    is actually available, otherwise it falls back to CPU so the embedder never
    crashes on a machine without a GPU.
    """
    if preference in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace runs and strip the ends.

    Docstrings and comments are frequently indented and line-wrapped; the raw
    layout carries no semantic signal for the embedding model, so we normalise
    to a single-spaced form.  This also makes the cache key stable across
    inputs that differ only in formatting.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_meaningful(text: str) -> bool:
    """Return ``True`` when *text* is worth embedding.

    Guards the "skip empty docstrings" requirement: a string is meaningful only
    if it contains at least one alphanumeric character after normalisation.
    Whitespace-only, empty, or pure-punctuation prose (``"..."``, ``"---"``) is
    skipped so we never spend a forward pass -- or a vector slot -- on noise.
    """
    return any(ch.isalnum() for ch in text)


@dataclass(frozen=True)
class Comment:
    """An inline comment (or a run of adjacent comment lines) from source."""

    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ReadmeSection:
    """A heading-scoped section of a markdown document."""

    heading: str | None
    level: int
    text: str
    start_line: int


def _flatten_symbols(symbols: Sequence[Any]) -> Iterator[Any]:
    """Yield every symbol in *symbols*, descending into methods and children.

    Duck-typed on the :class:`~reporag.ingestion.symbol_extractor.Symbol`
    dataclass (``.methods`` / ``.children`` lists) so this module never has to
    import the symbol layer at runtime -- mirroring how ``CodeEmbedder``
    duck-types on ``Chunk.content``.  This keeps the two subsystems decoupled
    and the unit tests free of tree-sitter.
    """
    stack: list[Any] = list(symbols)
    while stack:
        symbol = stack.pop()
        yield symbol
        stack.extend(getattr(symbol, "methods", None) or [])
        stack.extend(getattr(symbol, "children", None) or [])


def iter_symbol_docstrings(symbols: Sequence[Any]) -> Iterator[tuple[Any, str]]:
    """Yield ``(symbol, docstring)`` for every symbol carrying real prose.

    Symbols with a ``None``, empty, or non-meaningful docstring are skipped, so
    the caller only ever sees documentation that is worth embedding.
    """
    for symbol in _flatten_symbols(symbols):
        raw = getattr(symbol, "docstring", None)
        if not raw:
            continue
        normalised = _normalize_whitespace(raw)
        if _is_meaningful(normalised):
            yield symbol, normalised


def extract_python_comments(source: str) -> list[Comment]:
    """Extract inline comments from Python *source*.

    Uses the standard-library :mod:`tokenize`, which understands Python's
    lexical grammar -- so a ``#`` inside a string literal is never mistaken for
    a comment, and shebang lines / type comments are captured faithfully.

    Adjacent *own-line* comments are merged into a single :class:`Comment` block
    so a multi-line explanation becomes one coherent embedding unit.  A
    *trailing* comment (one that follows code on the same line, e.g.
    ``x = 1  # note``) is always emitted on its own, since it annotates that
    statement rather than continuing a comment paragraph.

    Tolerant of malformed input: a :class:`tokenize.TokenError` or
    ``IndentationError`` on truncated/invalid source returns whatever comments
    were recovered before the failure, rather than raising.
    """
    comments: list[Comment] = []
    lines = source.splitlines()

    def _is_own_line(row: int, col: int) -> bool:
        """True when only whitespace precedes the comment on its own line."""
        if 1 <= row <= len(lines):
            return not lines[row - 1][:col].strip()
        return True

    pending: list[str] = []
    pending_start = 0
    pending_end = 0

    def _flush() -> None:
        nonlocal pending
        if pending:
            text = _normalize_whitespace(" ".join(pending))
            if _is_meaningful(text):
                comments.append(
                    Comment(text=text, start_line=pending_start, end_line=pending_end)
                )
            pending = []

    def _emit_single(body: str, start: int, end: int) -> None:
        text = _normalize_whitespace(body)
        if _is_meaningful(text):
            comments.append(Comment(text=text, start_line=start, end_line=end))

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            # Strip the leading '#'(s) and surrounding spaces: "## note" -> "note".
            body = tok.string.lstrip("#").strip()
            row = tok.start[0]

            if not _is_own_line(row, tok.start[1]):
                # Trailing inline comment: stands alone, breaks any open block.
                _flush()
                _emit_single(body, row, tok.end[0])
            elif pending and row == pending_end + 1:
                pending.append(body)
                pending_end = row
            else:
                _flush()
                pending = [body]
                pending_start = pending_end = row
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        logger.debug(
            "Partial comment extraction (%s); returning recovered comments", exc
        )

    _flush()
    return comments


def split_readme_sections(markdown: str) -> list[ReadmeSection]:
    """Split *markdown* into heading-scoped sections.

    Each ATX heading (``#`` .. ``######``) starts a new section that runs until
    the next heading; the section text is the heading followed by its body, so
    the embedding carries both the title and its explanation.  Any preamble
    before the first heading becomes a leading section with ``heading=None``.

    Headings inside fenced code blocks (```` ``` ````/``~~~``) are ignored so a
    commented ``# shell prompt`` in an example is not treated as a section
    boundary.  Empty sections (a heading with no meaningful body and no title
    text) are dropped.
    """
    sections: list[ReadmeSection] = []
    heading: str | None = None
    level = 0
    start_line = 1
    body: list[str] = []
    in_fence = False

    def _flush() -> None:
        nonlocal heading, level, body
        parts: list[str] = []
        if heading:
            parts.append(heading)
        parts.append("\n".join(body))
        text = _normalize_whitespace("\n".join(p for p in parts if p))
        if _is_meaningful(text):
            sections.append(
                ReadmeSection(
                    heading=heading, level=level, text=text, start_line=start_line
                )
            )
        heading, level, body = None, 0, []

    for lineno, line in enumerate(markdown.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            _flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            start_line = lineno
        else:
            body.append(line)

    _flush()
    return sections


# ---------------------------------------------------------------------------
# DocEmbedding record
# ---------------------------------------------------------------------------


@dataclass
class DocEmbedding:
    """A single documentation embedding linked to its source location.

    Attributes:
        text:         The normalised natural-language text that was embedded.
        vector:       The ``(384,)`` float32, L2-normalised embedding.
        doc_type:     ``"docstring"``, ``"comment"``, or ``"readme"``.
        symbol_id:    Qualified name of the parent code symbol this prose
                      describes (``None`` for README sections or unlinked
                      comments).  Mirrors :attr:`Symbol.qualified_name` so doc
                      embeddings join back to symbols and code chunks.
        file_path:    Source file the prose came from.
        start_line:   1-based line where the prose begins.
        end_line:     1-based line where the prose ends (inclusive).
        metadata:     Free-form extras (e.g. ``{"heading": ...}`` for README
                      sections, ``{"symbol_type": "function"}`` for docstrings).
    """

    text: str
    vector: np.ndarray
    doc_type: DocType
    symbol_id: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable payload (vector as a plain list).

        Suitable as a Qdrant point payload or a JSONL record: every value is a
        JSON primitive, and the numpy vector is converted to a ``list[float]``.
        """
        return {
            "text": self.text,
            "vector": self.vector.astype(np.float32).tolist(),
            "doc_type": self.doc_type,
            "symbol_id": self.symbol_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        link = f" -> {self.symbol_id}" if self.symbol_id else ""
        return (
            f"DocEmbedding({self.doc_type}{link} "
            f"[{self.file_path}:{self.start_line}] {self.text[:40]!r})"
        )


# ---------------------------------------------------------------------------
# DocEmbedder
# ---------------------------------------------------------------------------


class DocEmbedder:
    """Embeds natural-language documentation into 384-dim vectors.

    Features
    --------
    * **Lazy loading** -- the model is downloaded / moved to device on the first
      ``embed`` call, not at construction time (cheap, test-friendly).
    * **Mean pooling** -- attention-mask-weighted mean over token embeddings,
      the pooling ``all-MiniLM-L6-v2`` was trained with.
    * **GPU acceleration** -- auto-selects CUDA / MPS / CPU with graceful
      fallback.
    * **LRU cache + in-batch dedup** -- content-addressed, bounded, so repeated
      docstrings (common across overloads / copies) cost one forward pass.
    * **Empty-skip** -- whitespace-only / punctuation-only text is never fed to
      the model; it maps to a zero vector.
    * **Documentation layer** -- :meth:`embed_symbols`, :meth:`embed_comments`,
      and :meth:`embed_readme` produce :class:`DocEmbedding` records linked to
      their originating code symbol.

    Args:
        model_name:    Hugging Face model id.  Defaults to
                       ``settings.doc_embedding_model``.
        device:        ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        batch_size:    Default mini-batch size for inference.
        max_length:    Maximum token length (``all-MiniLM-L6-v2`` truncates at
                       256).
        cache_maxsize: Upper bound on cached embeddings (0 disables the cache).
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
        """Output vector dimensionality (384)."""
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download (if needed) and move the model to the target device.

        Called automatically before the first forward pass; subsequent calls
        are a fast no-op.  A pre-injected ``_tokenizer`` / ``_model`` is
        respected, making tests network-free.
        """
        if self._loaded:
            return

        if self._tokenizer is None or self._model is None:
            from transformers import AutoModel, AutoTokenizer

            logger.info(
                "Loading doc embedding model '%s' on %s", self.model_name, self._device
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
        """Embed a single string, returning a ``(384,)`` vector."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Embed a batch of natural-language strings.

        Within a single call, duplicate texts are computed only once and shared
        across positions; previously seen texts are served from the LRU cache.
        Whitespace-only strings map to a zero vector without a forward pass, so
        empty docstrings are never sent to the model.

        Args:
            texts:             Strings to embed.
            batch_size:        Override the instance default for this call.
            progress_callback: Called as ``callback(completed, total)`` after
                               each mini-batch (and once up-front), where
                               *total* is ``len(texts)``.

        Returns:
            A ``(len(texts), 384)`` float32 array, L2-normalised row-wise.
        """
        cleaned = [str(t) for t in texts]
        total = len(cleaned)
        if total == 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        effective_bs = batch_size or self.batch_size
        results: list[np.ndarray | None] = [None] * total

        # --- Phase 1: resolve empties, cache hits, and unique misses ---------
        unique_miss_texts: list[str] = []
        miss_key_to_positions: dict[str, list[int]] = {}
        completed = 0

        for i, text in enumerate(cleaned):
            if not text.strip():
                # Never embed empty/whitespace text: emit a zero vector.
                results[i] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                completed += 1
                continue

            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                self._cache.move_to_end(key)
                self._hits += 1
                completed += 1
            else:
                self._misses += 1
                if key not in miss_key_to_positions:
                    miss_key_to_positions[key] = []
                    unique_miss_texts.append(text)
                miss_key_to_positions[key].append(i)

        if progress_callback is not None:
            progress_callback(completed, total)

        # --- Phase 2: batch-compute unique misses ---------------------------
        if unique_miss_texts:
            self._ensure_loaded()
            for start in range(0, len(unique_miss_texts), effective_bs):
                batch_texts = unique_miss_texts[start : start + effective_bs]
                vectors = self._forward(batch_texts)

                for text, vec in zip(batch_texts, vectors, strict=True):
                    key = self._cache_key(text)
                    positions = miss_key_to_positions[key]
                    for pos in positions:
                        results[pos] = vec
                    completed += len(positions)
                    self._cache[key] = vec
                    if self._cache_maxsize and len(self._cache) > self._cache_maxsize:
                        self._cache.popitem(last=False)

                if progress_callback is not None:
                    progress_callback(min(completed, total), total)

        return np.stack(results, dtype=np.float32)  # type: ignore[arg-type]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two strings.

        Embeddings are L2-normalised, so this is a dot product.  A zero vector
        (empty input) yields ``0.0``.
        """
        vecs = self.embed_batch([a, b])
        return float(np.dot(vecs[0], vecs[1]))

    # ------------------------------------------------------------------
    # Documentation layer -- linked DocEmbedding records
    # ------------------------------------------------------------------

    def embed_symbols(
        self,
        symbols: Sequence[Any],
        *,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[DocEmbedding]:
        """Embed the docstrings of *symbols*, linked to each parent symbol.

        Walks the symbol tree (methods and nested children included), skips
        symbols whose docstring is missing or empty, and returns one
        :class:`DocEmbedding` per remaining docstring with ``symbol_id`` set to
        the symbol's :attr:`~Symbol.qualified_name`.
        """
        pairs = list(iter_symbol_docstrings(symbols))
        if not pairs:
            return []

        vectors = self.embed_batch(
            [text for _, text in pairs],
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        embeddings: list[DocEmbedding] = []
        for (symbol, text), vector in zip(pairs, vectors, strict=True):
            symbol_id = getattr(symbol, "qualified_name", None) or getattr(
                symbol, "name", None
            )
            embeddings.append(
                DocEmbedding(
                    text=text,
                    vector=vector,
                    doc_type="docstring",
                    symbol_id=symbol_id,
                    file_path=getattr(symbol, "file_path", None),
                    start_line=getattr(symbol, "start_line", None),
                    end_line=getattr(symbol, "end_line", None),
                    metadata={"symbol_type": getattr(symbol, "type", None)},
                )
            )
        return embeddings

    def embed_comments(
        self,
        source: str,
        *,
        file_path: str | None = None,
        symbols: Sequence[Any] | None = None,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[DocEmbedding]:
        """Embed the inline comments of Python *source*.

        Comments are extracted with :func:`extract_python_comments` (so ``#``
        inside strings is ignored) and adjacent comment lines are merged into
        one block.  When *symbols* is provided, each comment is linked to the
        smallest enclosing symbol whose line range contains it, giving comments
        the same ``symbol_id`` provenance as docstrings.
        """
        comments = extract_python_comments(source)
        if not comments:
            return []

        ranges = _symbol_line_ranges(symbols) if symbols else []
        vectors = self.embed_batch(
            [c.text for c in comments],
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        embeddings: list[DocEmbedding] = []
        for comment, vector in zip(comments, vectors, strict=True):
            symbol_id = _enclosing_symbol(comment.start_line, ranges)
            embeddings.append(
                DocEmbedding(
                    text=comment.text,
                    vector=vector,
                    doc_type="comment",
                    symbol_id=symbol_id,
                    file_path=file_path,
                    start_line=comment.start_line,
                    end_line=comment.end_line,
                )
            )
        return embeddings

    def embed_readme(
        self,
        markdown: str,
        *,
        file_path: str | None = "README.md",
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[DocEmbedding]:
        """Embed a README, one :class:`DocEmbedding` per heading-scoped section.

        README sections are not tied to a code symbol, so ``symbol_id`` is
        ``None``; the section heading is preserved in ``metadata["heading"]``
        for display and filtering.
        """
        sections = split_readme_sections(markdown)
        if not sections:
            return []

        vectors = self.embed_batch(
            [s.text for s in sections],
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        embeddings: list[DocEmbedding] = []
        for section, vector in zip(sections, vectors, strict=True):
            embeddings.append(
                DocEmbedding(
                    text=section.text,
                    vector=vector,
                    doc_type="readme",
                    symbol_id=None,
                    file_path=file_path,
                    start_line=section.start_line,
                    end_line=section.start_line,
                    metadata={"heading": section.heading, "level": section.level},
                )
            )
        return embeddings

    # ------------------------------------------------------------------
    # Model forward pass
    # ------------------------------------------------------------------

    def _forward(self, texts: list[str]) -> list[np.ndarray]:
        """Run one forward pass and return L2-normalised ``(384,)`` vectors."""
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

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over the token dimension.

        Padding tokens are excluded from the average so variable-length inputs
        produce faithful representations.  This is the pooling strategy
        ``all-MiniLM-L6-v2`` was trained with.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """Content-addressed key incorporating the model name for safety."""
        raw = f"{self.model_name}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_stats(self) -> dict[str, int]:
        """Return ``{"hits": ..., "misses": ..., "size": ...}`` counters."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

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


# ---------------------------------------------------------------------------
# Comment -> symbol linking helpers
# ---------------------------------------------------------------------------


def _symbol_line_ranges(symbols: Sequence[Any]) -> list[tuple[int, int, str]]:
    """Return ``(start_line, end_line, symbol_id)`` for every symbol.

    Sorted by span width (narrowest last) so the *smallest* enclosing symbol
    wins when a comment falls inside nested scopes.
    """
    ranges: list[tuple[int, int, str]] = []
    for symbol in _flatten_symbols(symbols):
        start = getattr(symbol, "start_line", None)
        end = getattr(symbol, "end_line", None)
        symbol_id = getattr(symbol, "qualified_name", None) or getattr(
            symbol, "name", None
        )
        if start is not None and end is not None and symbol_id:
            ranges.append((start, end, symbol_id))
    # Widest spans first so the tightest enclosing scope is chosen last.
    ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
    return ranges


def _enclosing_symbol(line: int, ranges: Sequence[tuple[int, int, str]]) -> str | None:
    """Return the id of the narrowest symbol whose range contains *line*."""
    match: str | None = None
    for start, end, symbol_id in ranges:
        if start <= line <= end:
            match = symbol_id  # later (narrower) matches override earlier ones
    return match

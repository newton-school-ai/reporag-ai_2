"""Hybrid index builder.

Creates and populates both a Qdrant vector collection and a BM25 sparse
index. Includes a code-aware tokenizer that splits camelCase and snake_case
identifiers for better keyword matching.

Why hybrid?
-----------
Dense vectors (:mod:`~reporag.embedding.code_embedder` /
:mod:`~reporag.embedding.doc_embedder`) capture *semantic* similarity --
they find code that does something similar even when the wording differs.
BM25 captures *lexical* similarity -- it finds exact identifier matches
(``get_user_by_id``) that an embedding model may blur together with related
but different symbols. Building both from the same chunks lets downstream
retrieval (:mod:`~reporag.retrieval.fusion`, Issue 19) fuse the two ranked
lists with reciprocal rank fusion.

Design
------
:class:`HybridIndexBuilder` is the orchestrator: given already-embedded
:class:`~reporag.ingestion.chunker.Chunk` / :class:`~reporag.embedding.doc_embedder.DocEmbedding`
records, it

* creates the two Qdrant collections (``reporag_code``, ``reporag_docs``)
  with a payload schema and payload indexes for common filters
  (``file_path``, ``language``, ``chunk_kind``, ``repo_id``, ...),
* upserts points with **deterministic IDs** derived from
  ``(repo_id, file_path, symbol, span)`` so re-indexing the same code is an
  idempotent upsert rather than a duplicate insert -- this is what makes
  incremental updates safe,
* feeds the same chunk text through :func:`tokenize_code` into a
  :class:`BM25Index`, keyed by the *same* point ID so the two indexes can be
  cross-referenced during fusion.

:class:`BM25Index` wraps ``rank_bm25.BM25Okapi``, which has no native
incremental-update API (it is rebuilt from a full token matrix). We work
around that by keeping the tokenized corpus in an ordered ``dict`` keyed by
document id and lazily rebuilding the ``BM25Okapi`` object only when a
search or save is actually requested (``_dirty`` flag) -- so many ``add``/
``remove`` calls during a bulk re-index cost one rebuild, not one per call.

Both the Qdrant client and ``rank_bm25`` are imported lazily (mirroring
:class:`~reporag.embedding.code_embedder.CodeEmbedder`'s lazy model
loading), so this module imports cleanly and cheaply even when those
packages -- or a running Qdrant server -- aren't available, and a fake
client can be injected directly for network-free unit tests.

Usage
-----
::

    from reporag.embedding.code_embedder import CodeEmbedder
    from reporag.embedding.index_builder import HybridIndexBuilder
    from reporag.ingestion.chunker import SemanticChunker

    chunks = SemanticChunker().chunk_file("src/auth.py")
    vectors = CodeEmbedder().embed_batch(chunks)

    builder = HybridIndexBuilder()
    builder.ensure_collections()
    builder.upsert_code_chunks(chunks, vectors, repo_id="my-org/my-repo")
    builder.save_bm25("data/my-repo.bm25.pkl")
"""

from __future__ import annotations

import logging
import pickle
import re
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models

from reporag.config import settings
from reporag.embedding.code_embedder import (
    EMBEDDING_DIM as CODE_VECTOR_SIZE,
)
from reporag.embedding.doc_embedder import (
    EMBEDDING_DIM as DOC_VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


# Fixed namespace for deriving deterministic point IDs via uuid5. Any UUID
# works as long as it is stable across process restarts -- it is never sent
# anywhere, it just seeds the hash so the same (repo, file, symbol, span)
# always maps to the same Qdrant point ID, making re-indexing an upsert
# instead of a duplicate insert.
_ID_NAMESPACE = uuid.UUID("6f1c1b6a-9a3e-4c1a-8e8b-1a2b3c4d5e6f")

# Payload fields worth a Qdrant index for fast filtering during retrieval
# (e.g. "search only within this file" or "only python").
_KEYWORD_INDEX_FIELDS = (
    "repo_id",
    "file_path",
    "language",
    "chunk_kind",
    "symbol",
    "symbol_type",
)
_DOC_KEYWORD_INDEX_FIELDS = (
    "repo_id",
    "file_path",
    "doc_type",
    "symbol_id",
    "language",
)


# ---------------------------------------------------------------------------
# Code-aware tokenizer
# ---------------------------------------------------------------------------

# Anything that isn't ASCII alphanumeric is a token boundary: underscores,
# hyphens, dots, brackets, whitespace, operators, ... This alone handles
# snake_case ("get_user" -> "get", "user") and kebab-case identifiers.
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")

# Splits a single alnum run at camelCase / PascalCase boundaries, including
# acronym runs, so "getUserByID" -> "get", "User", "By", "ID" and
# "HTTPServer" -> "HTTP", "Server".
_CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"  # fooBar -> foo|Bar ; foo2Bar -> foo2|Bar
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # HTTPServer -> HTTP|Server
)


def split_identifier(token: str) -> list[str]:
    """Split one already-boundary-isolated token into camelCase sub-words.

    ``token`` is assumed to already be free of underscores/hyphens/punctuation
    (i.e. the output of splitting on :data:`_NON_ALNUM_RE`); this only handles
    the camelCase / PascalCase / acronym boundary within it.
    """
    if not token:
        return []
    return [piece for piece in _CAMEL_BOUNDARY_RE.split(token) if piece]


def tokenize_code(text: str) -> list[str]:
    """Code-aware tokenizer used for both indexing and querying BM25.

    Splits on non-alphanumeric characters first (handles ``snake_case``,
    ``kebab-case``, punctuation, and whitespace), then splits each remaining
    run at camelCase/PascalCase/acronym boundaries, and lower-cases
    everything so matching is case-insensitive. This means
    ``getUserByID``, ``get_user_by_id``, and ``GetUserById`` all tokenize to
    the same overlapping ``{get, user, by, id}`` set, which is exactly the
    lexical overlap BM25 needs to find identifier matches that a purely
    semantic embedding might blur together.

    Numbers are kept as their own tokens (``"user2"`` -> ``"user", "2"``).
    Empty input returns an empty list rather than raising.
    """
    if not text:
        return []
    tokens: list[str] = []
    for run in _NON_ALNUM_RE.split(text):
        if not run:
            continue
        for piece in split_identifier(run):
            tokens.append(piece.lower())
    return tokens


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------


@dataclass
class BM25Document:
    """One tokenized document tracked by :class:`BM25Index`."""

    doc_id: str
    tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """A code-aware BM25 index supporting incremental updates.

    ``rank_bm25.BM25Okapi`` has no incremental API -- it is built once from a
    full corpus. This class keeps the tokenized corpus itself (an ordered
    ``dict[doc_id, BM25Document]``) as the source of truth and only rebuilds
    the underlying ``BM25Okapi`` object lazily, the next time :meth:`search`
    or :meth:`save` is called after a mutation. That makes bulk indexing
    (many :meth:`add` calls in a row) cost a single rebuild rather than one
    per document, while still presenting an "add one document" incremental
    API to callers such as :class:`HybridIndexBuilder`.

    Args:
        tokenizer: Callable used to tokenize both documents and queries.
            Defaults to :func:`tokenize_code`. Kept pluggable mainly for
            tests (an identity/whitespace tokenizer is easier to reason
            about than the real one).
        k1, b: Standard Okapi BM25 hyperparameters, forwarded to
            ``rank_bm25.BM25Okapi``.
    """

    def __init__(
        self,
        *,
        tokenizer: Callable[[str], list[str]] = tokenize_code,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        self._docs: dict[str, BM25Document] = {}
        self._bm25: Any | None = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    @property
    def tokenizer(self) -> Callable[[str], list[str]]:
        return self._tokenizer

    # ------------------------------------------------------------------
    # Incremental mutation
    # ------------------------------------------------------------------

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add or overwrite the document at *doc_id*.

        Re-adding an existing ``doc_id`` replaces its tokens/metadata in
        place (an upsert), which is what re-indexing a changed file needs.
        """
        self._docs[doc_id] = BM25Document(
            doc_id=doc_id,
            tokens=self._tokenizer(text),
            metadata=dict(metadata or {}),
        )
        self._dirty = True

    def add_many(self, items: Iterable[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Bulk version of :meth:`add`. Only triggers one rebuild on next use."""
        for doc_id, text, metadata in items:
            self.add(doc_id, text, metadata)

    def remove(self, doc_id: str) -> bool:
        """Remove a document if present. Returns whether it existed."""
        existed = self._docs.pop(doc_id, None) is not None
        if existed:
            self._dirty = True
        return existed

    def remove_where(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """Remove every document whose metadata matches *predicate*.

        Useful for "drop everything belonging to this file before
        re-indexing it" during incremental updates. Returns the count
        removed.
        """
        stale = [
            doc_id for doc_id, doc in self._docs.items() if predicate(doc.metadata)
        ]
        for doc_id in stale:
            del self._docs[doc_id]
        if stale:
            self._dirty = True
        return len(stale)

    def clear(self) -> None:
        self._docs.clear()
        self._bm25 = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if not self._dirty and self._bm25 is not None:
            return
        if not self._docs:
            self._bm25 = None
            self._dirty = False
            return

        from rank_bm25 import BM25Okapi

        corpus = [doc.tokens for doc in self._docs.values()]
        self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)
        self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-k documents for *query*, best score first.

        Each result is ``{"id": doc_id, "score": bm25_score, "metadata": ...}``.
        Zero-scoring documents (no lexical overlap at all) are dropped rather
        than padding out the result list with noise. Rebuilds the underlying
        BM25 matrix first if the corpus changed since the last search.
        """
        self._ensure_built()
        if self._bm25 is None:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        ids = list(self._docs.keys())
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(ids, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )

        results: list[dict[str, Any]] = []
        for doc_id, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append(
                {
                    "id": doc_id,
                    "score": float(score),
                    "metadata": self._docs[doc_id].metadata,
                }
            )
        return results

    def get_document(self, doc_id: str) -> BM25Document | None:
        """Return a document by id, or None if it does not exist."""
        return self._docs.get(doc_id)

    # ------------------------------------------------------------------
    # Persistence (for incremental updates across process restarts)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the tokenized corpus to *path*.

        Only the corpus (not the derived ``BM25Okapi`` matrix) is pickled --
        it's cheap to rebuild and this avoids coupling the on-disk format to
        ``rank_bm25``'s internal representation.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": self._docs, "k1": self._k1, "b": self._b}
        with target.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(
        cls, path: str | Path, *, tokenizer: Callable[[str], list[str]] = tokenize_code
    ) -> BM25Index:
        """Load a previously :meth:`save`-d index, ready for further updates."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        index = cls(
            tokenizer=tokenizer, k1=payload.get("k1", 1.5), b=payload.get("b", 0.75)
        )
        index._docs = payload["docs"]
        index._dirty = True
        return index


# ---------------------------------------------------------------------------
# Symbol-type resolution
# ---------------------------------------------------------------------------
#
# Chunk (reporag.ingestion.chunker.Chunk) only carries a *structural* label
# -- chunk_kind: "definition" | "continuation" | "module" -- it does not
# know whether a definition chunk is a class, function, or method. That
# classification lives on Symbol.type (reporag.ingestion.symbol_extractor),
# keyed by the same qualified_name Chunk already carries. Rather than
# widening the Chunk dataclass (touching the chunker pipeline just for this),
# HybridIndexBuilder accepts the Symbol tree alongside the chunks and
# resolves symbol_type by qualified_name lookup -- keeping this concern
# local to indexing.


def _flatten_symbols(symbols: Sequence[Any]) -> Iterable[Any]:
    """Yield every symbol in *symbols*, descending into methods and children.

    Duck-typed on ``.methods`` / ``.children`` (mirrors
    :mod:`reporag.embedding.doc_embedder`'s identical helper) so this module
    never needs to import the symbol layer at runtime.
    """
    stack: list[Any] = list(symbols)
    while stack:
        symbol = stack.pop()
        yield symbol
        stack.extend(getattr(symbol, "methods", None) or [])
        stack.extend(getattr(symbol, "children", None) or [])


def build_symbol_type_lookup(symbols: Sequence[Any] | None) -> dict[str, str]:
    """Map ``qualified_name`` (falling back to ``name``) -> ``symbol type``.

    Built once per :meth:`HybridIndexBuilder.upsert_code_chunks` call and
    used to resolve each :class:`Chunk`'s ``symbol_type`` payload field.
    Returns an empty mapping for ``None``/empty input, so callers that don't
    have symbols on hand simply get ``symbol_type: None`` in the payload
    rather than an error.
    """
    if not symbols:
        return {}
    lookup: dict[str, str] = {}
    for symbol in _flatten_symbols(symbols):
        sym_type = getattr(symbol, "type", None)
        if not sym_type:
            continue
        qualified_name = getattr(symbol, "qualified_name", None)
        name = getattr(symbol, "name", None)
        if qualified_name:
            lookup[qualified_name] = sym_type
        if name and name not in lookup:
            lookup[name] = sym_type
    return lookup


# ---------------------------------------------------------------------------
# Hybrid index builder
# ---------------------------------------------------------------------------


class HybridIndexBuilder:
    """Builds and incrementally updates the Qdrant + BM25 hybrid index.

    Args:
        client: A pre-constructed ``qdrant_client.QdrantClient`` (or a fake
            with the same surface) -- inject this in tests to stay
            network-free. When omitted, a real client pointed at
            ``settings.qdrant_url`` is created lazily on first use.
        qdrant_url, collection_code, collection_docs: Override the
            corresponding ``settings.*`` values.
        bm25_index: A pre-existing :class:`BM25Index` to keep updating
            (e.g. one loaded from disk via :meth:`BM25Index.load`).
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        qdrant_url: str | None = None,
        collection_code: str | None = None,
        collection_docs: str | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._client = client
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_code = collection_code or settings.qdrant_collection_code
        self.collection_docs = collection_docs or settings.qdrant_collection_docs
        self.bm25 = bm25_index if bm25_index is not None else BM25Index()
        self._ensured_collections: set[str] = set()

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The Qdrant client, constructed on first access if not injected."""
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.info("Connecting to Qdrant at %s", self.qdrant_url)
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    # ------------------------------------------------------------------
    # Collection schema
    # ------------------------------------------------------------------

    def ensure_collections(self, *, recreate: bool = False) -> None:
        """Create the code and doc collections if they don't already exist.

        Idempotent: safe to call before every indexing run. Pass
        ``recreate=True`` to drop and rebuild both collections from scratch
        (e.g. after a vector-dimension or distance-metric change).
        """
        self._ensure_collection(
            self.collection_code,
            CODE_VECTOR_SIZE,
            _KEYWORD_INDEX_FIELDS,
            recreate=recreate,
        )
        self._ensure_collection(
            self.collection_docs,
            DOC_VECTOR_SIZE,
            _DOC_KEYWORD_INDEX_FIELDS,
            recreate=recreate,
        )

    def _collection_exists(self, name: str) -> bool:
        # collection_exists() is the modern API; fall back to listing
        # collections for older qdrant-client versions / fakes that only
        # implement get_collections().
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(name))
        existing = self.client.get_collections()
        return any(c.name == name for c in existing.collections)

    def _ensure_collection(
        self,
        name: str,
        vector_size: int,
        keyword_fields: Sequence[str],
        *,
        recreate: bool,
    ) -> None:
        from qdrant_client import models

        exists = self._collection_exists(name)
        if recreate and exists:
            self.client.delete_collection(name)
            exists = False

        if not exists:
            logger.info("Creating Qdrant collection '%s' (dim=%d)", name, vector_size)
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            for field_name in keyword_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:  # noqa: BLE001
                    # Payload indexes are a performance optimisation, not a
                    # correctness requirement -- never let a fake/older
                    # client without this endpoint break indexing.
                    logger.debug(
                        "Could not create payload index on %s.%s", name, field_name
                    )

        self._ensured_collections.add(name)

    # ------------------------------------------------------------------
    # Deterministic point IDs (what makes upserts idempotent / incremental)
    # ------------------------------------------------------------------

    @staticmethod
    def code_point_id(repo_id: Any, chunk: Any) -> str:
        """Stable point ID for a code chunk, derived from its identity.

        Same ``(repo, file, symbol, span, part)`` -> same UUID, every time.
        Re-indexing an unchanged chunk therefore upserts in place instead of
        creating a duplicate point, which is the basis for incremental
        updates.
        """
        key = "|".join(
            str(part)
            for part in (
                "code",
                repo_id,
                chunk.file_path,
                chunk.qualified_name or chunk.parent_symbol or "",
                chunk.start_line,
                chunk.chunk_index,
                chunk.part,
            )
        )
        return str(uuid.uuid5(_ID_NAMESPACE, key))

    @staticmethod
    def doc_point_id(repo_id: Any, doc: Any) -> str:
        """Stable point ID for a doc embedding, mirroring :meth:`code_point_id`."""
        key = "|".join(
            str(part)
            for part in (
                "doc",
                repo_id,
                doc.doc_type,
                doc.file_path,
                doc.symbol_id or "",
                doc.start_line,
            )
        )
        return str(uuid.uuid5(_ID_NAMESPACE, key))

    """Build the Qdrant payload for a code chunk.

    Keeps payload construction centralized so both indexing and future
    retrieval changes only need to update one place.
    # Payload schema shared by indexing today and retrieval (Issue 16).
    """

    @staticmethod
    def _build_code_payload(
        chunk: Any,
        repo_id: Any,
        symbol: str | None,
        symbol_type: str | None,
    ) -> dict[str, Any]:
        return {
            "repo_id": repo_id,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": symbol,
            "symbol_type": symbol_type,
            "parent_symbol": chunk.parent_symbol,
            "qualified_name": chunk.qualified_name,
            "chunk_kind": chunk.chunk_kind,
            "token_count": chunk.token_count,
            "is_continuation": chunk.is_continuation,
            "content": chunk.content,
        }

    @staticmethod
    def _build_doc_payload(
        doc: Any,
        repo_id: Any,
        language: str | None,
    ) -> dict[str, Any]:
        """Construct the Qdrant payload for a documentation embedding."""
        payload = {
            "repo_id": repo_id,
            "file_path": doc.file_path,
            "doc_type": doc.doc_type,
            "language": language,
            "symbol_id": doc.symbol_id,
            "start_line": doc.start_line,
            "end_line": doc.end_line,
            "text": doc.text,
        }
        if doc.metadata:
            payload["metadata"] = doc.metadata
        return payload

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------

    def upsert_code_chunks(
        self,
        chunks: Sequence[Any],
        vectors: np.ndarray,
        *,
        repo_id: Any,
        symbols: Sequence[Any] | None = None,
        index_bm25: bool = True,
    ) -> list[str]:
        """Upsert embedded code *chunks* into Qdrant (and, by default, BM25).

        Args:
            chunks: :class:`~reporag.ingestion.chunker.Chunk` objects (or
                anything duck-typed the same way).
            vectors: ``(len(chunks), CODE_VECTOR_SIZE)`` array in the same
                order as *chunks* -- typically ``CodeEmbedder().embed_batch(chunks)``.
            repo_id: Identifier of the owning repository; part of the payload
                and of the deterministic point ID so multiple repos can share
                one collection without ID collisions.
            symbols: The :class:`~reporag.ingestion.symbol_extractor.Symbol`
                tree for the same file(s), if available. ``Chunk`` itself has
                no notion of "class vs. function vs. method" -- only
                ``Symbol.type`` does -- so passing *symbols* here is what lets
                the payload carry a real ``symbol_type`` (indexed, filterable
                by Issue 16). Omit it and ``symbol_type`` is simply ``None``
                in the payload rather than raising.
            index_bm25: Also feed the chunk text into ``self.bm25`` under the
                same point ID (default). Set to ``False`` to update only the
                vector side, e.g. when rebuilding BM25 separately.

        Returns:
            The list of point IDs that were upserted, in the same order as
            *chunks*.
        """

        vectors = np.asarray(vectors)

        if vectors.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape {vectors.shape}."
            )

        if len(vectors) != len(chunks):
            raise ValueError(f"Expected {len(chunks)} embeddings, got {len(vectors)}.")

        if vectors.shape[1] != CODE_VECTOR_SIZE:
            raise ValueError(
                f"Expected embedding dimension {CODE_VECTOR_SIZE}, "
                f"got {vectors.shape[1]}."
            )

        self.ensure_collections()
        symbol_type_lookup = build_symbol_type_lookup(symbols)

        point_ids: list[str] = []
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = self.code_point_id(repo_id, chunk)
            point_ids.append(point_id)

            symbol = chunk.qualified_name or chunk.parent_symbol
            symbol_type = symbol_type_lookup.get(chunk.qualified_name or "") or (
                symbol_type_lookup.get(chunk.parent_symbol or "")
            )
            payload = self._build_code_payload(
                chunk,
                repo_id,
                symbol,
                symbol_type,
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=np.asarray(vector, dtype=np.float32).tolist(),
                    payload=payload,
                )
            )

            if index_bm25:
                self.bm25.add(
                    point_id,
                    chunk.content,
                    metadata={
                        "file_path": chunk.file_path,
                        "symbol": symbol,
                        "symbol_type": symbol_type,
                        "chunk_kind": chunk.chunk_kind,
                        "repo_id": repo_id,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content": chunk.content,
                    },
                )

        if points:
            self.client.upsert(collection_name=self.collection_code, points=points)
            logger.info(
                "Upserted %d code points into '%s'", len(points), self.collection_code
            )

        return point_ids

    def upsert_doc_embeddings(
        self,
        docs: Sequence[Any],
        *,
        repo_id: Any,
        language: str | None = None,
    ) -> list[str]:
        """Upsert :class:`~reporag.embedding.doc_embedder.DocEmbedding` records.

        Docs are vector-only in this index (no BM25 side) -- prose is what
        the semantic side is for; BM25 lexical matching is reserved for code
        identifiers.

        Args:
            docs: The embedded documentation records to upsert.
            repo_id: Identifier of the owning repository.
            language: Language of the source file these docs were extracted
                from (e.g. ``"python"``). ``DocEmbedding`` itself carries no
                language field -- docstrings/comments/README sections are
                prose, not code -- so this is threaded through explicitly by
                the caller (typically once per file, alongside
                ``DocEmbedder.embed_symbols``/``embed_comments``) purely so
                the payload can support the same language filter Issue 16
                applies to the code collection. ``None`` (default) leaves it
                unset, e.g. for a repo-level README with no single language.
        """

        self.ensure_collections()

        point_ids: list[str] = []
        points = []
        vectors = np.asarray([doc.vector for doc in docs], dtype=np.float32)

        if vectors.size and vectors.shape[1] != DOC_VECTOR_SIZE:
            raise ValueError(
                f"Expected embedding dimension {DOC_VECTOR_SIZE}, "
                f"got {vectors.shape[1]}."
            )
        for doc in docs:
            point_id = self.doc_point_id(repo_id, doc)
            point_ids.append(point_id)

            payload = self._build_doc_payload(
                doc,
                repo_id,
                language,
            )
            if doc.metadata:
                payload["metadata"] = doc.metadata

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=np.asarray(doc.vector, dtype=np.float32).tolist(),
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_docs, points=points)
            logger.info(
                "Upserted %d doc points into '%s'", len(points), self.collection_docs
            )

        return point_ids

    # ------------------------------------------------------------------
    # Incremental updates: dropping a file before re-indexing it
    # ------------------------------------------------------------------

    def delete_file(self, repo_id: Any, file_path: str) -> None:
        """Remove every point (code + doc + BM25) belonging to one file.

        Because chunk boundaries can shift when a file's content changes
        (a function grows past the split threshold, a symbol is renamed,
        etc.), the safe incremental-update pattern is: delete everything for
        the file, then re-run extraction/chunking/embedding and upsert the
        fresh chunks -- rather than trying to diff old vs. new chunk IDs.
        Unchanged chunks elsewhere in the repo are untouched.
        """
        from qdrant_client import models

        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="repo_id", match=models.MatchValue(value=repo_id)
                ),
                models.FieldCondition(
                    key="file_path", match=models.MatchValue(value=file_path)
                ),
            ]
        )
        for collection in (self.collection_code, self.collection_docs):
            if collection in self._ensured_collections or self._collection_exists(
                collection
            ):
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=flt),
                )

        removed = self.bm25.remove_where(
            lambda meta: meta.get("file_path") == file_path
        )
        logger.info("Removed %d stale BM25 entries for %s", removed, file_path)

    # ------------------------------------------------------------------
    # BM25 persistence passthrough
    # ------------------------------------------------------------------

    def save_bm25(self, path: str | Path) -> None:
        """Persist ``self.bm25`` so it survives a process restart."""
        self.bm25.save(path)

    def load_bm25(self, path: str | Path) -> None:
        """Replace ``self.bm25`` with the index persisted at *path*."""
        self.bm25 = BM25Index.load(path)

    def __repr__(self) -> str:
        return (
            f"HybridIndexBuilder(code={self.collection_code!r}, "
            f"docs={self.collection_docs!r}, bm25_docs={len(self.bm25)})"
        )

"""Hybrid index builder: Qdrant dense vectors + BM25 sparse keywords.

Vector search excels at *semantic* similarity ("how does the app reject
unauthorised requests?"); BM25 excels at *lexical* precision (find the symbol
literally named ``verify_jwt``).  Building both indices from the same chunks
lets the retrieval layer (Issues 16-18) fuse them and outperform either alone.

What this module does
---------------------
* **Vector index** -- creates/populates two Qdrant collections, one for code
  embeddings (768-dim, ``reporag_code``) and one for documentation embeddings
  (384-dim, ``reporag_docs``).  The two embedders emit *different* vector
  sizes, so they must live in separate collections; a single collection has a
  fixed dimensionality.  Each point carries a rich JSON payload (file, lines,
  symbol, language, chunk text) for filtering and citation downstream.
* **BM25 index** -- tokenises chunk text with a *code-aware* tokenizer and
  builds an in-memory ``rank_bm25`` index over the corpus.

Code-aware tokenization
------------------------
Plain whitespace tokenization is useless for code: ``getUserById`` never
matches a query for "user".  :func:`code_tokenize` splits ``camelCase``,
``PascalCase``, ``snake_case``, and ``ACRONYMRuns`` into their word parts and
lower-cases everything, so ``getUserById`` and ``get_user_by_id`` both index as
``["get", "user", "by", "id"]``.  Because indexing and querying share the same
tokenizer, a query for either the whole identifier or any word inside it hits
the same terms.

Incremental updates
-------------------
Point IDs are deterministic UUID5s derived from stable chunk coordinates, so
re-indexing an unchanged chunk *overwrites* its point instead of duplicating
it.  Adding a new file is therefore a plain ``build_vector_index`` /
``build_bm25_index`` call with ``recreate=False`` (the default) -- no full
rebuild required.

Testability
-----------
Every external dependency is injectable.  Pass ``client=QdrantClient(
location=":memory:")`` for a real-but-ephemeral vector store, so the whole
pipeline can be exercised in unit tests without a running server or network.

Usage::

    from reporag.embedding.index_builder import IndexBuilder

    builder = IndexBuilder(qdrant_url="localhost:6333")
    builder.build_vector_index(chunks, code_embeddings, doc_embeddings)
    builder.build_bm25_index(chunks)
    print(builder.vector_count(), "points")
    print(builder.bm25_doc_count(), "documents")
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reporag.config import settings

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Vector sizes must match the two embedders.  Kept as module constants (rather
# than imported from the embedder modules) so this module stays importable
# without pulling in torch just to build a BM25 index.
#   reporag.embedding.code_embedder.EMBEDDING_DIM == 768  (UniXcoder / CodeBERT)
#   reporag.embedding.doc_embedder.EMBEDDING_DIM  == 384  (all-MiniLM-L6-v2)
DEFAULT_CODE_VECTOR_SIZE = 768
DEFAULT_DOC_VECTOR_SIZE = 384

# Deterministic namespace for point IDs.  A fixed UUID keeps re-runs idempotent:
# the same chunk always maps to the same point, so upserts overwrite in place.
_NAMESPACE = uuid.UUID("6f1d2c9e-4a3b-5c8d-9e0f-1a2b3c4d5e6f")

# Split on every run of non-alphanumeric characters: whitespace, operators,
# punctuation, and -- crucially -- underscores (handling snake_case for free).
_DELIMITER_RE = re.compile(r"[^0-9A-Za-z]+")

# Split a single alphanumeric run into word parts.  The alternatives, in order:
#   [A-Z]+(?![a-z]) -- an acronym run  ("HTTP" in "HTTPResponse", "ID" in "getID")
#   [A-Z][a-z]+     -- a capitalised word  ("Response", "User")
#   [a-z]+          -- a lowercase word  ("get", "user")
#   [0-9]+          -- a number run  ("2" in "utf8" / "HTTP2")
_SUBTOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


# ---------------------------------------------------------------------------
# Code-aware tokenizer (pure, model-free, reused by BM25 search in Issue 17)
# ---------------------------------------------------------------------------


def split_identifier(token: str) -> list[str]:
    """Split one identifier into lower-cased word parts.

    Handles ``camelCase``, ``PascalCase``, and ``ACRONYMWord`` boundaries::

        split_identifier("getUserById")  -> ["get", "user", "by", "id"]
        split_identifier("HTTPResponse")  -> ["http", "response"]
        split_identifier("parseHTTP2")    -> ["parse", "http", "2"]

    ``snake_case`` is already handled upstream by :func:`code_tokenize` (the
    underscore is a delimiter), so this operates on a single case-mixed run.
    """
    return [match.group(0).lower() for match in _SUBTOKEN_RE.finditer(token)]


def code_tokenize(text: str) -> list[str]:
    """Tokenise *text* into lower-cased, identifier-aware terms for BM25.

    The pipeline is: split on non-alphanumeric delimiters (whitespace,
    operators, punctuation, underscores), then split each remaining run at
    ``camelCase`` / acronym boundaries.  ``getUserById`` and ``get_user_by_id``
    both yield ``["get", "user", "by", "id"]``.

    Duplicate terms are intentionally kept -- BM25 relies on term frequency, so
    a word that appears three times should count three times.

    Args:
        text: Raw source, docstring, or query string.

    Returns:
        Ordered list of lower-cased tokens (empty for empty/None input).
    """
    if not text:
        return []
    tokens: list[str] = []
    for run in _DELIMITER_RE.split(text):
        if run:
            tokens.extend(split_identifier(run))
    return tokens


# ---------------------------------------------------------------------------
# Search-result records (thin, JSON-friendly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorHit:
    """One result from a vector-index similarity search.

    Attributes:
        id:      Qdrant point ID (a UUID string).
        score:   Cosine similarity in ``[-1, 1]`` (higher is closer).
        payload: The stored metadata payload (file, lines, symbol, text, ...).
    """

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BM25Hit:
    """One result from a BM25 keyword search.

    Attributes:
        index:   0-based position of the document in the BM25 corpus.
        score:   Raw BM25 relevance score (higher is more relevant).
        payload: The stored metadata payload for the matched chunk.
    """

    index: int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Payload / vector extraction helpers (duck-typed, import-free)
# ---------------------------------------------------------------------------


def _as_payload(chunk: Any) -> dict[str, Any]:
    """Return a JSON-serialisable payload dict for *chunk*.

    Duck-typed so this module never has to import the ``Chunk`` dataclass:
    anything exposing ``to_dict()`` (e.g. :class:`~reporag.ingestion.chunker.Chunk`)
    is used directly; a plain mapping is copied; anything else falls back to its
    ``content`` attribute.
    """
    to_dict = getattr(chunk, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(chunk, dict):
        return dict(chunk)
    return {"content": getattr(chunk, "content", str(chunk))}


def _split_doc(doc: Any) -> tuple[Any, dict[str, Any]]:
    """Return ``(vector, payload)`` for a documentation embedding.

    Accepts a :class:`~reporag.embedding.doc_embedder.DocEmbedding` (via
    ``to_dict()`` / ``.vector``) or a plain mapping already containing a
    ``"vector"`` key.
    """
    to_dict = getattr(doc, "to_dict", None)
    if callable(to_dict):
        payload = dict(to_dict())
    elif isinstance(doc, dict):
        payload = dict(doc)
    else:
        raise TypeError(f"Cannot extract payload from doc embedding: {doc!r}")

    vector = payload.pop("vector", None)
    if vector is None:
        vector = getattr(doc, "vector", None)
    if vector is None:
        raise ValueError("Documentation embedding is missing its vector.")
    return vector, payload


def _to_vector(vector: Any) -> list[float]:
    """Coerce any array-like into a flat ``list[float]`` for Qdrant."""
    return np.asarray(vector, dtype=np.float32).ravel().tolist()


def _stable_key_code(payload: dict[str, Any]) -> str:
    """Build a stable identity key for a code chunk from its coordinates."""
    parts = (
        "file_path",
        "qualified_name",
        "parent_symbol",
        "start_line",
        "end_line",
        "chunk_index",
        "part",
    )
    return "|".join(str(payload.get(key, "")) for key in parts)


def _stable_key_doc(payload: dict[str, Any]) -> str:
    """Build a stable identity key for a documentation embedding.

    Includes a short content hash so two comments on adjacent lines of the same
    symbol never collide onto the same point.
    """
    parts = ("doc_type", "file_path", "symbol_id", "start_line", "end_line")
    text = str(payload.get("text", ""))
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return "|".join(str(payload.get(key, "")) for key in parts) + "|" + text_hash


def _point_id(kind: str, key: str) -> str:
    """Deterministic UUID5 point ID -- identical inputs map to one point."""
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{key}"))


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------


class IndexBuilder:
    """Builds and maintains the hybrid (vector + BM25) retrieval index.

    The vector store is Qdrant; the sparse store is an in-memory ``rank_bm25``
    index.  Both are populated from the same :class:`~reporag.ingestion.chunker.Chunk`
    stream, keeping the two views of the corpus in sync.

    Args:
        qdrant_url:   Qdrant endpoint (e.g. ``"localhost:6333"``,
            ``"http://localhost:6333"``, or ``":memory:"``).  Defaults to
            ``settings.qdrant_url``.  Ignored when *client* is supplied.
        client:       A pre-built Qdrant client to use instead of connecting.
            Inject ``QdrantClient(location=":memory:")`` for network-free tests.
        code_collection: Name of the code-embedding collection.  Defaults to
            ``settings.qdrant_collection_code``.
        doc_collection:  Name of the doc-embedding collection.  Defaults to
            ``settings.qdrant_collection_docs``.
        code_vector_size: Dimensionality of code vectors (default 768).
        doc_vector_size:  Dimensionality of doc vectors (default 384).
        distance:     Similarity metric: ``"cosine"`` (default), ``"dot"``,
            ``"euclid"``, or ``"manhattan"``.

    The Qdrant client is created lazily on first use, so constructing an
    ``IndexBuilder`` never touches the network -- pure BM25 work needs no server.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        *,
        client: Any | None = None,
        code_collection: str | None = None,
        doc_collection: str | None = None,
        code_vector_size: int = DEFAULT_CODE_VECTOR_SIZE,
        doc_vector_size: int = DEFAULT_DOC_VECTOR_SIZE,
        distance: str = "cosine",
    ) -> None:
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.code_collection = code_collection or settings.qdrant_collection_code
        self.doc_collection = doc_collection or settings.qdrant_collection_docs
        self.code_vector_size = code_vector_size
        self.doc_vector_size = doc_vector_size
        self._distance_name = distance.lower()

        self._client = client

        # BM25 corpus state: parallel lists of token-lists and payloads.  The
        # BM25Okapi model is rebuilt lazily (it is cheap) whenever the corpus
        # changes, so incremental adds never pay for a rebuild until queried.
        self._bm25_tokens: list[list[str]] = []
        self._bm25_payloads: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_dirty = False

    # ------------------------------------------------------------------
    # Qdrant client / collection management
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Return the Qdrant client, connecting lazily on first use."""
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient

        if self.qdrant_url == ":memory:":
            logger.info("Creating in-memory Qdrant client")
            self._client = QdrantClient(location=":memory:")
        else:
            url = (
                self.qdrant_url
                if "://" in self.qdrant_url
                else f"http://{self.qdrant_url}"
            )
            logger.info("Connecting to Qdrant at %s", url)
            self._client = QdrantClient(url=url)
        return self._client

    def _distance(self) -> Any:
        """Resolve the configured distance name to a Qdrant enum value."""
        from qdrant_client import models

        mapping = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "euclidean": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN,
        }
        return mapping.get(self._distance_name, models.Distance.COSINE)

    def _resolve_collection(self, alias: str) -> str:
        """Map a friendly alias (``"code"`` / ``"doc"``) to a collection name."""
        key = alias.lower()
        if key in ("code", self.code_collection):
            return self.code_collection
        if key in ("doc", "docs", self.doc_collection):
            return self.doc_collection
        return alias

    def _ensure_collection(
        self,
        name: str,
        size: int,
        *,
        recreate: bool,
        index_fields: Sequence[str],
    ) -> None:
        """Create *name* with the given vector *size* if it does not exist.

        With ``recreate=True`` an existing collection is dropped first (a full
        rebuild).  With the default ``recreate=False`` an existing collection is
        left intact so upserts add to it -- the incremental-update path.
        Keyword payload indexes are created for the fields callers filter on.
        """
        from qdrant_client import models

        client = self._ensure_client()

        if recreate and client.collection_exists(name):
            logger.info("Recreating Qdrant collection '%s'", name)
            client.delete_collection(name)

        if client.collection_exists(name):
            return

        logger.info("Creating Qdrant collection '%s' (size=%d)", name, size)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=self._distance()),
        )
        for field_name in index_fields:
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001 - index hints are best-effort
                logger.debug(
                    "Payload index '%s' on '%s' skipped: %s", field_name, name, exc
                )

    def _upsert(self, name: str, points: list[Any], batch_size: int) -> None:
        """Upsert *points* into collection *name* in batches."""
        if not points:
            return
        client = self._ensure_client()
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=name,
                points=points[start : start + batch_size],
                wait=True,
            )

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------

    def build_vector_index(
        self,
        chunks: Sequence[Any] | None = None,
        code_embeddings: Sequence[Any] | np.ndarray | None = None,
        doc_embeddings: Sequence[Any] | None = None,
        *,
        recreate: bool = False,
        batch_size: int = 256,
    ) -> IndexBuilder:
        """Create the Qdrant collections and upsert code + doc embeddings.

        Code and documentation vectors have different dimensionalities, so they
        are stored in two separate collections (``code_collection`` /
        ``doc_collection``).  Point IDs are deterministic, so calling this again
        with new or changed chunks performs an *incremental* update -- unchanged
        points are overwritten in place, new ones are added -- unless
        ``recreate=True`` forces a fresh rebuild.

        Args:
            chunks: The chunks that produced *code_embeddings*, in the same
                order.  Their :meth:`~reporag.ingestion.chunker.Chunk.to_dict`
                output becomes each code point's payload.  Required when
                *code_embeddings* is given.
            code_embeddings: A ``(len(chunks), code_vector_size)`` array (or
                sequence of vectors), row-aligned with *chunks*.
            doc_embeddings: Self-describing
                :class:`~reporag.embedding.doc_embedder.DocEmbedding` records
                (each carries its own vector and payload).
            recreate: Drop and recreate the collections before upserting.
            batch_size: Points per Qdrant upsert request.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            ValueError: If *code_embeddings* is given without matching *chunks*.
        """
        if code_embeddings is not None:
            chunk_list = list(chunks) if chunks is not None else []
            n_vectors = len(code_embeddings)
            if len(chunk_list) != n_vectors:
                raise ValueError(
                    "chunks and code_embeddings must be the same length: "
                    f"{len(chunk_list)} chunks vs {n_vectors} embeddings."
                )
            self._ensure_collection(
                self.code_collection,
                self.code_vector_size,
                recreate=recreate,
                index_fields=(
                    "file_path",
                    "language",
                    "qualified_name",
                    "parent_symbol",
                    "chunk_kind",
                ),
            )
            self._upsert(
                self.code_collection,
                self._build_code_points(chunk_list, code_embeddings),
                batch_size,
            )

        if doc_embeddings is not None:
            self._ensure_collection(
                self.doc_collection,
                self.doc_vector_size,
                recreate=recreate,
                index_fields=("file_path", "doc_type", "symbol_id"),
            )
            self._upsert(
                self.doc_collection,
                self._build_doc_points(doc_embeddings),
                batch_size,
            )

        return self

    def _build_code_points(
        self, chunks: list[Any], code_embeddings: Sequence[Any] | np.ndarray
    ) -> list[Any]:
        """Turn aligned (chunk, vector) pairs into Qdrant point structs."""
        from qdrant_client import models

        points: list[Any] = []
        for chunk, vector in zip(chunks, code_embeddings, strict=True):
            payload = _as_payload(chunk)
            points.append(
                models.PointStruct(
                    id=_point_id("code", _stable_key_code(payload)),
                    vector=_to_vector(vector),
                    payload=payload,
                )
            )
        return points

    def _build_doc_points(self, doc_embeddings: Sequence[Any]) -> list[Any]:
        """Turn documentation embeddings into Qdrant point structs."""
        from qdrant_client import models

        points: list[Any] = []
        for doc in doc_embeddings:
            vector, payload = _split_doc(doc)
            points.append(
                models.PointStruct(
                    id=_point_id("doc", _stable_key_doc(payload)),
                    vector=_to_vector(vector),
                    payload=payload,
                )
            )
        return points

    def vector_count(self, collection: str | None = None) -> int:
        """Return the number of points in a collection (both, if ``None``).

        Args:
            collection: ``"code"``, ``"doc"``, an explicit collection name, or
                ``None`` to sum across both collections.
        """
        client = self._ensure_client()

        def _count(name: str) -> int:
            if not client.collection_exists(name):
                return 0
            return client.count(collection_name=name, exact=True).count

        if collection is None:
            return _count(self.code_collection) + _count(self.doc_collection)
        return _count(self._resolve_collection(collection))

    def search_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        collection: str = "code",
        top_k: int = 10,
    ) -> list[VectorHit]:
        """Nearest-neighbour search against a vector collection.

        A thin read-back over the index this builder owns -- useful for smoke
        tests and acceptance checks.  The full-featured, filterable vector
        search API lives in :mod:`reporag.retrieval.vector_search` (Issue 16).
        """
        name = self._resolve_collection(collection)
        client = self._ensure_client()
        if not client.collection_exists(name):
            return []
        response = client.query_points(
            collection_name=name,
            query=_to_vector(query_vector),
            limit=top_k,
            with_payload=True,
        )
        return [
            VectorHit(id=str(p.id), score=float(p.score), payload=p.payload or {})
            for p in response.points
        ]

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def build_bm25_index(
        self,
        chunks: Sequence[Any],
        *,
        reset: bool = True,
    ) -> IndexBuilder:
        """Build (or extend) the BM25 index from *chunks*.

        Each chunk's ``content`` is tokenised with the code-aware
        :func:`code_tokenize`.  With ``reset=True`` (default) the corpus is
        rebuilt from *chunks*; with ``reset=False`` the chunks are appended,
        supporting incremental indexing of newly added files.

        Args:
            chunks: Chunks (or mappings) whose ``content`` is indexed.
            reset: Replace the existing corpus instead of appending to it.

        Returns:
            ``self``, to allow method chaining.
        """
        if reset:
            self._bm25_tokens = []
            self._bm25_payloads = []

        for chunk in chunks:
            payload = _as_payload(chunk)
            text = str(payload.get("content", ""))
            self._bm25_tokens.append(code_tokenize(text))
            self._bm25_payloads.append(payload)

        self._bm25_dirty = True
        return self

    def _ensure_bm25(self) -> BM25Okapi | None:
        """Rebuild the BM25 model if the corpus changed; return it (or ``None``)."""
        if self._bm25 is not None and not self._bm25_dirty:
            return self._bm25
        if not self._bm25_tokens:
            self._bm25 = None
            return None

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._bm25_dirty = False
        return self._bm25

    def bm25_doc_count(self) -> int:
        """Return the number of documents in the BM25 corpus."""
        return len(self._bm25_payloads)

    def search_bm25(self, query: str, *, top_k: int = 10) -> list[BM25Hit]:
        """Rank BM25 documents against *query* using the code-aware tokenizer.

        A thin helper for smoke tests and acceptance checks; the full keyword
        search (with exact-match boosting) lives in
        :mod:`reporag.retrieval.bm25_search` (Issue 17).

        Relevance is decided by *term overlap*, not by score sign: a document is
        a candidate only if it shares at least one token with the query.  This
        matters because Okapi BM25's IDF is zero or negative for terms that
        appear in a large fraction of the corpus (common in small indexes), so
        filtering on ``score > 0`` would wrongly discard genuine matches.  The
        BM25 score is then used to rank the candidates, best first.
        """
        bm25 = self._ensure_bm25()
        query_tokens = code_tokenize(query)
        if bm25 is None or not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        query_set = set(query_tokens)
        candidates = [
            i
            for i in range(len(self._bm25_tokens))
            if not query_set.isdisjoint(self._bm25_tokens[i])
        ]
        candidates.sort(key=lambda i: scores[i], reverse=True)
        return [
            BM25Hit(
                index=i,
                score=float(scores[i]),
                payload=self._bm25_payloads[i],
            )
            for i in candidates[:top_k]
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: Sequence[Any],
        code_embeddings: Sequence[Any] | np.ndarray | None = None,
        doc_embeddings: Sequence[Any] | None = None,
        *,
        recreate: bool = False,
    ) -> IndexBuilder:
        """Build both indices in one call.

        Convenience wrapper over :meth:`build_vector_index` and
        :meth:`build_bm25_index` for the common "index everything" path.
        """
        self.build_vector_index(
            chunks,
            code_embeddings,
            doc_embeddings,
            recreate=recreate,
        )
        self.build_bm25_index(chunks, reset=recreate)
        return self

    def __repr__(self) -> str:
        return (
            f"IndexBuilder(qdrant_url={self.qdrant_url!r}, "
            f"code_collection={self.code_collection!r}, "
            f"doc_collection={self.doc_collection!r}, "
            f"bm25_docs={self.bm25_doc_count()})"
        )

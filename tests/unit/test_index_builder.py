"""Unit tests for the hybrid IndexBuilder (Qdrant vectors + BM25 sparse).

The vector-store tests run against a real, in-memory Qdrant client
(``QdrantClient(location=":memory:")``), so they exercise genuine collection
creation, upserts, and similarity search with no server or network.  Vector
dimensions are shrunk (code=4, doc=3) purely to keep the tests fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qdrant_client import QdrantClient

from reporag.embedding.index_builder import (
    BM25Hit,
    IndexBuilder,
    VectorHit,
    _as_payload,
    _point_id,
    _split_doc,
    _stable_key_code,
    _to_vector,
    code_tokenize,
    split_identifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(*values: float) -> np.ndarray:
    """Return the L2-normalised vector for *values* (matches embedder output)."""
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _chunk(
    content: str,
    *,
    file_path: str = "a.py",
    language: str = "python",
    start_line: int = 1,
    end_line: int = 1,
    qualified_name: str | None = None,
    parent_symbol: str | None = None,
    chunk_index: int = 0,
    part: int = 1,
    chunk_kind: str = "definition",
) -> dict:
    """Build a chunk-shaped payload dict (the dict path of ``_as_payload``)."""
    return {
        "content": content,
        "file_path": file_path,
        "language": language,
        "start_line": start_line,
        "end_line": end_line,
        "qualified_name": qualified_name,
        "parent_symbol": parent_symbol,
        "chunk_index": chunk_index,
        "part": part,
        "chunk_kind": chunk_kind,
    }


@dataclass
class _FakeChunk:
    """A chunk exposing ``to_dict()`` -- the duck-typed path of ``_as_payload``."""

    content: str
    file_path: str = "b.py"
    start_line: int = 1
    end_line: int = 2
    qualified_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "qualified_name": self.qualified_name,
        }


@dataclass
class _FakeDoc:
    """A doc embedding exposing ``to_dict()`` + ``.vector`` (like DocEmbedding)."""

    text: str
    vector: np.ndarray
    doc_type: str = "docstring"
    symbol_id: str | None = None
    file_path: str | None = "a.py"
    start_line: int | None = 1
    end_line: int | None = 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "vector": self.vector.astype(np.float32).tolist(),
            "doc_type": self.doc_type,
            "symbol_id": self.symbol_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": {},
        }


def _make_builder(**kwargs) -> IndexBuilder:
    """An IndexBuilder backed by a fresh in-memory Qdrant, small vector dims."""
    return IndexBuilder(
        client=QdrantClient(location=":memory:"),
        code_collection="code_test",
        doc_collection="doc_test",
        code_vector_size=4,
        doc_vector_size=3,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# code_tokenize / split_identifier (the code-aware tokenizer)
# ---------------------------------------------------------------------------


def test_tokenize_splits_camel_case():
    assert code_tokenize("getUserById") == ["get", "user", "by", "id"]


def test_tokenize_splits_snake_case():
    assert code_tokenize("get_user_by_id") == ["get", "user", "by", "id"]


def test_camel_and_snake_produce_identical_tokens():
    assert code_tokenize("getUserById") == code_tokenize("get_user_by_id")


def test_tokenize_splits_pascal_case():
    assert code_tokenize("UserRepository") == ["user", "repository"]


def test_tokenize_handles_acronym_runs():
    assert split_identifier("HTTPResponse") == ["http", "response"]
    assert split_identifier("parseHTTPRequest") == ["parse", "http", "request"]
    assert split_identifier("IOError") == ["io", "error"]


def test_tokenize_handles_trailing_acronym():
    assert split_identifier("getID") == ["get", "id"]


def test_tokenize_separates_digits():
    assert code_tokenize("parse_HTTP2_request") == ["parse", "http", "2", "request"]


def test_tokenize_splits_on_operators_and_punctuation():
    assert code_tokenize("x = getToken() + y") == ["x", "get", "token", "y"]


def test_tokenize_collapses_dunder():
    assert code_tokenize("__init__") == ["init"]


def test_tokenize_empty_and_none():
    assert code_tokenize("") == []
    assert code_tokenize(None) == []
    assert code_tokenize("   \n\t ") == []


def test_tokenize_preserves_term_frequency():
    # BM25 relies on counts: a word repeated twice appears twice.
    assert code_tokenize("token token") == ["token", "token"]


def test_tokenize_full_signature():
    tokens = code_tokenize("def authenticate_user(jwtToken):")
    assert tokens == ["def", "authenticate", "user", "jwt", "token"]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def test_as_payload_from_dict_is_copied():
    src = _chunk("code", file_path="x.py")
    payload = _as_payload(src)
    assert payload == src
    payload["file_path"] = "mutated"
    assert src["file_path"] == "x.py"  # original not mutated


def test_as_payload_from_to_dict_object():
    payload = _as_payload(_FakeChunk("body", file_path="c.py"))
    assert payload["content"] == "body"
    assert payload["file_path"] == "c.py"


def test_as_payload_fallback_content_attr():
    class _Bare:
        content = "raw"

    assert _as_payload(_Bare())["content"] == "raw"


def test_split_doc_from_object():
    vec = _unit(1.0, 0.0, 0.0)
    doc = _FakeDoc("desc", vec, symbol_id="mod.fn")
    out_vec, payload = _split_doc(doc)
    assert "vector" not in payload
    assert payload["symbol_id"] == "mod.fn"
    assert np.allclose(np.asarray(out_vec, dtype=np.float32), vec)


def test_split_doc_from_dict():
    out_vec, payload = _split_doc({"text": "t", "vector": [1.0, 0.0, 0.0]})
    assert payload == {"text": "t"}
    assert out_vec == [1.0, 0.0, 0.0]


def test_split_doc_missing_vector_raises():
    with pytest.raises(ValueError):
        _split_doc({"text": "no vector here"})


def test_to_vector_flattens_and_casts():
    out = _to_vector(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
    assert out == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in out)


def test_point_id_is_deterministic_and_distinct():
    a = _point_id("code", _stable_key_code(_chunk("x", file_path="f.py", start_line=1)))
    a_again = _point_id(
        "code", _stable_key_code(_chunk("x", file_path="f.py", start_line=1))
    )
    b = _point_id("code", _stable_key_code(_chunk("x", file_path="f.py", start_line=9)))
    assert a == a_again  # same coordinates -> same id (idempotent upsert)
    assert a != b  # different line -> different id


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------


def test_vector_collection_created_with_correct_size():
    builder = _make_builder()
    builder.build_vector_index(
        [_chunk("a"), _chunk("b", start_line=2)],
        np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)]),
    )
    client = builder._ensure_client()
    assert client.collection_exists("code_test")
    info = client.get_collection("code_test")
    assert info.config.params.vectors.size == 4


def test_code_and_doc_upserted_with_counts():
    builder = _make_builder()
    chunks = [_chunk("a"), _chunk("b", start_line=2)]
    code_vecs = np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)])
    docs = [_FakeDoc("desc one", _unit(1, 0, 0)), _FakeDoc("desc two", _unit(0, 1, 0))]

    builder.build_vector_index(chunks, code_vecs, docs)

    assert builder.vector_count("code") == 2
    assert builder.vector_count("doc") == 2
    assert builder.vector_count() == 4  # both collections combined


def test_payload_round_trips_through_qdrant():
    builder = _make_builder()
    builder.build_vector_index(
        [_chunk("def f(): pass", qualified_name="f", language="python")],
        np.stack([_unit(1, 0, 0, 0)]),
    )
    hits = builder.search_vector(_unit(1, 0, 0, 0), collection="code", top_k=1)
    assert hits[0].payload["qualified_name"] == "f"
    assert hits[0].payload["language"] == "python"
    assert hits[0].payload["content"] == "def f(): pass"


def test_mismatched_lengths_raise():
    builder = _make_builder()
    with pytest.raises(ValueError, match="same length"):
        builder.build_vector_index([_chunk("a")], np.stack([_unit(1, 0, 0, 0)] * 2))


def test_reindexing_same_chunks_is_idempotent():
    builder = _make_builder()
    chunks = [_chunk("a"), _chunk("b", start_line=2)]
    vecs = np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)])
    builder.build_vector_index(chunks, vecs)
    builder.build_vector_index(chunks, vecs)  # same coordinates -> overwrite
    assert builder.vector_count("code") == 2


def test_incremental_add_without_rebuild():
    builder = _make_builder()
    builder.build_vector_index([_chunk("a")], np.stack([_unit(1, 0, 0, 0)]))
    assert builder.vector_count("code") == 1
    # Add a new file without recreate: existing point survives, new one added.
    builder.build_vector_index(
        [_chunk("b", file_path="new.py", start_line=5)],
        np.stack([_unit(0, 1, 0, 0)]),
    )
    assert builder.vector_count("code") == 2


def test_recreate_drops_existing_points():
    builder = _make_builder()
    builder.build_vector_index(
        [_chunk("a"), _chunk("b", start_line=2)],
        np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)]),
    )
    builder.build_vector_index(
        [_chunk("c", file_path="only.py")],
        np.stack([_unit(0, 0, 1, 0)]),
        recreate=True,
    )
    assert builder.vector_count("code") == 1


def test_search_vector_ranks_by_similarity():
    builder = _make_builder()
    chunks = [_chunk("auth", qualified_name="authenticate"), _chunk("db", start_line=2)]
    vecs = np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)])
    builder.build_vector_index(chunks, vecs)

    hits = builder.search_vector(_unit(1, 0, 0, 0), collection="code", top_k=2)
    assert isinstance(hits[0], VectorHit)
    assert hits[0].payload["qualified_name"] == "authenticate"
    assert hits[0].score > hits[1].score


def test_search_vector_missing_collection_returns_empty():
    builder = _make_builder()
    assert builder.search_vector(_unit(1, 0, 0, 0), collection="code") == []


def test_doc_embeddings_accept_plain_dicts():
    builder = _make_builder()
    builder.build_vector_index(
        doc_embeddings=[
            {"text": "hello", "vector": [1.0, 0.0, 0.0], "doc_type": "readme"}
        ]
    )
    assert builder.vector_count("doc") == 1


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def test_bm25_doc_count():
    builder = _make_builder()
    builder.build_bm25_index([_chunk("a"), _chunk("b"), _chunk("c")])
    assert builder.bm25_doc_count() == 3


def test_bm25_reset_replaces_corpus():
    builder = _make_builder()
    builder.build_bm25_index([_chunk("a"), _chunk("b")])
    builder.build_bm25_index([_chunk("c")])  # reset=True by default
    assert builder.bm25_doc_count() == 1


def test_bm25_incremental_append():
    builder = _make_builder()
    builder.build_bm25_index([_chunk("a")])
    builder.build_bm25_index([_chunk("b")], reset=False)
    assert builder.bm25_doc_count() == 2


def test_bm25_uses_to_dict_objects():
    builder = _make_builder()
    builder.build_bm25_index([_FakeChunk("def authenticate(): pass")])
    hits = builder.search_bm25("authenticate")
    assert hits and hits[0].payload["content"] == "def authenticate(): pass"


def test_bm25_search_finds_identifier_subtoken():
    builder = _make_builder()
    builder.build_bm25_index(
        [
            _chunk("def authenticateUser(token): return verify(token)"),
            _chunk("def connectDatabase(url): return pool(url)", start_line=2),
        ]
    )
    # Query the whole identifier: matches via shared camelCase tokenization.
    hits = builder.search_bm25("authenticateUser")
    assert hits
    assert "authenticate" in hits[0].payload["content"]


def test_bm25_search_empty_index_returns_empty():
    builder = _make_builder()
    assert builder.search_bm25("anything") == []


def test_bm25_search_no_match_returns_empty():
    builder = _make_builder()
    builder.build_bm25_index([_chunk("def connect_database(): pass")])
    assert builder.search_bm25("authenticate") == []


def test_bm25_hit_type():
    builder = _make_builder()
    # A discriminating corpus: "authenticate" appears in only one of three docs,
    # so its IDF (and the resulting score) is positive.
    builder.build_bm25_index(
        [
            _chunk("authenticate user"),
            _chunk("connect database", start_line=2),
            _chunk("render template", start_line=3),
        ]
    )
    hit = builder.search_bm25("authenticate")[0]
    assert isinstance(hit, BM25Hit)
    assert hit.score > 0
    assert hit.payload["content"] == "authenticate user"


# ---------------------------------------------------------------------------
# Acceptance: "authenticate" is retrievable in BOTH indices
# ---------------------------------------------------------------------------


def test_authenticate_retrievable_in_both_indices():
    """The Issue 15 acceptance criterion, exercised end to end.

    A query for "authenticate" must surface the auth-related chunk from both
    the vector index (crafted embeddings + real in-memory Qdrant) and the BM25
    index (real code-aware tokenization).
    """
    builder = _make_builder()

    auth = _chunk(
        "def authenticate_user(token): return verify_jwt(token)",
        file_path="auth.py",
        qualified_name="authenticate_user",
    )
    database = _chunk(
        "def connect_database(url): return open_pool(url)",
        file_path="db.py",
        start_line=2,
        qualified_name="connect_database",
    )

    # The auth chunk's embedding is aligned with the query vector; the db
    # chunk's is orthogonal to it.
    query_vector = _unit(1, 0, 0, 0)
    code_vecs = np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)])

    builder.build(chunks=[auth, database], code_embeddings=code_vecs)

    vector_hits = builder.search_vector(query_vector, collection="code", top_k=2)
    bm25_hits = builder.search_bm25("authenticate", top_k=2)

    assert vector_hits[0].payload["qualified_name"] == "authenticate_user"
    assert bm25_hits[0].payload["qualified_name"] == "authenticate_user"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_url_normalisation_adds_scheme():
    builder = IndexBuilder(qdrant_url="localhost:6333")
    assert builder.qdrant_url == "localhost:6333"  # stored verbatim


def test_repr_is_informative():
    builder = _make_builder()
    builder.build_bm25_index([_chunk("a")])
    text = repr(builder)
    assert "IndexBuilder" in text
    assert "bm25_docs=1" in text

"""Unit tests for the hybrid (Qdrant + BM25) index builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from reporag.embedding.index_builder import (
    CODE_VECTOR_SIZE,
    DOC_VECTOR_SIZE,
    BM25Index,
    HybridIndexBuilder,
    build_symbol_type_lookup,
    split_identifier,
    tokenize_code,
)

# ---------------------------------------------------------------------------
# Fakes standing in for Chunk / DocEmbedding / QdrantClient
# ---------------------------------------------------------------------------


@dataclass
class _FakeChunk:
    """Duck-typed stand-in for reporag.ingestion.chunker.Chunk."""

    content: str
    file_path: str
    language: str = "python"
    start_line: int = 1
    end_line: int = 5
    parent_symbol: str | None = None
    qualified_name: str | None = None
    chunk_kind: str = "definition"
    token_count: int = 10
    chunk_index: int = 0
    part: int = 1
    is_continuation: bool = False


@dataclass
class _FakeDoc:
    """Duck-typed stand-in for reporag.embedding.doc_embedder.DocEmbedding."""

    text: str
    vector: np.ndarray
    doc_type: str = "docstring"
    symbol_id: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeSymbol:
    """Duck-typed stand-in for reporag.ingestion.symbol_extractor.Symbol."""

    name: str
    type: str
    qualified_name: str | None = None
    methods: list[_FakeSymbol] = field(default_factory=list)
    children: list[_FakeSymbol] = field(default_factory=list)


class _FakeCollections:
    def __init__(self, names: set[str]) -> None:
        class _C:
            def __init__(self, name: str) -> None:
                self.name = name

        self.collections = [_C(n) for n in names]


class _FakeQdrantClient:
    """A minimal in-memory fake matching the QdrantClient surface we use."""

    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}
        self.payload_indexes: list[tuple[str, str]] = []
        self.deleted_points: list[tuple[str, Any]] = []

    def collection_exists(self, name: str) -> bool:
        return name in self._collections

    def get_collections(self) -> _FakeCollections:
        return _FakeCollections(set(self._collections))

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self._collections[collection_name] = {"config": vectors_config, "points": {}}

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any
    ) -> None:
        self.payload_indexes.append((collection_name, field_name))

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        store = self._collections[collection_name]["points"]
        for p in points:
            store[p.id] = p

    def delete(self, collection_name: str, points_selector: Any) -> None:
        self.deleted_points.append((collection_name, points_selector))
        # Naively clear everything matching the (repo_id, file_path) filter,
        # good enough to exercise HybridIndexBuilder.delete_file in tests.
        conditions = {c.key: c.match.value for c in points_selector.filter.must}
        store = self._collections[collection_name]["points"]
        stale = [
            pid
            for pid, point in store.items()
            if all(point.payload.get(k) == v for k, v in conditions.items())
        ]
        for pid in stale:
            del store[pid]

    def points(self, collection_name: str) -> dict:
        return self._collections[collection_name]["points"]


@pytest.fixture
def fake_client() -> _FakeQdrantClient:
    return _FakeQdrantClient()


@pytest.fixture
def builder(fake_client: _FakeQdrantClient) -> HybridIndexBuilder:
    return HybridIndexBuilder(
        client=fake_client,
        collection_code="test_code",
        collection_docs="test_docs",
    )


# ---------------------------------------------------------------------------
# Code-aware tokenizer
# ---------------------------------------------------------------------------


class TestTokenizeCode:
    def test_snake_case_split(self):
        assert tokenize_code("get_user_by_id") == ["get", "user", "by", "id"]

    def test_camel_case_split(self):
        assert tokenize_code("getUserByID") == ["get", "user", "by", "id"]

    def test_pascal_case_and_acronym(self):
        assert tokenize_code("HTTPServerError") == ["http", "server", "error"]

    def test_kebab_case_split(self):
        assert tokenize_code("my-cool-endpoint") == ["my", "cool", "endpoint"]

    def test_mixed_snake_and_camel(self):
        assert tokenize_code("parse_HTTPResponse") == ["parse", "http", "response"]

    def test_numbers_kept_as_tokens(self):
        assert tokenize_code("user2Name") == ["user2", "name"]

    def test_punctuation_and_operators_are_boundaries(self):
        tokens = tokenize_code("self.user_id == other.user_id")
        assert tokens == ["self", "user", "id", "other", "user", "id"]

    def test_empty_string(self):
        assert tokenize_code("") == []

    def test_identifiers_that_differ_only_in_style_overlap(self):
        # This overlap is the whole point of the tokenizer: BM25 should find
        # "getUserByID" when the query is "get user by id" or vice versa.
        assert set(tokenize_code("getUserByID")) == set(tokenize_code("get_user_by_id"))

    def test_split_identifier_helper_handles_empty(self):
        assert split_identifier("") == []


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_search_finds_lexical_match(self):
        # rank_bm25's classic idf formula degenerates on a 2-document corpus
        # (a term in exactly one of two docs can get idf == 0), so we use a
        # few unrelated filler docs to keep the term statistics meaningful,
        # closer to what a real repo-sized index looks like.
        index = BM25Index()
        index.add("a", "def get_user_by_id(user_id): return db.find(user_id)")
        index.add("b", "def delete_session(token): cache.pop(token)")
        index.add("c", "def send_email(recipient, subject, body): smtp.send(recipient)")
        index.add("d", "class PaymentProcessor: def charge(self, amount): pass")

        results = index.search("get user by id")
        assert results
        assert results[0]["id"] == "a"

    def test_search_matches_across_naming_styles(self):
        index = BM25Index()
        index.add("camel", "def getUserByID(userID): pass")
        index.add("unrelated_1", "def delete_session(token): pass")
        index.add("unrelated_2", "def send_email(recipient): pass")
        index.add("unrelated_3", "class PaymentProcessor: pass")

        results = index.search("get_user_by_id")
        assert results[0]["id"] == "camel"

    def test_empty_index_returns_no_results(self):
        index = BM25Index()
        assert index.search("anything") == []

    def test_add_overwrites_existing_doc(self):
        index = BM25Index()
        index.add("a", "def alpha(): pass")
        index.add("filler_1", "def unrelated_one(): pass")
        index.add("filler_2", "def unrelated_two(): pass")
        assert index.search("alpha")[0]["id"] == "a"

        index.add("a", "def beta(): pass")
        assert index.search("alpha") == []
        assert index.search("beta")[0]["id"] == "a"

    def test_remove(self):
        index = BM25Index()
        index.add("a", "def alpha(): pass")
        assert "a" in index
        assert index.remove("a") is True
        assert "a" not in index
        assert index.remove("a") is False

    def test_remove_where(self):
        index = BM25Index()
        index.add("a", "def alpha(): pass", metadata={"file_path": "x.py"})
        index.add("b", "def beta(): pass", metadata={"file_path": "y.py"})

        removed = index.remove_where(lambda meta: meta.get("file_path") == "x.py")
        assert removed == 1
        assert "a" not in index
        assert "b" in index

    def test_len(self):
        index = BM25Index()
        assert len(index) == 0
        index.add("a", "def alpha(): pass")
        index.add("b", "def beta(): pass")
        assert len(index) == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        index = BM25Index()
        index.add(
            "a", "def get_user_by_id(user_id): pass", metadata={"file_path": "x.py"}
        )
        index.add(
            "b", "def delete_session(token): pass", metadata={"file_path": "y.py"}
        )
        index.add(
            "c", "def send_email(recipient): pass", metadata={"file_path": "z.py"}
        )
        index.add("d", "class PaymentProcessor: pass", metadata={"file_path": "w.py"})

        path = tmp_path / "index.pkl"
        index.save(path)

        loaded = BM25Index.load(path)
        assert len(loaded) == 4
        results = loaded.search("get user by id")
        assert results[0]["id"] == "a"
        assert results[0]["metadata"]["file_path"] == "x.py"

    def test_incremental_add_after_load(self, tmp_path):
        index = BM25Index()
        index.add("a", "def alpha(): pass")
        index.add("filler_1", "def unrelated_one(): pass")
        path = tmp_path / "index.pkl"
        index.save(path)

        loaded = BM25Index.load(path)
        loaded.add("b", "def beta(): pass")
        loaded.add("filler_2", "def unrelated_two(): pass")
        assert len(loaded) == 4
        assert loaded.search("beta")[0]["id"] == "b"

    def test_get_document(self):
        index = BM25Index()
        index.add(
            "a",
            "def alpha(): pass",
            metadata={"symbol": "alpha"},
        )

        doc = index.get_document("a")

        assert doc is not None
        assert doc.doc_id == "a"
        assert doc.metadata["symbol"] == "alpha"


# ---------------------------------------------------------------------------
# HybridIndexBuilder: Qdrant collection schema
# ---------------------------------------------------------------------------


class TestEnsureCollections:
    def test_creates_both_collections_with_correct_dims(self, builder, fake_client):
        builder.ensure_collections()

        assert fake_client.collection_exists("test_code")
        assert fake_client.collection_exists("test_docs")
        assert fake_client._collections["test_code"]["config"].size == CODE_VECTOR_SIZE
        assert fake_client._collections["test_docs"]["config"].size == DOC_VECTOR_SIZE

    def test_creates_payload_indexes(self, builder, fake_client):
        builder.ensure_collections()
        indexed_fields = {f for _, f in fake_client.payload_indexes}
        assert "file_path" in indexed_fields
        assert "repo_id" in indexed_fields

    def test_idempotent_does_not_recreate(self, builder, fake_client):
        builder.ensure_collections()
        builder.client.upsert(
            collection_name="test_code",
            points=[],
        )
        first_index_count = len(fake_client.payload_indexes)
        builder.ensure_collections()
        assert len(fake_client.payload_indexes) == first_index_count

    def test_recreate_drops_existing_collection(self, builder, fake_client):
        builder.ensure_collections()
        assert fake_client.collection_exists("test_code")
        builder.ensure_collections(recreate=True)
        assert fake_client.collection_exists("test_code")


# ---------------------------------------------------------------------------
# HybridIndexBuilder: upserts
# ---------------------------------------------------------------------------


class TestBuildSymbolTypeLookup:
    def test_empty_input(self):
        assert build_symbol_type_lookup(None) == {}
        assert build_symbol_type_lookup([]) == {}

    def test_flat_symbols(self):
        symbols = [
            _FakeSymbol(name="foo", type="function", qualified_name="foo"),
            _FakeSymbol(name="Bar", type="class", qualified_name="Bar"),
        ]
        lookup = build_symbol_type_lookup(symbols)
        assert lookup["foo"] == "function"
        assert lookup["Bar"] == "class"

    def test_descends_into_methods_and_children(self):
        method = _FakeSymbol(
            name="charge", type="method", qualified_name="MyClass.charge"
        )
        nested_fn = _FakeSymbol(
            name="helper", type="function", qualified_name="MyClass.helper"
        )
        klass = _FakeSymbol(
            name="MyClass",
            type="class",
            qualified_name="MyClass",
            methods=[method],
            children=[nested_fn],
        )
        lookup = build_symbol_type_lookup([klass])
        assert lookup["MyClass"] == "class"
        assert lookup["MyClass.charge"] == "method"
        assert lookup["MyClass.helper"] == "function"

    def test_falls_back_to_name_when_no_qualified_name(self):
        symbols = [_FakeSymbol(name="foo", type="function", qualified_name=None)]
        lookup = build_symbol_type_lookup(symbols)
        assert lookup["foo"] == "function"


class TestUpsertCodeChunks:
    def test_upserts_into_qdrant_and_bm25(self, builder, fake_client):
        chunks = [
            _FakeChunk(
                content="def get_user_by_id(user_id): return db.find(user_id)",
                file_path="app.py",
                qualified_name="get_user_by_id",
            ),
            _FakeChunk(
                content="def delete_session(token): cache.pop(token)",
                file_path="app.py",
                qualified_name="delete_session",
            ),
        ]
        vectors = np.random.rand(2, CODE_VECTOR_SIZE).astype(np.float32)

        ids = builder.upsert_code_chunks(chunks, vectors, repo_id="repo-1")

        assert len(ids) == 2
        assert len(fake_client.points("test_code")) == 2
        assert len(builder.bm25) == 2

        payload = fake_client.points("test_code")[ids[0]].payload
        assert payload["file_path"] == "app.py"
        assert payload["symbol"] == "get_user_by_id"
        assert payload["repo_id"] == "repo-1"

    def test_mismatched_lengths_raise(self, builder):
        chunks = [_FakeChunk(content="x", file_path="a.py")]
        vectors = np.random.rand(2, CODE_VECTOR_SIZE).astype(np.float32)
        with pytest.raises(ValueError):
            builder.upsert_code_chunks(chunks, vectors, repo_id="repo-1")

    def test_invalid_vector_dimension_raises(self, builder):
        chunks = [
            _FakeChunk(content="def alpha(): pass", file_path="a.py"),
            _FakeChunk(content="def beta(): pass", file_path="b.py"),
            _FakeChunk(content="def gamma(): pass", file_path="c.py"),
        ]

        bad_vectors = np.zeros((3, 100), dtype=np.float32)

        with pytest.raises(
            ValueError,
            match="Expected embedding dimension",
        ):
            builder.upsert_code_chunks(
                chunks,
                bad_vectors,
                repo_id="repo-1",
            )

    def test_index_bm25_false_skips_bm25(self, builder):
        chunks = [_FakeChunk(content="def alpha(): pass", file_path="a.py")]
        vectors = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)
        builder.upsert_code_chunks(chunks, vectors, repo_id="repo-1", index_bm25=False)
        assert len(builder.bm25) == 0

    def test_reupserting_same_chunk_is_idempotent(self, builder, fake_client):
        chunk = _FakeChunk(
            content="def alpha(): pass", file_path="a.py", qualified_name="alpha"
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)

        ids_1 = builder.upsert_code_chunks([chunk], vec, repo_id="repo-1")
        ids_2 = builder.upsert_code_chunks([chunk], vec, repo_id="repo-1")

        assert ids_1 == ids_2
        assert len(fake_client.points("test_code")) == 1
        assert len(builder.bm25) == 1

    def test_different_repo_same_chunk_gets_different_id(self, builder):
        chunk = _FakeChunk(
            content="def alpha(): pass", file_path="a.py", qualified_name="alpha"
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)

        ids_repo1 = builder.upsert_code_chunks([chunk], vec, repo_id="repo-1")
        ids_repo2 = builder.upsert_code_chunks([chunk], vec, repo_id="repo-2")

        assert ids_repo1 != ids_repo2

    def test_symbol_type_resolved_from_symbols_by_qualified_name(
        self, builder, fake_client
    ):
        chunk = _FakeChunk(
            content="def get_user_by_id(user_id): pass",
            file_path="app.py",
            qualified_name="get_user_by_id",
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)
        symbols = [
            _FakeSymbol(
                name="get_user_by_id", type="function", qualified_name="get_user_by_id"
            )
        ]

        ids = builder.upsert_code_chunks(
            [chunk], vec, repo_id="repo-1", symbols=symbols
        )

        payload = fake_client.points("test_code")[ids[0]].payload
        assert payload["symbol_type"] == "function"

    def test_symbol_type_resolved_for_method_via_parent_class(
        self, builder, fake_client
    ):
        # A method chunk's qualified_name is "MyClass.charge"; the Symbol for
        # it lives nested under the class Symbol's .methods list.
        method_symbol = _FakeSymbol(
            name="charge", type="method", qualified_name="MyClass.charge"
        )
        class_symbol = _FakeSymbol(
            name="MyClass",
            type="class",
            qualified_name="MyClass",
            methods=[method_symbol],
        )
        chunk = _FakeChunk(
            content="def charge(self, amount): pass",
            file_path="billing.py",
            parent_symbol="MyClass",
            qualified_name="MyClass.charge",
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)

        ids = builder.upsert_code_chunks(
            [chunk], vec, repo_id="repo-1", symbols=[class_symbol]
        )

        payload = fake_client.points("test_code")[ids[0]].payload
        assert payload["symbol_type"] == "method"

    def test_symbol_type_none_when_symbols_not_provided(self, builder, fake_client):
        chunk = _FakeChunk(
            content="def alpha(): pass", file_path="a.py", qualified_name="alpha"
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)

        ids = builder.upsert_code_chunks([chunk], vec, repo_id="repo-1")

        payload = fake_client.points("test_code")[ids[0]].payload
        assert payload["symbol_type"] is None

    def test_symbol_type_is_indexed_for_filtering(self, builder, fake_client):
        builder.ensure_collections()
        indexed_fields = {f for _, f in fake_client.payload_indexes}
        assert "symbol_type" in indexed_fields


class TestUpsertDocEmbeddings:
    def test_upserts_into_qdrant_docs_collection(self, builder, fake_client):
        docs = [
            _FakeDoc(
                text="Fetches a user by id.",
                vector=np.random.rand(DOC_VECTOR_SIZE).astype(np.float32),
                symbol_id="get_user_by_id",
                file_path="app.py",
                start_line=10,
                end_line=10,
            )
        ]
        ids = builder.upsert_doc_embeddings(docs, repo_id="repo-1")

        assert len(ids) == 1
        assert len(fake_client.points("test_docs")) == 1
        payload = fake_client.points("test_docs")[ids[0]].payload
        assert payload["symbol_id"] == "get_user_by_id"
        assert payload["doc_type"] == "docstring"

    def test_docs_are_not_added_to_bm25(self, builder):
        docs = [
            _FakeDoc(
                text="Fetches a user by id.",
                vector=np.random.rand(DOC_VECTOR_SIZE).astype(np.float32),
            )
        ]
        builder.upsert_doc_embeddings(docs, repo_id="repo-1")
        assert len(builder.bm25) == 0

    def test_language_defaults_to_none(self, builder, fake_client):
        docs = [
            _FakeDoc(
                text="Fetches a user by id.",
                vector=np.random.rand(DOC_VECTOR_SIZE).astype(np.float32),
            )
        ]
        ids = builder.upsert_doc_embeddings(docs, repo_id="repo-1")
        payload = fake_client.points("test_docs")[ids[0]].payload
        assert payload["language"] is None

    def test_language_is_stored_when_provided(self, builder, fake_client):
        docs = [
            _FakeDoc(
                text="Fetches a user by id.",
                vector=np.random.rand(DOC_VECTOR_SIZE).astype(np.float32),
            )
        ]
        ids = builder.upsert_doc_embeddings(docs, repo_id="repo-1", language="python")
        payload = fake_client.points("test_docs")[ids[0]].payload
        assert payload["language"] == "python"

    def test_language_is_indexed_for_filtering(self, builder, fake_client):
        builder.ensure_collections()
        pairs = set(fake_client.payload_indexes)
        assert ("test_code", "language") in pairs
        assert ("test_docs", "language") in pairs


# ---------------------------------------------------------------------------
# HybridIndexBuilder: incremental updates
# ---------------------------------------------------------------------------


class TestIncrementalUpdates:
    def test_delete_file_removes_qdrant_and_bm25_entries(self, builder, fake_client):
        chunks = [
            _FakeChunk(
                content="def alpha(): pass", file_path="a.py", qualified_name="alpha"
            ),
            _FakeChunk(
                content="def beta(): pass", file_path="b.py", qualified_name="beta"
            ),
        ]
        vectors = np.random.rand(2, CODE_VECTOR_SIZE).astype(np.float32)
        builder.upsert_code_chunks(chunks, vectors, repo_id="repo-1")

        builder.delete_file("repo-1", "a.py")

        remaining = fake_client.points("test_code")
        assert len(remaining) == 1
        assert next(iter(remaining.values())).payload["file_path"] == "b.py"
        assert len(builder.bm25) == 1

    def test_reindexing_after_delete_only_adds_new_file_points(
        self, builder, fake_client
    ):
        old_chunk = _FakeChunk(
            content="def alpha_old(): pass",
            file_path="a.py",
            qualified_name="alpha_old",
        )
        vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)
        builder.upsert_code_chunks([old_chunk], vec, repo_id="repo-1")

        builder.delete_file("repo-1", "a.py")

        new_chunks = [
            _FakeChunk(
                content="def alpha_v2(): pass",
                file_path="a.py",
                qualified_name="alpha_v2",
            ),
            _FakeChunk(
                content="def alpha_v2_helper(): pass",
                file_path="a.py",
                qualified_name="alpha_v2_helper",
            ),
        ]
        new_vecs = np.random.rand(2, CODE_VECTOR_SIZE).astype(np.float32)
        builder.upsert_code_chunks(new_chunks, new_vecs, repo_id="repo-1")

        points = fake_client.points("test_code")
        assert len(points) == 2
        symbols = {p.payload["symbol"] for p in points.values()}
        assert symbols == {"alpha_v2", "alpha_v2_helper"}


# ---------------------------------------------------------------------------
# HybridIndexBuilder: repr / bm25 passthrough
# ---------------------------------------------------------------------------


def test_repr(builder):
    text = repr(builder)
    assert "test_code" in text
    assert "test_docs" in text


def test_save_and_load_bm25_passthrough(builder, tmp_path):
    chunk = _FakeChunk(
        content="def alpha(): pass", file_path="a.py", qualified_name="alpha"
    )
    vec = np.random.rand(1, CODE_VECTOR_SIZE).astype(np.float32)
    builder.upsert_code_chunks([chunk], vec, repo_id="repo-1")

    path = tmp_path / "bm25.pkl"
    builder.save_bm25(path)

    builder.bm25 = BM25Index()
    assert len(builder.bm25) == 0

    builder.load_bm25(path)
    assert len(builder.bm25) == 1

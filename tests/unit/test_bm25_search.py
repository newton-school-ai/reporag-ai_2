"""Unit tests for the BM25 sparse keyword search retrieval engine (Issue 17).

Payload schema mirrors the Issue 15 / HybridIndexBuilder canonical layout
that ``HybridIndexBuilder.upsert_code_chunks`` feeds into ``BM25Index.add``:
  metadata: file_path, symbol, symbol_type, chunk_kind, repo_id,
            start_line, end_line, content
  (note: no ``language`` key -- see the "Known limitation" section of
  bm25_search.py's module docstring, and ``TestLanguageFilter`` below).

Several fixtures deliberately include a handful of unrelated "filler"
documents alongside the documents under test. This mirrors
``tests/unit/test_index_builder.py``'s ``TestBM25Index`` fixtures: BM25's
classic idf formula degenerates on a two-or-three-document corpus (a term
appearing in most of the docs can get idf <= 0, silently zeroing scores),
so filler keeps the term statistics realistic -- closer to what a
repo-sized index actually looks like -- and avoids flaky/zero-score tests.
"""

from __future__ import annotations

import pytest

from reporag.embedding.index_builder import BM25Index
from reporag.retrieval.bm25_search import BM25Search
from reporag.retrieval.vector_search import RetrievalResult as VectorRetrievalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FILLER_DOCS = [
    (
        "filler_delete_session",
        "def delete_session(token): cache.pop(token)",
        {
            "file_path": "src/session.py",
            "symbol": "delete_session",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "start_line": 1,
            "end_line": 2,
            "content": "def delete_session(token): cache.pop(token)",
        },
    ),
    (
        "filler_send_email",
        "def send_email(recipient, subject, body): smtp.send(recipient)",
        {
            "file_path": "src/mailer.py",
            "symbol": "send_email",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "start_line": 1,
            "end_line": 2,
            "content": "def send_email(recipient, subject, body): smtp.send(recipient)",
        },
    ),
    (
        "filler_payment_processor",
        "class PaymentProcessor: def charge(self, amount): pass",
        {
            "file_path": "src/billing.py",
            "symbol": "PaymentProcessor",
            "symbol_type": "class",
            "chunk_kind": "definition",
            "start_line": 1,
            "end_line": 2,
            "content": "class PaymentProcessor: def charge(self, amount): pass",
        },
    ),
]


def _add_filler(index: BM25Index) -> None:
    for doc_id, text, metadata in _FILLER_DOCS:
        index.add(doc_id, text, metadata=metadata)


@pytest.fixture
def basic_index() -> BM25Index:
    """A BM25Index with one clear definition chunk plus filler for realistic idf."""
    index = BM25Index()
    index.add(
        "def_get_user_by_id",
        "def get_user_by_id(user_id): return db.find(user_id)",
        metadata={
            "file_path": "src/app.py",
            "symbol": "get_user_by_id",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "org/repo",
            "start_line": 10,
            "end_line": 11,
            "content": "def get_user_by_id(user_id): return db.find(user_id)",
        },
    )
    _add_filler(index)
    return index


@pytest.fixture
def boost_scenario_index() -> BM25Index:
    """A corpus where a raw BM25 ranking would put a *caller* above the *definer*.

    ``caller_chunk`` repeats the bare identifier several times with no other
    tokens, so its raw term-frequency score outranks the definition, which
    dilutes the same tokens across a full function body plus docstring.
    This is exactly the scenario the exact-identifier boost exists to fix:
    a literal identifier query should surface the *defining* chunk as
    top-1, not whichever chunk happens to repeat the name the most.
    """
    index = BM25Index()
    index.add(
        "def_chunk",
        (
            'def get_user_by_id(user_id): """Fetch a user record from the '
            'database by primary key.""" return db.query(User).filter('
            "User.id == user_id).first()"
        ),
        metadata={
            "file_path": "src/app.py",
            "symbol": "get_user_by_id",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "start_line": 10,
            "end_line": 12,
            "content": "def get_user_by_id(user_id): return db.find(user_id)",
        },
    )
    index.add(
        "caller_chunk",
        "get_user_by_id get_user_by_id get_user_by_id get_user_by_id",
        metadata={
            "file_path": "src/main.py",
            "symbol": "main",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "start_line": 1,
            "end_line": 2,
            "content": ("get_user_by_id get_user_by_id get_user_by_id get_user_by_id"),
        },
    )
    _add_filler(index)
    return index


@pytest.fixture
def multi_language_index() -> BM25Index:
    """Corpus where some docs *do* carry a language key in metadata.

    Real ``HybridIndexBuilder.upsert_code_chunks`` output never has this key
    (see the module docstring's "Known limitation"), but callers who add
    their own metadata may include one -- this fixture proves filtering
    works correctly in that case.
    """
    index = BM25Index()
    index.add(
        "py_render",
        "def render_widget(name): return template.render(name)",
        metadata={
            "file_path": "src/widgets.py",
            "symbol": "render_widget",
            "symbol_type": "function",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "content": "def render_widget(name): return template.render(name)",
        },
    )
    index.add(
        "js_render",
        "function renderWidget(name) { return template.render(name); }",
        metadata={
            "file_path": "static/widgets.js",
            "symbol": "renderWidget",
            "symbol_type": "function",
            "language": "javascript",
            "start_line": 1,
            "end_line": 2,
            "content": "function renderWidget(name) { return template.render(name); }",
        },
    )
    _add_filler(index)
    return index


# ---------------------------------------------------------------------------
# Schema parity with vector search
# ---------------------------------------------------------------------------


class TestSchemaParity:
    """BM25Search must return the exact same RetrievalResult type as VectorSearch."""

    def test_returns_vector_search_retrieval_result_class(
        self, basic_index: BM25Index
    ) -> None:
        results = BM25Search(basic_index).search("get_user_by_id")
        assert results
        assert isinstance(results[0], VectorRetrievalResult)

    def test_result_has_all_expected_fields(self, basic_index: BM25Index) -> None:
        r = BM25Search(basic_index).search("get_user_by_id")[0]
        assert r.file_path == "src/app.py"
        assert r.start_line == 10
        assert r.end_line == 11
        assert r.symbol_name == "get_user_by_id"
        assert r.chunk_text == "def get_user_by_id(user_id): return db.find(user_id)"
        assert isinstance(r.metadata, dict)
        assert r.score > 0


# ---------------------------------------------------------------------------
# Payload field alignment
# ---------------------------------------------------------------------------


class TestPayloadFieldAlignment:
    """Verify RetrievalResult fields are resolved from the correct metadata keys."""

    def test_symbol_name_from_symbol_key(self, basic_index: BM25Index) -> None:
        r = BM25Search(basic_index).search("get_user_by_id")[0]
        assert r.symbol_name == "get_user_by_id"

    def test_chunk_text_from_content_key(self, basic_index: BM25Index) -> None:
        r = BM25Search(basic_index).search("get_user_by_id")[0]
        assert r.chunk_text == "def get_user_by_id(user_id): return db.find(user_id)"

    def test_metadata_contains_full_payload(self, basic_index: BM25Index) -> None:
        r = BM25Search(basic_index).search("get_user_by_id")[0]
        for key in (
            "file_path",
            "symbol",
            "symbol_type",
            "chunk_kind",
            "repo_id",
            "start_line",
            "end_line",
            "content",
        ):
            assert key in r.metadata, f"Missing metadata key: {key!r}"

    def test_missing_optional_metadata_defaults_safely(self) -> None:
        """A doc with sparse metadata should not raise -- fields fall back safely."""
        index = BM25Index()
        index.add("sparse", "def alpha(): pass", metadata={})
        _add_filler(index)
        r = BM25Search(index).search("alpha")[0]
        assert r.file_path == ""
        assert r.start_line is None
        assert r.end_line is None
        assert r.symbol_name is None
        assert r.chunk_text == ""

    def test_no_metadata_at_all_does_not_raise(self) -> None:
        """A doc added with metadata=None entirely should still convert safely."""
        index = BM25Index()
        index.add("no_meta", "def alpha(): pass")  # metadata defaults to {}
        _add_filler(index)
        results = BM25Search(index).search("alpha")
        assert results
        assert results[0].file_path == ""


# ---------------------------------------------------------------------------
# Exact-match boosting
# ---------------------------------------------------------------------------


class TestExactMatchBoost:
    """Verify the exact-identifier boost surfaces the defining chunk as top-1."""

    def test_exact_identifier_query_returns_definer_as_top1(
        self, boost_scenario_index: BM25Index
    ) -> None:
        results = BM25Search(boost_scenario_index).search("get_user_by_id")
        assert results[0].file_path == "src/app.py"
        assert results[0].symbol_name == "get_user_by_id"

    def test_without_boost_caller_outranks_definer(
        self, boost_scenario_index: BM25Index
    ) -> None:
        """Pin the *raw* ranking so the boost test above is meaningfully testing something."""
        results = BM25Search(boost_scenario_index).search(
            "get_user_by_id", boost_exact_match=False
        )
        assert results[0].file_path == "src/main.py"

    def test_boost_actually_changes_the_score(
        self, boost_scenario_index: BM25Index
    ) -> None:
        boosted = BM25Search(boost_scenario_index, exact_match_boost=2.0).search(
            "get_user_by_id"
        )
        by_symbol = {r.symbol_name: r.score for r in boosted}
        unboosted = BM25Search(boost_scenario_index).search(
            "get_user_by_id", boost_exact_match=False
        )
        raw_by_symbol = {r.symbol_name: r.score for r in unboosted}
        assert by_symbol["get_user_by_id"] == pytest.approx(
            raw_by_symbol["get_user_by_id"] * 2.0
        )
        # Non-exact-match candidate's score is untouched.
        assert by_symbol["main"] == pytest.approx(raw_by_symbol["main"])

    def test_boost_matches_across_naming_styles(self) -> None:
        """A camelCase query must still boost a snake_case definer (and vice versa).

        Both tokenize (via tokenize_code) to the same {get, user, by, id}
        set, so the boost's tokenized-equality check must fire regardless
        of the literal spelling used in the query vs. the indexed symbol.
        """
        index = BM25Index()
        index.add(
            "def_chunk",
            "def getUserByID(userID): return db.find(userID)",
            metadata={
                "file_path": "src/app.py",
                "symbol": "getUserByID",
                "symbol_type": "function",
                "content": "def getUserByID(userID): return db.find(userID)",
            },
        )
        index.add(
            "caller_chunk",
            "getUserByID getUserByID getUserByID getUserByID",
            metadata={
                "file_path": "src/main.py",
                "symbol": "main",
                "symbol_type": "function",
                "content": "getUserByID getUserByID getUserByID getUserByID",
            },
        )
        _add_filler(index)
        # Query uses snake_case; indexed symbol is camelCase.
        results = BM25Search(index).search("get_user_by_id")
        assert results[0].symbol_name == "getUserByID"

    def test_boost_does_not_fire_on_partial_match(self) -> None:
        """A query that only partially overlaps a symbol must not be boosted."""
        index = BM25Index()
        index.add(
            "def_chunk",
            "def get_user_by_id_and_tenant(user_id, tenant_id): pass",
            metadata={
                "file_path": "src/app.py",
                "symbol": "get_user_by_id_and_tenant",
                "symbol_type": "function",
                "content": "def get_user_by_id_and_tenant(user_id, tenant_id): pass",
            },
        )
        _add_filler(index)
        boosted = BM25Search(index).search("get_user_by_id")
        unboosted = BM25Search(index).search("get_user_by_id", boost_exact_match=False)
        # Partial-match symbol is NOT exactly equal to the query tokens, so
        # its score must be identical with or without boosting.
        assert boosted[0].score == pytest.approx(unboosted[0].score)

    def test_boost_disabled_via_flag(self, boost_scenario_index: BM25Index) -> None:
        results = BM25Search(boost_scenario_index).search(
            "get_user_by_id", boost_exact_match=False
        )
        # No reordering happens -- caller (higher raw score) stays on top.
        assert results[0].file_path == "src/main.py"

    def test_exact_match_boost_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="exact_match_boost"):
            BM25Search(exact_match_boost=0.5)

    def test_exact_match_boost_of_exactly_one_is_a_noop(
        self, boost_scenario_index: BM25Index
    ) -> None:
        """boost == 1.0 is valid (allowed) but should not change the ranking."""
        results = BM25Search(boost_scenario_index, exact_match_boost=1.0).search(
            "get_user_by_id"
        )
        assert results[0].file_path == "src/main.py"

    def test_empty_query_never_boosts_or_raises(self, basic_index: BM25Index) -> None:
        assert BM25Search(basic_index).search("   ") == []


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestSymbolTypeFilter:
    def test_function_filter_excludes_class(
        self, boost_scenario_index: BM25Index
    ) -> None:
        results = BM25Search(boost_scenario_index).search(
            "get_user_by_id", symbol_type="function"
        )
        assert results
        assert all(r.metadata.get("symbol_type") == "function" for r in results)

    def test_class_filter_returns_only_classes(self, basic_index: BM25Index) -> None:
        results = BM25Search(basic_index).search(
            "PaymentProcessor", symbol_type="class"
        )
        assert results
        assert all(r.symbol_name == "PaymentProcessor" for r in results)

    def test_symbol_type_filter_can_produce_empty_result(
        self, basic_index: BM25Index
    ) -> None:
        results = BM25Search(basic_index).search("get_user_by_id", symbol_type="class")
        assert results == []


class TestFilePathFilter:
    def test_exact_file_path_match(self, basic_index: BM25Index) -> None:
        results = BM25Search(basic_index).search(
            "get_user_by_id", file_path="src/app.py"
        )
        assert results
        assert all(r.file_path == "src/app.py" for r in results)

    def test_exact_file_path_no_match_returns_empty(
        self, basic_index: BM25Index
    ) -> None:
        results = BM25Search(basic_index).search(
            "get_user_by_id", file_path="src/other.py"
        )
        assert results == []

    def test_glob_file_path_match(self, basic_index: BM25Index) -> None:
        results = BM25Search(basic_index).search("get_user_by_id", file_path="src/*.py")
        assert results
        assert all(r.file_path.startswith("src/") for r in results)

    def test_glob_file_path_no_match(self, basic_index: BM25Index) -> None:
        results = BM25Search(basic_index).search(
            "get_user_by_id", file_path="static/*.js"
        )
        assert results == []


class TestLanguageFilter:
    """See bm25_search.py's module docstring: "Known limitation -- language filtering"."""

    def test_language_filter_works_when_metadata_has_language(
        self, multi_language_index: BM25Index
    ) -> None:
        results = BM25Search(multi_language_index).search(
            "render_widget", language="python"
        )
        assert results
        assert all(r.file_path.endswith(".py") for r in results)

    def test_language_filter_excludes_other_language(
        self, multi_language_index: BM25Index
    ) -> None:
        results = BM25Search(multi_language_index).search(
            "render_widget", language="javascript"
        )
        assert all(r.file_path != "src/widgets.py" for r in results)

    def test_language_filter_returns_empty_when_metadata_lacks_language(
        self, basic_index: BM25Index
    ) -> None:
        """Documented limitation: real upsert_code_chunks metadata has no
        'language' key, so any language filter against it is unsatisfiable
        and must return [] rather than silently ignoring the filter."""
        results = BM25Search(basic_index).search("get_user_by_id", language="python")
        assert results == []


class TestCombinedFilters:
    def test_symbol_type_and_file_path_together(
        self, boost_scenario_index: BM25Index
    ) -> None:
        results = BM25Search(boost_scenario_index).search(
            "get_user_by_id", symbol_type="function", file_path="src/app.py"
        )
        assert results
        assert all(
            r.file_path == "src/app.py" and r.metadata.get("symbol_type") == "function"
            for r in results
        )


# ---------------------------------------------------------------------------
# top_k / candidate pool behaviour
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_caps_results(self) -> None:
        index = BM25Index()
        for i in range(10):
            index.add(
                f"extra_{i}",
                f"def get_user_by_id(x): return lookup_{i}(x)",
                metadata={"file_path": f"src/extra_{i}.py", "symbol": f"extra_{i}"},
            )
        _add_filler(index)
        results = BM25Search(index).search("get_user_by_id", top_k=3)
        assert len(results) == 3

    def test_top_k_zero_raises(self, basic_index: BM25Index) -> None:
        with pytest.raises(ValueError, match="top_k"):
            BM25Search(basic_index).search("get_user_by_id", top_k=0)

    def test_top_k_negative_raises(self, basic_index: BM25Index) -> None:
        with pytest.raises(ValueError, match="top_k"):
            BM25Search(basic_index).search("get_user_by_id", top_k=-1)

    def test_boost_surfaces_result_outside_naive_top_k_window(self) -> None:
        """A low-raw-score exact match must still surface after boosting +
        widened candidate pooling, even if a caller asks for a small top_k."""
        index = BM25Index()
        # One exact-match definer with a low raw score (diluted content).
        index.add(
            "def_chunk",
            (
                "def get_user_by_id(user_id): "
                "'''Long docstring padding out this chunk considerably so the "
                "raw term frequency ratio for the identifier is low relative "
                "to chunks that just repeat it.''' "
                "return db.query(User).filter(User.id == user_id).first()"
            ),
            metadata={
                "file_path": "src/app.py",
                "symbol": "get_user_by_id",
                "symbol_type": "function",
            },
        )
        # Several higher-raw-score "noise" chunks that repeat the token
        # without being the definer, to push the definer down the raw
        # ranking below a top_k=1 window.
        for i in range(5):
            index.add(
                f"noise_{i}",
                "get_user_by_id get_user_by_id get_user_by_id",
                metadata={"file_path": f"src/noise_{i}.py", "symbol": f"noise_{i}"},
            )
        _add_filler(index)

        # Confirm the definer is NOT the raw top-1 (i.e. this scenario is
        # actually exercising the widened candidate pool, not a no-op).
        raw_top1 = BM25Search(index).search(
            "get_user_by_id", top_k=1, boost_exact_match=False
        )
        assert raw_top1[0].symbol_name != "get_user_by_id"

        boosted_top1 = BM25Search(index).search("get_user_by_id", top_k=1)
        assert boosted_top1[0].symbol_name == "get_user_by_id"


# ---------------------------------------------------------------------------
# Failure cases and edge cases
# ---------------------------------------------------------------------------


class TestFailureAndEdgeCases:
    def test_empty_index_returns_empty(self) -> None:
        assert BM25Search(BM25Index()).search("anything") == []

    def test_no_index_or_path_given_starts_empty(self) -> None:
        """Omitting both bm25_index and index_path must not raise; it starts empty."""
        assert BM25Search().search("anything") == []

    def test_both_index_and_path_raises(self, tmp_path) -> None:
        index = BM25Index()
        path = tmp_path / "index.pkl"
        index.save(path)
        with pytest.raises(ValueError, match="index_path"):
            BM25Search(index, index_path=path)

    def test_no_lexical_overlap_returns_empty(self, basic_index: BM25Index) -> None:
        results = BM25Search(basic_index).search("completely_unrelated_zzz_query")
        assert results == []

    def test_whitespace_only_query_returns_empty(self, basic_index: BM25Index) -> None:
        assert BM25Search(basic_index).search("   ") == []

    def test_query_with_only_punctuation_returns_empty(
        self, basic_index: BM25Index
    ) -> None:
        """Punctuation-only queries tokenize to [] (no alnum runs) -> no results."""
        assert BM25Search(basic_index).search("!!!???...") == []

    def test_lazy_load_from_index_path(self, tmp_path) -> None:
        index = BM25Index()
        index.add(
            "a",
            "def get_user_by_id(user_id): return db.find(user_id)",
            metadata={"file_path": "src/app.py", "symbol": "get_user_by_id"},
        )
        _add_filler(index)
        path = tmp_path / "index.pkl"
        index.save(path)

        searcher = BM25Search(index_path=path)
        # Index isn't loaded until first use of the `.index` property/search.
        results = searcher.search("get_user_by_id")
        assert results
        assert results[0].symbol_name == "get_user_by_id"

    def test_lazy_load_is_cached_not_reloaded_per_search(self, tmp_path) -> None:
        """The `.index` property should only load once, then reuse the same object."""
        index = BM25Index()
        index.add("a", "def alpha(): pass", metadata={"symbol": "alpha"})
        _add_filler(index)
        path = tmp_path / "index.pkl"
        index.save(path)

        searcher = BM25Search(index_path=path)
        first = searcher.index
        second = searcher.index
        assert first is second

    def test_case_insensitive_matching(self, basic_index: BM25Index) -> None:
        """tokenize_code lower-cases everything, so case shouldn't matter."""
        results_lower = BM25Search(basic_index).search("get_user_by_id")
        results_upper = BM25Search(basic_index).search("GET_USER_BY_ID")
        assert [r.symbol_name for r in results_lower] == [
            r.symbol_name for r in results_upper
        ]

    def test_zero_score_results_are_never_returned(
        self, basic_index: BM25Index
    ) -> None:
        """BM25Index.search already drops <= 0 scores; confirm none leak through."""
        results = BM25Search(basic_index).search("get_user_by_id", top_k=100)
        assert all(r.score > 0 for r in results)

    def test_results_sorted_descending_by_score(
        self, boost_scenario_index: BM25Index
    ) -> None:
        results = BM25Search(boost_scenario_index).search("get_user_by_id", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_candidate_multiplier_is_configurable(self) -> None:
        """A tiny candidate_multiplier should still work (no crash) even if it
        limits how far outside the naive top_k window boosting can reach."""
        index = BM25Index()
        index.add("a", "def alpha(): pass", metadata={"symbol": "alpha"})
        _add_filler(index)
        searcher = BM25Search(index, candidate_multiplier=1)
        results = searcher.search("alpha", top_k=1)
        assert results

    def test_candidate_multiplier_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="candidate_multiplier"):
            BM25Search(candidate_multiplier=0)

    def test_candidate_multiplier_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="candidate_multiplier"):
            BM25Search(candidate_multiplier=-5)

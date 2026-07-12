"""Unit tests for the DocEmbedder pipeline.

Split into three layers:

1. **Pure helpers** -- whitespace/meaningfulness, comment extraction (stdlib
   ``tokenize``), README section splitting, and symbol traversal.  No model.
2. **Embedding engine** -- shape/dtype/normalisation, empty-skip, caching,
   deduplication, batching, and progress callbacks, all with an injected mock
   model (network-free).
3. **Documentation layer** -- ``embed_symbols`` / ``embed_comments`` /
   ``embed_readme`` producing ``DocEmbedding`` records linked to symbols.

A single opt-out semantic test exercises the real ``all-MiniLM-L6-v2`` model and
skips gracefully when it cannot be downloaded (offline CI).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from reporag.embedding.doc_embedder import (
    EMBEDDING_DIM,
    Comment,
    DocEmbedder,
    DocEmbedding,
    ReadmeSection,
    _enclosing_symbol,
    _flatten_symbols,
    _is_meaningful,
    _normalize_whitespace,
    _resolve_device,
    _symbol_line_ranges,
    extract_python_comments,
    iter_symbol_docstrings,
    split_readme_sections,
)

# ======================================================================
# Test doubles
# ======================================================================


class _FakeBatchEncoding(dict):
    """Minimal stand-in for tokenizer output that supports ``.to()``."""

    def to(self, device):
        return self


@dataclass
class _FakeSymbol:
    """Simulates the real ``Symbol`` dataclass from the ingestion module."""

    name: str
    type: str = "function"
    file_path: str = "auth.py"
    start_line: int = 1
    end_line: int = 10
    docstring: str | None = None
    qualified_name: str | None = None
    methods: list = field(default_factory=list)
    children: list = field(default_factory=list)


def _make_embedder(batch_size: int = 1) -> DocEmbedder:
    """Return a DocEmbedder with a mock model/tokenizer pre-injected.

    The mock model emits a fixed ``(batch_size, 5, 384)`` hidden state, so use
    this only when exactly *batch_size* texts reach the forward pass; use
    :func:`_make_counting_embedder` for variable batch sizes.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = _FakeBatchEncoding(
        {
            "input_ids": torch.ones(batch_size, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
        }
    )

    model = MagicMock()
    outputs = MagicMock()
    outputs.last_hidden_state = torch.randn(batch_size, 5, EMBEDDING_DIM)
    model.return_value = outputs
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    embedder = DocEmbedder(model_name="test/model", device="cpu")
    embedder._tokenizer = tokenizer
    embedder._model = model
    return embedder


def _make_counting_embedder(batch_size: int) -> DocEmbedder:
    """Return an embedder whose mock sizes each forward pass to the input."""
    call_sizes: list[int] = []

    tokenizer = MagicMock()

    def fake_tokenize(texts, **kwargs):
        n = len(texts)
        call_sizes.append(n)
        return _FakeBatchEncoding(
            {
                "input_ids": torch.ones(n, 5, dtype=torch.long),
                "attention_mask": torch.ones(n, 5, dtype=torch.long),
            }
        )

    tokenizer.side_effect = fake_tokenize

    model = MagicMock()

    def fake_forward(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.randn(n, 5, EMBEDDING_DIM)
        return out

    model.side_effect = fake_forward
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    embedder = DocEmbedder(model_name="test/model", device="cpu", batch_size=batch_size)
    embedder._tokenizer = tokenizer
    embedder._model = model
    embedder._call_sizes = call_sizes  # expose for assertions
    return embedder


# ======================================================================
# Layer 1: pure helpers
# ======================================================================

# -- _resolve_device ----------------------------------------------------


def test_resolve_device_explicit_cpu():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device("auto") == torch.device("cpu")


def test_resolve_device_explicit_cuda_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device("cuda") == torch.device("cpu")


# -- _normalize_whitespace / _is_meaningful -----------------------------


def test_normalize_whitespace_collapses_runs():
    assert _normalize_whitespace("  hello\n\n   world \t") == "hello world"


def test_is_meaningful_rejects_empty_and_punctuation():
    assert not _is_meaningful("")
    assert not _is_meaningful("   ")
    assert not _is_meaningful("...")
    assert not _is_meaningful("---")


def test_is_meaningful_accepts_real_text():
    assert _is_meaningful("TODO")
    assert _is_meaningful("verify JWT token")


# -- extract_python_comments -------------------------------------------


def test_extract_comments_basic():
    src = "x = 1  # assign one\n# standalone\n"
    comments = extract_python_comments(src)
    texts = [c.text for c in comments]
    assert "assign one" in texts
    assert "standalone" in texts


def test_extract_comments_ignores_hash_in_strings():
    """A '#' inside a string literal must not be read as a comment."""
    src = 's = "not # a comment"\nurl = "http://x#frag"\n'
    assert extract_python_comments(src) == []


def test_extract_comments_merges_consecutive_lines():
    """Adjacent comment lines collapse into a single block."""
    src = "# line one\n# line two\n# line three\n"
    comments = extract_python_comments(src)
    assert len(comments) == 1
    assert comments[0].text == "line one line two line three"
    assert comments[0].start_line == 1
    assert comments[0].end_line == 3


def test_extract_comments_separates_non_adjacent():
    src = "# first\nx = 1\n# second\n"
    comments = extract_python_comments(src)
    assert [c.text for c in comments] == ["first", "second"]


def test_extract_comments_skips_empty():
    src = "#\n#   \nx = 1\n"
    assert extract_python_comments(src) == []


def test_extract_comments_tolerates_malformed_source():
    """Truncated source returns recovered comments instead of raising."""
    src = "# good comment\ndef broken(\n"
    comments = extract_python_comments(src)
    assert any(c.text == "good comment" for c in comments)


# -- split_readme_sections ---------------------------------------------


def test_split_readme_basic_sections():
    md = "# Title\nIntro line.\n\n## Install\nRun pip install.\n"
    sections = split_readme_sections(md)
    headings = [s.heading for s in sections]
    assert headings == ["Title", "Install"]
    assert sections[1].level == 2
    assert "pip install" in sections[1].text


def test_split_readme_preamble_has_no_heading():
    md = "Some intro before any heading.\n\n# First\nBody.\n"
    sections = split_readme_sections(md)
    assert sections[0].heading is None
    assert "intro before" in sections[0].text
    assert sections[1].heading == "First"


def test_split_readme_ignores_headings_in_code_fence():
    md = "# Real\n```\n# not a heading\n```\nAfter.\n"
    sections = split_readme_sections(md)
    assert [s.heading for s in sections] == ["Real"]
    # The fenced '# not a heading' stays inside the Real section body.
    assert "not a heading" in sections[0].text


def test_split_readme_drops_empty_sections():
    md = "# Empty\n\n# Full\nHas content.\n"
    sections = split_readme_sections(md)
    # "# Empty" has no body -> only the heading text survives (meaningful),
    # while a truly empty document yields nothing.
    assert split_readme_sections("") == []
    assert any(s.heading == "Full" for s in sections)


def test_split_readme_records_line_numbers():
    md = "# One\nbody\n## Two\nbody2\n"
    sections = split_readme_sections(md)
    assert sections[0].start_line == 1
    assert sections[1].start_line == 3


# -- symbol traversal ---------------------------------------------------


def test_flatten_symbols_recurses_methods_and_children():
    method = _FakeSymbol(name="m", type="method")
    child = _FakeSymbol(name="c", type="class")
    root = _FakeSymbol(name="R", type="class", methods=[method], children=[child])
    names = {s.name for s in _flatten_symbols([root])}
    assert names == {"R", "m", "c"}


def test_iter_symbol_docstrings_skips_empty_and_recurses():
    method = _FakeSymbol(
        name="login",
        type="method",
        docstring="Authenticate the user.",
        qualified_name="Auth.login",
    )
    empty = _FakeSymbol(name="noop", docstring="   ", qualified_name="noop")
    none_doc = _FakeSymbol(name="bare", docstring=None, qualified_name="bare")
    root = _FakeSymbol(
        name="Auth",
        type="class",
        docstring="Auth service.",
        qualified_name="Auth",
        methods=[method, empty, none_doc],
    )
    pairs = dict((s.qualified_name, text) for s, text in iter_symbol_docstrings([root]))
    assert pairs == {"Auth": "Auth service.", "Auth.login": "Authenticate the user."}


def test_symbol_line_ranges_and_enclosing_prefers_narrowest():
    outer = _FakeSymbol(name="C", qualified_name="C", start_line=1, end_line=100)
    inner = _FakeSymbol(name="m", qualified_name="C.m", start_line=10, end_line=20)
    ranges = _symbol_line_ranges([outer, inner])
    # A line inside the method resolves to the method, not the class.
    assert _enclosing_symbol(15, ranges) == "C.m"
    # A line only inside the class resolves to the class.
    assert _enclosing_symbol(5, ranges) == "C"
    # A line outside everything resolves to nothing.
    assert _enclosing_symbol(200, ranges) is None


# ======================================================================
# Layer 2: embedding engine
# ======================================================================


def test_embedding_dim_is_384():
    assert EMBEDDING_DIM == 384
    assert DocEmbedder(model_name="test/model").embedding_dim == 384


def test_lazy_loading_does_not_load_model_at_init():
    embedder = DocEmbedder(model_name="test/model")
    assert not embedder._loaded
    assert embedder._model is None
    assert embedder._tokenizer is None


def test_embed_batch_shape_and_dtype():
    embedder = _make_embedder(batch_size=1)
    result = embedder.embed_batch(["Authenticate the user."])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, EMBEDDING_DIM)
    assert result.dtype == np.float32


def test_embed_batch_l2_normalised():
    embedder = _make_embedder(batch_size=3)
    result = embedder.embed_batch(["a", "b", "c"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_embed_batch_empty_returns_empty():
    embedder = DocEmbedder(model_name="test/model")
    result = embedder.embed_batch([])
    assert result.shape == (0, EMBEDDING_DIM)
    assert not embedder._loaded


def test_empty_strings_map_to_zero_vector_without_loading():
    """Whitespace-only inputs must never trigger a forward pass."""
    embedder = _make_embedder(batch_size=1)
    result = embedder.embed_batch(["", "   ", "\n\t"])
    assert result.shape == (3, EMBEDDING_DIM)
    assert np.count_nonzero(result) == 0
    assert not embedder._loaded  # model never loaded for all-empty input


def test_mixed_empty_and_real_inputs():
    embedder = _make_embedder(batch_size=1)
    result = embedder.embed_batch(["", "real docstring"])
    assert np.count_nonzero(result[0]) == 0
    np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0, rtol=1e-5)


def test_duplicates_in_batch_computed_once():
    embedder = _make_embedder(batch_size=1)
    model = embedder._model
    result = embedder.embed_batch(["dup", "dup", "dup"])
    assert result.shape == (3, EMBEDDING_DIM)
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[1], result[2])
    assert model.call_count == 1


def test_cache_prevents_recomputation():
    embedder = _make_embedder(batch_size=1)
    model = embedder._model
    first = embedder.embed_batch(["parse the request body"])
    calls_after_first = model.call_count
    second = embedder.embed_batch(["parse the request body"])
    np.testing.assert_array_equal(first, second)
    assert model.call_count == calls_after_first


def test_cache_stats_and_clear():
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["x"])  # miss
    embedder.embed_batch(["x"])  # hit
    assert embedder.cache_stats() == {"hits": 1, "misses": 1, "size": 1}
    embedder.clear_cache()
    assert embedder.cache_stats() == {"hits": 0, "misses": 0, "size": 0}


def test_cache_eviction_respects_maxsize():
    embedder = _make_embedder(batch_size=1)
    embedder._cache_maxsize = 2
    embedder.embed_batch(["a"])
    embedder.embed_batch(["b"])
    embedder.embed_batch(["c"])  # evicts "a"
    assert embedder.cache_stats()["size"] == 2


def test_cache_key_isolates_different_models():
    e1 = DocEmbedder(model_name="model-A", device="cpu")
    e2 = DocEmbedder(model_name="model-B", device="cpu")
    assert e1._cache_key("same text") != e2._cache_key("same text")


def test_empty_strings_not_cached():
    """Zero-vector empties should not consume cache slots."""
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["", "   "])
    assert embedder.cache_stats()["size"] == 0


def test_batching_splits_into_correct_chunks():
    embedder = _make_counting_embedder(batch_size=3)
    texts = [f"doc number {i}" for i in range(7)]
    result = embedder.embed_batch(texts)
    assert result.shape == (7, EMBEDDING_DIM)
    assert embedder._call_sizes == [3, 3, 1]


def test_batch_size_override_per_call():
    embedder = _make_counting_embedder(batch_size=100)
    texts = [f"doc number {i}" for i in range(6)]
    embedder.embed_batch(texts, batch_size=2)
    assert embedder._call_sizes == [2, 2, 2]


def test_batch_preserves_input_order():
    embedder = _make_counting_embedder(batch_size=1)
    result = embedder.embed_batch(["alpha", "beta", "gamma"])
    assert result.shape == (3, EMBEDDING_DIM)
    for i in range(3):
        np.testing.assert_allclose(np.linalg.norm(result[i]), 1.0, rtol=1e-5)


def test_progress_callback_reports_completion():
    embedder = _make_counting_embedder(batch_size=2)
    events: list[tuple[int, int]] = []
    embedder.embed_batch(
        [f"doc {i}" for i in range(5)],
        progress_callback=lambda done, total: events.append((done, total)),
    )
    # Every event reports the same total, is monotonic, and finishes complete.
    assert all(total == 5 for _, total in events)
    dones = [done for done, _ in events]
    assert dones == sorted(dones)
    assert events[-1] == (5, 5)


def test_progress_callback_counts_cache_hits_upfront():
    embedder = _make_counting_embedder(batch_size=10)
    embedder.embed_batch(["cached"])  # warm the cache
    events: list[tuple[int, int]] = []
    embedder.embed_batch(
        ["cached", "fresh"],
        progress_callback=lambda done, total: events.append((done, total)),
    )
    # First event fires after cache resolution: the hit is already counted.
    assert events[0] == (1, 2)
    assert events[-1] == (2, 2)


def test_embed_single():
    embedder = _make_embedder(batch_size=1)
    vec = embedder.embed("a single docstring")
    assert vec.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5)


def test_similarity_returns_float_in_range():
    embedder = _make_embedder(batch_size=2)
    score = embedder.similarity("auth", "login")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_similarity_with_empty_is_zero():
    embedder = _make_embedder(batch_size=1)
    assert embedder.similarity("", "real text") == 0.0


def test_device_propagated_to_model():
    embedder = _make_embedder(batch_size=1)
    embedder.embed_batch(["x"])
    embedder._model.to.assert_called_with(torch.device("cpu"))


def test_repr_is_informative():
    embedder = DocEmbedder(model_name="test/model", device="cpu")
    text = repr(embedder)
    assert "DocEmbedder" in text
    assert "test/model" in text


# ======================================================================
# Layer 3: documentation layer (linked DocEmbedding records)
# ======================================================================


def test_embed_symbols_links_docstrings_to_symbols():
    login = _FakeSymbol(
        name="login",
        type="method",
        file_path="auth.py",
        start_line=5,
        end_line=12,
        docstring="Verify the JWT token.",
        qualified_name="Auth.login",
    )
    auth = _FakeSymbol(
        name="Auth",
        type="class",
        file_path="auth.py",
        start_line=1,
        end_line=20,
        docstring="Authentication service.",
        qualified_name="Auth",
        methods=[login],
    )
    embedder = _make_counting_embedder(batch_size=8)
    docs = embedder.embed_symbols([auth])

    by_id = {d.symbol_id: d for d in docs}
    assert set(by_id) == {"Auth", "Auth.login"}
    for doc in docs:
        assert doc.doc_type == "docstring"
        assert doc.file_path == "auth.py"
        assert doc.vector.shape == (EMBEDDING_DIM,)
    assert by_id["Auth.login"].metadata["symbol_type"] == "method"


def test_embed_symbols_skips_empty_docstrings():
    good = _FakeSymbol(name="a", docstring="Real doc.", qualified_name="a")
    empty = _FakeSymbol(name="b", docstring="   ", qualified_name="b")
    none = _FakeSymbol(name="c", docstring=None, qualified_name="c")
    embedder = _make_counting_embedder(batch_size=8)
    docs = embedder.embed_symbols([good, empty, none])
    assert [d.symbol_id for d in docs] == ["a"]


def test_embed_symbols_empty_input_returns_empty_list():
    embedder = _make_embedder(batch_size=1)
    assert embedder.embed_symbols([]) == []
    assert not embedder._loaded


def test_embed_comments_produces_records():
    src = "# configure the retry budget\nx = 1\n"
    embedder = _make_counting_embedder(batch_size=8)
    docs = embedder.embed_comments(src, file_path="worker.py")
    assert len(docs) == 1
    assert docs[0].doc_type == "comment"
    assert docs[0].file_path == "worker.py"
    assert docs[0].text == "configure the retry budget"


def test_embed_comments_links_to_enclosing_symbol():
    src = "def run():\n    # inner note\n    return 1\n"
    fn = _FakeSymbol(name="run", qualified_name="run", start_line=1, end_line=3)
    embedder = _make_counting_embedder(batch_size=8)
    docs = embedder.embed_comments(src, file_path="w.py", symbols=[fn])
    assert docs[0].symbol_id == "run"


def test_embed_comments_no_comments_returns_empty():
    embedder = _make_embedder(batch_size=1)
    assert embedder.embed_comments("x = 1\n", file_path="w.py") == []
    assert not embedder._loaded


def test_embed_readme_sections():
    md = "# RepoRAG\nA code-aware RAG engine.\n\n## Usage\nRun the server.\n"
    embedder = _make_counting_embedder(batch_size=8)
    docs = embedder.embed_readme(md)
    assert len(docs) == 2
    for doc in docs:
        assert doc.doc_type == "readme"
        assert doc.symbol_id is None
    assert docs[1].metadata["heading"] == "Usage"
    assert docs[1].metadata["level"] == 2


# -- DocEmbedding record ------------------------------------------------


def test_doc_embedding_to_dict_is_json_serialisable():
    doc = DocEmbedding(
        text="Verify the JWT token.",
        vector=np.ones(EMBEDDING_DIM, dtype=np.float32),
        doc_type="docstring",
        symbol_id="Auth.login",
        file_path="auth.py",
        start_line=5,
        end_line=12,
        metadata={"symbol_type": "method"},
    )
    payload = doc.to_dict()
    assert isinstance(payload["vector"], list)
    assert len(payload["vector"]) == EMBEDDING_DIM
    assert payload["symbol_id"] == "Auth.login"
    # Round-trips through JSON without error.
    import json

    assert json.loads(json.dumps(payload))["doc_type"] == "docstring"


def test_doc_embedding_repr():
    doc = DocEmbedding(
        text="Verify the JWT token.",
        vector=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        doc_type="docstring",
        symbol_id="Auth.login",
        file_path="auth.py",
        start_line=5,
    )
    text = repr(doc)
    assert "docstring" in text
    assert "Auth.login" in text


# -- dataclass sanity ---------------------------------------------------


def test_comment_and_readme_section_are_frozen():
    c = Comment(text="hi", start_line=1, end_line=1)
    s = ReadmeSection(heading="H", level=1, text="H body", start_line=1)
    with pytest.raises(FrozenInstanceError):
        c.text = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        s.heading = "mutated"  # type: ignore[misc]


# ======================================================================
# Semantic acceptance test (real model; skips when unavailable)
# ======================================================================


@pytest.fixture(scope="module")
def real_embedder():
    """Load the real all-MiniLM-L6-v2 model, skipping the test if unavailable."""
    embedder = DocEmbedder(device="cpu")
    try:
        embedder.embed("warm up the model")
    except Exception as exc:  # noqa: BLE001 -- offline / download failure -> skip
        pytest.skip(f"all-MiniLM-L6-v2 unavailable (offline?): {exc}")
    return embedder


def test_authentication_query_close_to_jwt_docstring(real_embedder):
    """AC: an "authentication" query embeds close to a "verify JWT token" doc.

    We assert *relative* ordering (robust across model versions): the auth query
    is more similar to the JWT docstring than to an unrelated docstring.
    """
    query = "how does authentication work"
    jwt_doc = "Verify the JWT token and return the authenticated user."
    unrelated = "Parse a CSV file into a list of rows."

    sim_relevant = real_embedder.similarity(query, jwt_doc)
    sim_unrelated = real_embedder.similarity(query, unrelated)

    assert sim_relevant > sim_unrelated
    assert real_embedder.embed(jwt_doc).shape == (EMBEDDING_DIM,)

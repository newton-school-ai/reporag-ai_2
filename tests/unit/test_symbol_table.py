"""Unit tests for the global SymbolTable / registry (Issue 11).

Covers every acceptance criterion -- fully qualified ``module.class.method``
registration, exact-name lookup returning all cross-file matches, unique
qualified-name lookup, regex lookup (``test_.*`` finds test functions), and
JSON round-tripping -- plus the required edge cases: name collisions across
files, nested class methods, and module-level variables. A small end-to-end
test drives the registry from the extractor over the sample repo.
"""

from __future__ import annotations

import pytest

from src.reporag.graph.symbol_table import SymbolRecord, SymbolTable
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def extractor() -> SymbolExtractor:
    """Provide a shared SymbolExtractor (grammar loads once per test)."""
    return SymbolExtractor()


def _table_from_sources(
    extractor: SymbolExtractor, sources: dict[str, str]
) -> SymbolTable:
    """Extract every source and register it into a fresh SymbolTable."""
    table = SymbolTable()
    for path, code in sources.items():
        table.register_symbols(extractor.extract_from_source(code, file_path=path))
    return table


def _var(name: str, file_path: str, line: int, module: str) -> SymbolRecord:
    """Build a module-level variable record (the extractor does not emit these)."""
    return SymbolRecord(
        symbol_id="",
        name=name,
        qualified_name=f"{module}.{name}",
        type="variable",
        file_path=file_path,
        module=module,
        start_line=line,
        end_line=line,
    )


# ---------------------------------------------------------------------------
# Fully qualified registration
# ---------------------------------------------------------------------------


def test_module_level_function_is_qualified_by_module(
    extractor: SymbolExtractor,
) -> None:
    """A top-level ``def`` is registered as ``module.func``."""
    table = _table_from_sources(extractor, {"pkg/svc.py": "def run():\n    pass\n"})
    record = table.lookup_qualified("pkg.svc.run")
    assert record is not None
    assert record.name == "run"
    assert record.type == "function"
    assert record.parent is None
    assert record.file_path == "pkg/svc.py"


def test_nested_class_method_is_fully_qualified(extractor: SymbolExtractor) -> None:
    """A method is registered as ``module.Class.method`` with a class parent."""
    code = "class Calculator:\n    def add(self, x, y):\n        return x + y\n"
    table = _table_from_sources(extractor, {"math_utils.py": code})

    method = table.lookup_qualified("math_utils.Calculator.add")
    assert method is not None
    assert method.type == "method"
    assert method.name == "add"
    assert method.parent == "math_utils.Calculator"
    assert method.signature == "def add(self, x, y)"

    cls = table.lookup_qualified("math_utils.Calculator")
    assert cls is not None
    assert cls.type == "class"
    assert cls.parent is None


def test_init_file_folds_into_package_module(extractor: SymbolExtractor) -> None:
    """``pkg/__init__.py`` symbols are qualified under ``pkg`` (no ``__init__``)."""
    table = _table_from_sources(
        extractor, {"pkg/__init__.py": "def boot():\n    pass\n"}
    )
    assert table.lookup_qualified("pkg.boot") is not None


def test_signature_and_docstring_are_captured(extractor: SymbolExtractor) -> None:
    """Records carry the callable signature and docstring metadata."""
    code = 'def greet(name: str) -> str:\n    """Say hi."""\n    return name\n'
    table = _table_from_sources(extractor, {"g.py": code})
    record = table.lookup_qualified("g.greet")
    assert record is not None
    assert record.signature == "def greet(name: str) -> str"
    assert record.docstring == "Say hi."


def test_async_and_decorator_metadata_preserved(extractor: SymbolExtractor) -> None:
    """``async`` and decorators survive into the record."""
    code = "@staticmethod\nasync def fetch():\n    pass\n"
    table = _table_from_sources(extractor, {"a.py": code})
    record = table.lookup_qualified("a.fetch")
    assert record is not None
    assert record.is_async is True
    assert record.decorators == ["staticmethod"]


# ---------------------------------------------------------------------------
# Exact-name lookup (all matches)
# ---------------------------------------------------------------------------


def test_exact_name_lookup_returns_all_matches_across_files(
    extractor: SymbolExtractor,
) -> None:
    """A name defined in two files yields two records from ``lookup``."""
    table = _table_from_sources(
        extractor,
        {"a.py": "def handle():\n    pass\n", "b.py": "def handle():\n    pass\n"},
    )
    matches = table.lookup("handle")
    assert len(matches) == 2
    assert {r.qualified_name for r in matches} == {"a.handle", "b.handle"}
    assert {r.file_path for r in matches} == {"a.py", "b.py"}


def test_same_name_across_classes_stays_distinct(extractor: SymbolExtractor) -> None:
    """``save`` on two classes in one file are separate qualified records."""
    code = (
        "class User:\n    def save(self):\n        pass\n\n"
        "class Account:\n    def save(self):\n        pass\n"
    )
    table = _table_from_sources(extractor, {"m.py": code})
    matches = table.lookup("save")
    assert {r.qualified_name for r in matches} == {"m.User.save", "m.Account.save"}


def test_lookup_unknown_name_returns_empty_list(extractor: SymbolExtractor) -> None:
    """Looking up a name that was never registered returns ``[]``."""
    table = _table_from_sources(extractor, {"a.py": "def foo():\n    pass\n"})
    assert table.lookup("nonexistent") == []


# ---------------------------------------------------------------------------
# Qualified lookup (unique) and redefinition semantics
# ---------------------------------------------------------------------------


def test_qualified_lookup_returns_unique_match(extractor: SymbolExtractor) -> None:
    """The fully qualified name resolves to exactly one record."""
    table = _table_from_sources(
        extractor,
        {"a.py": "def handle():\n    pass\n", "b.py": "def handle():\n    pass\n"},
    )
    record = table.lookup_qualified("a.handle")
    assert record is not None
    assert record.file_path == "a.py"
    assert table.lookup_qualified("does.not.exist") is None


def test_redefinition_keeps_both_but_qualified_lookup_wins_latest(
    extractor: SymbolExtractor,
) -> None:
    """Redefining a name keeps both records; qualified lookup returns the last."""
    code = "def f():\n    return 1\n\n\ndef f():\n    return 2\n"
    table = _table_from_sources(extractor, {"m.py": code})

    # Both definitions are retained and reachable by exact name...
    all_f = table.lookup("f")
    assert len(all_f) == 2
    assert {r.symbol_id for r in all_f} == {"m.f", "m.f#2"}

    # ...but qualified lookup points at the most recent (line 5), like Python.
    # Ids are assignment-stable: the first def keeps the base id ``m.f`` and the
    # redefinition takes ``m.f#2`` -- which is what qualified lookup now returns.
    latest = table.lookup_qualified("m.f")
    assert latest is not None
    assert latest.start_line == 5
    assert latest.symbol_id == "m.f#2"
    # The shadowed first definition is still fetchable under the base id.
    shadowed = table.get("m.f")
    assert shadowed is not None
    assert shadowed.start_line == 1


# ---------------------------------------------------------------------------
# Regex lookup
# ---------------------------------------------------------------------------


def test_regex_lookup_finds_all_test_functions(extractor: SymbolExtractor) -> None:
    """``test_.*`` matches every test function and nothing else."""
    code = (
        "def test_login():\n    pass\n\n"
        "def test_logout():\n    pass\n\n"
        "def helper():\n    pass\n"
    )
    table = _table_from_sources(extractor, {"tests.py": code})
    names = {r.name for r in table.lookup_regex(r"test_.*")}
    assert names == {"test_login", "test_logout"}


def test_regex_lookup_over_qualified_names(extractor: SymbolExtractor) -> None:
    """``target='qualified'`` matches against the fully qualified name."""
    table = _table_from_sources(
        extractor,
        {"pkg/a.py": "def x():\n    pass\n", "other.py": "def x():\n    pass\n"},
    )
    matches = table.lookup_regex(r"^pkg\.", target="qualified")
    assert [r.qualified_name for r in matches] == ["pkg.a.x"]


def test_regex_lookup_over_file_paths(extractor: SymbolExtractor) -> None:
    """``target='file'`` matches against the file path."""
    table = _table_from_sources(
        extractor,
        {"pkg/a.py": "def x():\n    pass\n", "other.py": "def y():\n    pass\n"},
    )
    matches = table.lookup_regex(r"\.py$", target="file")
    assert {r.name for r in matches} == {"x", "y"}
    assert {r.name for r in table.lookup_regex(r"^pkg/", target="file")} == {"x"}


def test_regex_accepts_precompiled_pattern(extractor: SymbolExtractor) -> None:
    """A pre-compiled ``re.Pattern`` is accepted as well as a string."""
    import re

    table = _table_from_sources(extractor, {"m.py": "def alpha():\n    pass\n"})
    assert [r.name for r in table.lookup_regex(re.compile("^al"))] == ["alpha"]


# ---------------------------------------------------------------------------
# File lookup
# ---------------------------------------------------------------------------


def test_lookup_file_returns_definitions_in_order(extractor: SymbolExtractor) -> None:
    """``lookup_file`` returns every definition in a file, in definition order."""
    code = "def first():\n    pass\n\n\ndef second():\n    pass\n"
    table = _table_from_sources(
        extractor, {"a.py": code, "b.py": "def other():\n    pass\n"}
    )
    names = [r.name for r in table.lookup_file("a.py")]
    assert names == ["first", "second"]
    assert table.lookup_file("missing.py") == []


# ---------------------------------------------------------------------------
# Module-level variables (registered directly, extractor does not emit them)
# ---------------------------------------------------------------------------


def test_module_level_variables_can_be_registered_and_found() -> None:
    """The registry is source-agnostic: variable records look up like any other."""
    table = SymbolTable()
    table.add(_var("MAX_RETRIES", "config.py", 3, "config"))
    table.add(_var("MAX_RETRIES", "other.py", 7, "other"))

    both = table.lookup("MAX_RETRIES")
    assert len(both) == 2
    assert {r.type for r in both} == {"variable"}

    record = table.lookup_qualified("config.MAX_RETRIES")
    assert record is not None
    assert record.start_line == 3
    assert table.counts_by_type()["variable"] == 2


# ---------------------------------------------------------------------------
# Import handling
# ---------------------------------------------------------------------------


def test_imports_skipped_by_default(extractor: SymbolExtractor) -> None:
    """Import bindings do not pollute the registry by default."""
    code = "from db import get_user\n\n\ndef get_user_local():\n    pass\n"
    table = _table_from_sources(extractor, {"app.py": code})
    # The imported ``get_user`` binding is not registered as a definition.
    assert table.lookup("get_user") == []
    assert table.counts_by_type() == {"function": 1}


def test_imports_included_when_requested(extractor: SymbolExtractor) -> None:
    """``include_imports=True`` registers import bindings too."""
    table = SymbolTable()
    symbols = extractor.extract_from_source("import os\n", file_path="app.py")
    table.register_symbols(symbols, include_imports=True)
    assert table.counts_by_type().get("import") == 1


# ---------------------------------------------------------------------------
# JSON serialisation round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_is_lossless(extractor: SymbolExtractor) -> None:
    """to_json -> from_json reproduces an equivalent table, ids preserved."""
    code = (
        "class User:\n"
        '    """A user."""\n'
        "    def save(self):\n"
        "        pass\n\n\n"
        "def test_it():\n"
        "    pass\n"
    )
    table = _table_from_sources(extractor, {"m.py": code})
    table.add(_var("VERSION", "m.py", 1, "m"))

    restored = SymbolTable.from_json(table.to_json())

    assert restored.to_dict() == table.to_dict()
    assert len(restored) == len(table)
    assert restored.lookup_qualified("m.User.save") is not None
    assert {r.symbol_id for r in restored} == {r.symbol_id for r in table}


def test_from_dict_rejects_malformed_payload() -> None:
    """A non-list ``symbols`` value is rejected rather than silently ignored."""
    with pytest.raises(ValueError):
        SymbolTable.from_dict({"version": 1, "symbols": {"not": "a list"}})


def test_save_and_load_round_trip(tmp_path, extractor: SymbolExtractor) -> None:
    """save() then load() reconstructs the table from disk."""
    table = _table_from_sources(extractor, {"m.py": "def f():\n    pass\n"})
    path = tmp_path / "symbols.json"
    table.save(path)
    reloaded = SymbolTable.load(path)
    assert reloaded.to_dict() == table.to_dict()


# ---------------------------------------------------------------------------
# Container protocol
# ---------------------------------------------------------------------------


def test_container_protocol(extractor: SymbolExtractor) -> None:
    """len, iter, in, and the introspection helpers behave as documented."""
    table = _table_from_sources(
        extractor,
        {"a.py": "def foo():\n    pass\n", "b.py": "class Bar:\n    pass\n"},
    )
    assert len(table) == 2
    assert "a.foo" in table
    assert "missing" not in table
    assert {r.name for r in table} == {"foo", "Bar"}
    assert table.files == ["a.py", "b.py"]
    assert table.names() == ["Bar", "foo"]


# ---------------------------------------------------------------------------
# End-to-end over the sample repo
# ---------------------------------------------------------------------------


def test_end_to_end_over_sample_repo(extractor: SymbolExtractor) -> None:
    """Drive the registry from the extractor over the bundled sample repo."""
    table = SymbolTable()
    for path in (
        "examples/sample_repo/app.py",
        "examples/sample_repo/auth.py",
        "examples/sample_repo/db.py",
    ):
        table.register_symbols(extractor.extract_from_file(path))

    # authenticate_user is defined once, in auth.py.
    auth = table.lookup("authenticate_user")
    assert len(auth) == 1
    assert auth[0].qualified_name == "examples.sample_repo.auth.authenticate_user"

    # get_user_by_email is *defined* in db.py; the auth.py/app.py imports of it
    # are bindings and must not appear as definitions.
    getters = table.lookup("get_user_by_email")
    assert [r.file_path for r in getters] == ["examples/sample_repo/db.py"]

    # The User constructor resolves as a fully qualified nested method.
    init = table.lookup_qualified("examples.sample_repo.db.User.__init__")
    assert init is not None
    assert init.type == "method"

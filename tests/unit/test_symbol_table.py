"""Unit tests for the SymbolTable."""

from __future__ import annotations

import pytest

from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol


@pytest.fixture
def empty_table() -> SymbolTable:
    return SymbolTable()


def test_infer_module_name(empty_table: SymbolTable) -> None:
    """Verify module names are inferred correctly from file paths."""
    assert (
        empty_table._infer_module_name("src/reporag/graph/symbol_table.py")
        == "src.reporag.graph.symbol_table"
    )
    assert empty_table._infer_module_name("tests/unit/__init__.py") == "tests.unit"
    assert empty_table._infer_module_name("my_script.py") == "my_script"


def test_register_symbols(empty_table: SymbolTable) -> None:
    """Verify symbols and their nested children are flattened and registered."""
    method = Symbol(
        name="authenticate",
        type="method",
        file_path="auth.py",
        start_line=10,
        end_line=20,
        qualified_name="Auth.authenticate",
    )
    cls = Symbol(
        name="Auth",
        type="class",
        file_path="auth.py",
        start_line=5,
        end_line=25,
        methods=[method],
    )

    empty_table.register_symbols([cls])

    # Check that both cls and method are registered
    assert len(empty_table) == 2

    # Check fully qualified names are generated with the inferred module prefix
    auth_cls = empty_table.lookup("Auth")[0]
    auth_method = empty_table.lookup("authenticate")[0]

    assert auth_cls.qualified_name == "auth.Auth"
    assert auth_method.qualified_name == "auth.Auth.authenticate"


def test_lookup_exact_name(empty_table: SymbolTable) -> None:
    """Verify lookup by exact name returns all matches across files."""
    sym1 = Symbol(
        name="fetch", type="function", file_path="api.py", start_line=1, end_line=2
    )
    sym2 = Symbol(
        name="fetch", type="function", file_path="db.py", start_line=1, end_line=2
    )

    empty_table.register_symbols([sym1, sym2])

    results = empty_table.lookup("fetch")
    assert len(results) == 2
    assert {r.file_path for r in results} == {"api.py", "db.py"}


def test_lookup_by_qualified_name(empty_table: SymbolTable) -> None:
    """Verify lookup by fully qualified name returns the unique match."""
    sym = Symbol(
        name="fetch", type="function", file_path="src/api.py", start_line=1, end_line=2
    )
    empty_table.register_symbols([sym])

    results = empty_table.lookup_by_qualified_name("src.api.fetch")
    assert len(results) == 1
    assert results[0].name == "fetch"
    assert results[0].qualified_name == "src.api.fetch"


def test_lookup_by_regex(empty_table: SymbolTable) -> None:
    """Verify regex lookup works properly on names or qualified names."""
    sym1 = Symbol(
        name="test_login",
        type="function",
        file_path="auth_tests.py",
        start_line=1,
        end_line=2,
    )
    sym2 = Symbol(
        name="test_logout",
        type="function",
        file_path="auth_tests.py",
        start_line=4,
        end_line=5,
    )
    sym3 = Symbol(
        name="helper",
        type="function",
        file_path="auth_tests.py",
        start_line=7,
        end_line=8,
    )

    empty_table.register_symbols([sym1, sym2, sym3])

    results = empty_table.lookup_by_regex(r"^test_")
    assert len(results) == 2
    assert {r.name for r in results} == {"test_login", "test_logout"}


def test_lookup_by_file(empty_table: SymbolTable) -> None:
    """Verify we can fetch all symbols inside a specific file."""
    sym1 = Symbol(
        name="f1", type="function", file_path="a.py", start_line=1, end_line=2
    )
    sym2 = Symbol(
        name="f2", type="function", file_path="a.py", start_line=3, end_line=4
    )
    sym3 = Symbol(
        name="f3", type="function", file_path="b.py", start_line=1, end_line=2
    )

    empty_table.register_symbols([sym1, sym2, sym3])

    results = empty_table.lookup_by_file("a.py")
    assert len(results) == 2
    assert {r.name for r in results} == {"f1", "f2"}


def test_lookup_by_type(empty_table: SymbolTable) -> None:
    """Verify we can fetch all symbols of a specific type."""
    sym1 = Symbol(
        name="f1", type="function", file_path="a.py", start_line=1, end_line=2
    )
    sym2 = Symbol(name="C1", type="class", file_path="a.py", start_line=3, end_line=4)
    sym3 = Symbol(
        name="f2", type="function", file_path="b.py", start_line=1, end_line=2
    )

    empty_table.register_symbols([sym1, sym2, sym3])

    funcs = empty_table.lookup_by_type("function")
    classes = empty_table.lookup_by_type("class")

    assert len(funcs) == 2
    assert len(classes) == 1
    assert classes[0].name == "C1"


def test_module_level_vars(empty_table: SymbolTable) -> None:
    """Verify registration and lookup of module-level variables."""
    sym = Symbol(
        name="GLOBAL_TIMEOUT",
        type="variable",
        file_path="config.py",
        start_line=10,
        end_line=10,
        docstring="The global timeout",
    )
    empty_table.register_symbols([sym])

    results = empty_table.lookup("GLOBAL_TIMEOUT")
    assert len(results) == 1
    assert results[0].type == "variable"
    assert results[0].qualified_name == "config.GLOBAL_TIMEOUT"


def test_collection_protocols(empty_table: SymbolTable) -> None:
    """Verify pythonic container behaviors like len(), in, and iter()."""
    sym = Symbol(
        name="fetch", type="function", file_path="api.py", start_line=1, end_line=2
    )
    empty_table.register_symbols([sym])

    # __len__
    assert len(empty_table) == 1

    # __contains__
    assert "fetch" in empty_table
    assert "api.fetch" in empty_table
    assert "missing" not in empty_table

    # __iter__
    records = list(empty_table)
    assert len(records) == 1
    assert records[0].name == "fetch"


def test_lookup_by_file_pattern(empty_table: SymbolTable) -> None:
    """Verify we can find symbols across directories using glob patterns."""
    sym1 = Symbol(
        name="f1", type="function", file_path="src/api.py", start_line=1, end_line=2
    )
    sym2 = Symbol(
        name="f2", type="function", file_path="src/db.py", start_line=1, end_line=2
    )
    sym3 = Symbol(
        name="t1",
        type="function",
        file_path="tests/test_api.py",
        start_line=1,
        end_line=2,
    )

    empty_table.register_symbols([sym1, sym2, sym3])

    results = empty_table.lookup_by_file_pattern("src/*.py")
    assert len(results) == 2
    assert {r.name for r in results} == {"f1", "f2"}

    results = empty_table.lookup_by_file_pattern("tests/*")
    assert len(results) == 1
    assert results[0].name == "t1"


def test_lookup_by_position(empty_table: SymbolTable) -> None:
    """Verify we can find the innermost symbol enclosing a specific line."""
    method = Symbol(
        name="inner",
        type="method",
        file_path="auth.py",
        start_line=10,
        end_line=20,
        qualified_name="Auth.inner",
    )
    cls = Symbol(
        name="Auth",
        type="class",
        file_path="auth.py",
        start_line=5,
        end_line=25,
        methods=[method],
    )
    empty_table.register_symbols([cls])

    # Line 15 is inside both 'Auth' and 'inner', but 'inner' is tighter
    innermost = empty_table.lookup_by_position("auth.py", 15)
    assert innermost is not None
    assert innermost.name == "inner"

    # Line 6 is inside 'Auth' but outside 'inner'
    outer = empty_table.lookup_by_position("auth.py", 6)
    assert outer is not None
    assert outer.name == "Auth"

    # Line 30 is outside all symbols
    assert empty_table.lookup_by_position("auth.py", 30) is None


def test_clear(empty_table: SymbolTable) -> None:
    """Verify clearing the table safely wipes all data and indices."""
    sym = Symbol(name="f1", type="function", file_path="a.py", start_line=1, end_line=2)
    empty_table.register_symbols([sym])
    assert len(empty_table) == 1

    empty_table.clear()

    assert len(empty_table) == 0
    assert empty_table.lookup("f1") == []
    assert empty_table.lookup_by_type("function") == []
    assert empty_table.lookup_by_file("a.py") == []


def test_remove_by_file_and_update(empty_table: SymbolTable) -> None:
    """Verify that removing and updating files cleanly maintains index integrity."""
    method = Symbol(
        name="inner",
        type="method",
        file_path="auth.py",
        start_line=10,
        end_line=20,
        qualified_name="Auth.inner",
    )
    cls = Symbol(
        name="Auth",
        type="class",
        file_path="auth.py",
        start_line=5,
        end_line=25,
        methods=[method],
    )
    sym2 = Symbol(
        name="other", type="function", file_path="other.py", start_line=1, end_line=5
    )

    empty_table.register_symbols([cls, sym2])
    assert len(empty_table) == 3

    # Remove auth.py
    empty_table.remove_by_file("auth.py")

    assert len(empty_table) == 1
    assert len(empty_table.lookup_by_file("auth.py")) == 0
    assert len(empty_table.lookup("Auth")) == 0
    assert len(empty_table.lookup("inner")) == 0
    assert len(empty_table.lookup("other")) == 1

    # Update auth.py with new symbols
    new_sym = Symbol(
        name="NewAuth", type="class", file_path="auth.py", start_line=1, end_line=10
    )
    empty_table.update_file("auth.py", [new_sym])

    assert len(empty_table) == 2
    assert len(empty_table.lookup_by_file("auth.py")) == 1
    assert empty_table.lookup("NewAuth")[0].name == "NewAuth"


def test_serialization(empty_table: SymbolTable) -> None:
    """Verify JSON serialization and deserialization retains all data and rebuilds indices."""
    sym = Symbol(
        name="MyClass",
        type="class",
        file_path="models.py",
        start_line=10,
        end_line=20,
        docstring="Test class",
        signature=None,
    )
    empty_table.register_symbols([sym])

    json_data = empty_table.to_json()
    new_table = SymbolTable.from_json(json_data)

    assert len(new_table) == 1
    record = new_table.lookup("MyClass")[0]

    assert record.name == "MyClass"
    assert record.file_path == "models.py"
    assert record.docstring == "Test class"

    # Check that indices were rebuilt successfully
    assert len(new_table.lookup("MyClass")) == 1
    assert len(new_table.lookup_by_qualified_name("models.MyClass")) == 1
    assert len(new_table.lookup_by_file("models.py")) == 1
    assert len(new_table.lookup_by_type("class")) == 1


def test_hierarchy_and_relational_pointers(empty_table: SymbolTable) -> None:
    """Verify parent/child relationships and context breadcrumbs."""
    method = Symbol(
        name="inner",
        type="method",
        file_path="auth.py",
        start_line=10,
        end_line=20,
        qualified_name="Auth.inner",
    )
    cls = Symbol(
        name="Auth",
        type="class",
        file_path="auth.py",
        start_line=5,
        end_line=25,
        methods=[method],
    )
    empty_table.register_symbols([cls])

    # 1. Relational Graph Pointers
    auth_rec = empty_table.lookup("Auth")[0]
    inner_rec = empty_table.lookup("inner")[0]

    assert inner_rec.parent_id == auth_rec.symbol_id
    assert empty_table.get_parent(inner_rec.symbol_id) == auth_rec

    children = empty_table.get_children(auth_rec.symbol_id)
    assert len(children) == 1
    assert children[0] == inner_rec

    # 2. Context Breadcrumbs
    hierarchy = empty_table.lookup_hierarchy_by_position("auth.py", 15)
    assert len(hierarchy) == 2
    assert hierarchy[0].name == "Auth"
    assert hierarchy[1].name == "inner"


def test_fuzzy_search(empty_table: SymbolTable) -> None:
    """Verify fuzzy typo-tolerant searching."""
    sym1 = Symbol(
        name="authenticate_user",
        type="function",
        file_path="auth.py",
        start_line=1,
        end_line=2,
    )
    sym2 = Symbol(
        name="fetch_data", type="function", file_path="api.py", start_line=1, end_line=2
    )
    empty_table.register_symbols([sym1, sym2])

    # User makes a typo 'auth_user'
    results = empty_table.lookup_fuzzy("auth_user")
    assert len(results) == 1
    assert results[0].name == "authenticate_user"

    # User types 'fet_dat'
    results = empty_table.lookup_fuzzy("fet_dat")
    assert len(results) == 1
    assert results[0].name == "fetch_data"

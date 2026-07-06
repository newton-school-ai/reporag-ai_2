"""Unit tests for the SymbolTable (Issue 11)."""

from __future__ import annotations

import json

from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol


def test_symbol_table_initialization() -> None:
    """An initialized SymbolTable is empty."""
    table = SymbolTable()
    assert len(table.registry) == 0


def test_name_collision_across_files() -> None:
    """Name collision across files resolves properly.

    - Lookup by exact name returns all matches across files.
    - Lookup by qualified name returns unique match.
    """
    table = SymbolTable()

    sym1 = Symbol(
        name="authenticate",
        type="function",
        file_path="src/auth.py",
        start_line=10,
        end_line=20,
        signature="def authenticate(username, password)",
        docstring="Authenticate user.",
    )
    sym2 = Symbol(
        name="authenticate",
        type="function",
        file_path="src/api/v1/auth.py",
        start_line=15,
        end_line=25,
        signature="def authenticate(token)",
        docstring="Authenticate API token.",
    )

    table.register_symbols([sym1, sym2])

    # Lookup by exact name returns both matches
    results = table.lookup("authenticate")
    assert len(results) == 2
    paths = {r.file_path for r in results}
    assert paths == {"src/auth.py", "src/api/v1/auth.py"}

    # Lookup by qualified name returns unique match
    res1 = table.lookup("src.auth.authenticate")
    assert len(res1) == 1
    assert res1[0].file_path == "src/auth.py"

    res2 = table.lookup("src.api.v1.auth.authenticate")
    assert len(res2) == 1
    assert res2[0].file_path == "src/api/v1/auth.py"


def test_nested_class_methods() -> None:
    """Nested class methods are registered and resolved using fully qualified names."""
    table = SymbolTable()

    method_sym = Symbol(
        name="my_method",
        type="method",
        file_path="src/app.py",
        start_line=12,
        end_line=18,
        signature="def my_method(self)",
        docstring="My method docstring.",
        qualified_name="MyClass.my_method",
    )

    class_sym = Symbol(
        name="MyClass",
        type="class",
        file_path="src/app.py",
        start_line=5,
        end_line=30,
        docstring="My class.",
        qualified_name="MyClass",
        methods=[method_sym],
    )

    table.register_symbols([class_sym])

    # Method should be registered recursively
    results = table.lookup("my_method")
    assert len(results) == 1
    assert results[0].qualified_name == "src.app.MyClass.my_method"

    # Class lookup by suffix fully qualified name
    class_results = table.lookup("MyClass")
    assert len(class_results) == 1
    assert class_results[0].qualified_name == "src.app.MyClass"

    # Lookup by fully qualified name
    res_fq = table.lookup("src.app.MyClass.my_method")
    assert len(res_fq) == 1
    assert res_fq[0].start_line == 12


def test_module_level_variables() -> None:
    """Module-level variables can be registered and resolved."""
    table = SymbolTable()

    # Mock variables as Symbol objects
    var1 = Symbol(
        name="DEFAULT_TIMEOUT",
        type="variable",
        file_path="src/config.py",
        start_line=5,
        end_line=5,
        signature="DEFAULT_TIMEOUT = 30",
        docstring="Default timeout value.",
    )

    table.register_symbols([var1])

    results = table.lookup("DEFAULT_TIMEOUT")
    assert len(results) == 1
    assert results[0].qualified_name == "src.config.DEFAULT_TIMEOUT"
    assert results[0].type == "variable"

    res_fq = table.lookup("src.config.DEFAULT_TIMEOUT")
    assert len(res_fq) == 1
    assert res_fq[0].signature == "DEFAULT_TIMEOUT = 30"


def test_regex_lookup() -> None:
    """Regex lookup finds all matching functions/symbols."""
    table = SymbolTable()

    symbols = [
        Symbol(
            name="test_login",
            type="function",
            file_path="tests/test_auth.py",
            start_line=5,
            end_line=15,
        ),
        Symbol(
            name="test_logout",
            type="function",
            file_path="tests/test_auth.py",
            start_line=20,
            end_line=30,
        ),
        Symbol(
            name="helper_func",
            type="function",
            file_path="tests/test_auth.py",
            start_line=35,
            end_line=40,
        ),
    ]

    table.register_symbols(symbols)

    # Search for all test functions
    test_results = table.lookup("test_.*")
    assert len(test_results) == 2
    names = {r.name for r in test_results}
    assert names == {"test_login", "test_logout"}

    # Search by anchor
    logout_results = table.lookup(".*logout$")
    assert len(logout_results) == 1
    assert logout_results[0].name == "test_logout"


def test_lookup_by_file_path() -> None:
    """Lookup by file path matches correctly."""
    table = SymbolTable()

    symbols = [
        Symbol(
            name="foo",
            type="function",
            file_path="src/foo.py",
            start_line=1,
            end_line=5,
        ),
        Symbol(
            name="bar",
            type="function",
            file_path="src/bar.py",
            start_line=1,
            end_line=5,
        ),
    ]

    table.register_symbols(symbols)

    # Exact file path match
    res_exact = table.lookup("src/foo.py")
    assert len(res_exact) == 1
    assert res_exact[0].name == "foo"

    # Endswith suffix match
    res_suffix = table.lookup("bar.py")
    assert len(res_suffix) == 1
    assert res_suffix[0].name == "bar"


def test_json_serialization_deserialization() -> None:
    """SymbolTable can be serialized to and deserialized from JSON."""
    table = SymbolTable()

    sym = Symbol(
        name="process_data",
        type="function",
        file_path="src/processor.py",
        start_line=10,
        end_line=20,
        signature="def process_data(data)",
        docstring="Process incoming data.",
    )

    table.register_symbols([sym])

    # Serialize
    json_str = table.to_json()
    assert isinstance(json_str, str)

    # Check that it's valid JSON containing expected data
    parsed_json = json.loads(json_str)
    assert len(parsed_json) == 1
    key = list(parsed_json.keys())[0]
    assert key == "src/processor.py::src.processor.process_data"
    assert parsed_json[key]["name"] == "process_data"

    # Deserialize
    loaded_table = SymbolTable.from_json(json_str)
    assert len(loaded_table.registry) == 1

    # Verify lookup on loaded table
    results = loaded_table.lookup("process_data")
    assert len(results) == 1
    record = results[0]
    assert record.name == "process_data"
    assert record.file_path == "src/processor.py"
    assert record.start_line == 10
    assert record.end_line == 20
    assert record.signature == "def process_data(data)"
    assert record.docstring == "Process incoming data."
    assert record.qualified_name == "src.processor.process_data"

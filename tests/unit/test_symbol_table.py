"""Unit tests for the SymbolTable (Issue 11).

Covers the acceptance criteria -- fully qualified names, multiple lookup
modes (exact, qualified, regex, file path), JSON serialization, and
handling name collisions across files -- plus registration/re-registration
semantics and import-specific metadata.
"""

from __future__ import annotations

import re

import pytest

from src.reporag.graph.symbol_table import SymbolRecord, SymbolTable
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def table() -> SymbolTable:
    """Provide a fresh, empty SymbolTable for each test."""
    return SymbolTable()


SAMPLE_A = """
class Foo:
    def process(self):
        pass

    def other(self):
        pass

def process():
    pass

import os
from . import sibling
from typing import Dict as D
"""

SAMPLE_B = """
def process():
    pass
"""


# ---------------------------------------------------------------------------
# Fully qualified names
# ---------------------------------------------------------------------------


def test_module_level_function_gets_module_qualified_name(table: SymbolTable) -> None:
    """A module-level function is registered as ``module.function_name``."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    matches = table.lookup_qualified("pkg.mod.process")
    assert len(matches) == 1
    assert matches[0].type == "function"


def test_method_gets_module_class_method_qualified_name(table: SymbolTable) -> None:
    """A method is registered as ``module.Class.method``."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    matches = table.lookup_qualified("pkg.mod.Foo.process")
    assert len(matches) == 1
    record = matches[0]
    assert record.type == "method"
    assert record.parent_symbol == "Foo"
    assert record.parent_qualified_name == "pkg.mod.Foo"


def test_class_gets_module_qualified_name(table: SymbolTable) -> None:
    """A class is registered as ``module.ClassName``."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    matches = table.lookup_qualified("pkg.mod.Foo")
    assert len(matches) == 1
    assert matches[0].type == "class"


def test_import_gets_module_qualified_bound_name(table: SymbolTable) -> None:
    """An import is registered as ``module.bound_name``."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": "import os\n"})
    matches = table.lookup_qualified("pkg.mod.os")
    assert len(matches) == 1
    assert matches[0].type == "import"
    assert matches[0].import_source == "os"


def test_init_module_strips_init_from_qualified_name(table: SymbolTable) -> None:
    """A symbol in __init__.py is qualified under the package name, not '__init__'."""
    table = SymbolTable.build_from_sources(
        {"pkg/__init__.py": "def helper():\n    pass\n"}
    )
    matches = table.lookup_qualified("pkg.helper")
    assert len(matches) == 1


def test_all_symbols_get_a_symbol_id(table: SymbolTable) -> None:
    """Every registered symbol has a non-empty, unique symbol_id."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    ids = [r.symbol_id for r in table.all()]
    assert all(ids)
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Lookup modes
# ---------------------------------------------------------------------------


def test_lookup_exact_matches_bare_name(table: SymbolTable) -> None:
    """lookup_exact matches the bare name regardless of file/class."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    matches = table.lookup_exact("process")
    assert len(matches) == 3  # Foo.process, module-level process, mod_b's process


def test_lookup_exact_no_match_returns_empty_list(table: SymbolTable) -> None:
    """lookup_exact for an unknown name returns an empty list, not an error."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    assert table.lookup_exact("does_not_exist") == []


def test_lookup_qualified_disambiguates_exact_match(table: SymbolTable) -> None:
    """lookup_qualified returns exactly the one symbol named precisely."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    matches = table.lookup_qualified("pkg.mod_b.process")
    assert len(matches) == 1
    assert matches[0].file_path == "pkg/mod_b.py"


def test_lookup_regex_on_qualified_name(table: SymbolTable) -> None:
    """lookup_regex matches every qualified_name ending in 'process'."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    matches = table.lookup_regex(r"process$")
    assert {m.qualified_name for m in matches} == {
        "pkg.mod_a.Foo.process",
        "pkg.mod_a.process",
        "pkg.mod_b.process",
    }


def test_lookup_regex_accepts_compiled_pattern(table: SymbolTable) -> None:
    """lookup_regex also accepts compiled regex patterns."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})

    pattern = re.compile(r"process$")
    matches = table.lookup_regex(pattern)

    assert len(matches) == 2


def test_lookup_regex_on_name_field(table: SymbolTable) -> None:
    """lookup_regex can match against the bare name instead of qualified_name."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    matches = table.lookup_regex(r"^oth", field_name="name")
    assert [m.name for m in matches] == ["other"]


def test_lookup_regex_invalid_field_raises(table: SymbolTable) -> None:
    """lookup_regex rejects an unsupported field_name."""
    with pytest.raises(ValueError, match="field_name"):
        table.lookup_regex(r".*", field_name="file_path")


def test_lookup_by_file_returns_only_that_files_symbols(table: SymbolTable) -> None:
    """lookup_by_file scopes results to exactly one file."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    matches = table.lookup_by_file("pkg/mod_b.py")
    assert len(matches) == 1
    assert matches[0].qualified_name == "pkg.mod_b.process"


def test_get_by_symbol_id(table: SymbolTable) -> None:
    """get() retrieves the exact record for a known symbol_id."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_B})
    [record] = table.lookup_qualified("pkg.mod.process")
    assert table.get(record.symbol_id) is record


def test_get_unknown_symbol_id_returns_none(table: SymbolTable) -> None:
    """get() returns None (not an error) for an unregistered id."""
    assert table.get("nonexistent::id:1") is None


# ---------------------------------------------------------------------------
# Name collisions across files
# ---------------------------------------------------------------------------


def test_same_name_different_files_disambiguated_by_qualified_name(
    table: SymbolTable,
) -> None:
    """Two files both defining 'process' collide on exact name but not qualified name."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_B, "pkg/mod_b.py": SAMPLE_B}
    )
    assert len(table.lookup_exact("process")) == 2
    assert len(table.lookup_qualified("pkg.mod_a.process")) == 1
    assert len(table.lookup_qualified("pkg.mod_b.process")) == 1


def test_rebinding_same_name_same_file_produces_distinct_records(
    table: SymbolTable,
) -> None:
    """Two from-imports rebinding the same name share a qualified_name but differ by id."""
    table = SymbolTable.build_from_sources(
        {"app.py": "from a import shared\nfrom b import shared\n"}
    )
    matches = table.lookup_qualified("app.shared")
    assert len(matches) == 2
    assert matches[0].symbol_id != matches[1].symbol_id
    assert {m.import_source for m in matches} == {"a", "b"}


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_to_json_round_trip_preserves_all_records(table: SymbolTable) -> None:
    """to_json / from_json round-trips without losing or altering any record."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    restored = SymbolTable.from_json(table.to_json())
    assert restored.to_dict() == table.to_dict()
    assert len(restored) == len(table)


def test_to_dict_is_deterministically_sorted(table: SymbolTable) -> None:
    """to_dict output is sorted by symbol_id for stable diffs."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    ids = [r["symbol_id"] for r in table.to_dict()]
    assert ids == sorted(ids)


def test_symbol_record_to_dict_is_json_primitive(table: SymbolTable) -> None:
    """A single SymbolRecord.to_dict() contains only JSON-primitive values."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    record = table.lookup_qualified("pkg.mod.Foo")[0]
    import json

    json.dumps(record.to_dict())  # must not raise


def test_symbol_record_from_dict_round_trip() -> None:
    """SymbolRecord.from_dict(record.to_dict()) reconstructs an equal record."""
    record = SymbolRecord(
        symbol_id="a.py::a.foo:1",
        name="foo",
        qualified_name="a.foo",
        type="function",
        file_path="a.py",
        module="a",
        start_line=1,
        end_line=2,
    )
    assert SymbolRecord.from_dict(record.to_dict()) == record


# ---------------------------------------------------------------------------
# Registration / re-registration semantics
# ---------------------------------------------------------------------------


def test_register_symbols_returns_new_ids(table: SymbolTable) -> None:
    """register_symbols returns the symbol_id for every registered symbol."""
    symbols = SymbolExtractor().extract_from_source(
        SAMPLE_B,
        language="python",
        file_path="pkg/mod.py",
    )

    ids = table.register_symbols({"pkg/mod.py": symbols})
    assert len(ids) == 1
    assert ids[0]


def test_reregistering_same_symbol_id_does_not_duplicate(table: SymbolTable) -> None:
    """Re-registering the same file doesn't leave stale duplicate entries."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_B})
    before = len(table)
    symbols = table.lookup_by_file("pkg/mod.py")
    # Re-run registration using the freshly extracted symbols for the same file.
    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    fresh = SymbolExtractor().extract_from_source(
        SAMPLE_B, language="python", file_path="pkg/mod.py"
    )
    table.register_symbols({"pkg/mod.py": fresh})
    assert len(table) == before
    assert len(symbols) == len(table.lookup_by_file("pkg/mod.py"))


def test_unregister_file_removes_all_its_records(table: SymbolTable) -> None:
    """unregister_file clears every record for that file from every index."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod_a.py": SAMPLE_A, "pkg/mod_b.py": SAMPLE_B}
    )
    table.unregister_file("pkg/mod_a.py")
    assert table.lookup_by_file("pkg/mod_a.py") == []
    assert table.lookup_exact("process") == [
        r for r in table.all() if r.file_path == "pkg/mod_b.py"
    ]


def test_contains_and_len(table: SymbolTable) -> None:
    """__contains__ and __len__ behave as expected."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_B})
    [record] = table.all()
    assert record.symbol_id in table
    assert "not-a-real-id" not in table
    assert len(table) == 1


def test_build_from_symbols_matches_build_from_sources(table: SymbolTable) -> None:
    """build_from_symbols (Issue 7 entry point) matches build_from_sources exactly."""
    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    symbols_by_file = {
        "pkg/mod.py": SymbolExtractor().extract_from_source(
            SAMPLE_A, language="python", file_path="pkg/mod.py"
        )
    }
    via_symbols = SymbolTable.build_from_symbols(symbols_by_file)
    via_sources = SymbolTable.build_from_sources({"pkg/mod.py": SAMPLE_A})
    assert via_symbols.to_dict() == via_sources.to_dict()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def test_build_from_files(tmp_path) -> None:
    """build_from_files parses files from disk and registers their symbols."""
    source = tmp_path / "sample.py"
    source.write_text(
        """
def hello():
    pass
"""
    )

    table = SymbolTable.build_from_files([source])

    matches = table.lookup_exact("hello")
    assert len(matches) == 1
    assert matches[0].qualified_name.endswith("hello")


def test_build_from_files_skips_unreadable_file(tmp_path) -> None:
    """Unreadable files are skipped instead of failing the entire build."""
    good = tmp_path / "good.py"
    good.write_text(
        """
def ok():
    pass
"""
    )

    missing = tmp_path / "missing.py"

    table = SymbolTable.build_from_files([good, missing])

    assert len(table.lookup_exact("ok")) == 1


# ---------------------------------------------------------------------------
# Import metadata
# ---------------------------------------------------------------------------


def test_import_alias_metadata_preserved(table: SymbolTable) -> None:
    """Import aliases are preserved on SymbolRecord."""
    table = SymbolTable.build_from_sources(
        {"pkg/mod.py": "from typing import Dict as D\n"}
    )

    [record] = table.lookup_exact("D")

    assert record.import_alias == "D"
    assert record.import_source == "typing.Dict"


def test_wildcard_import_metadata_preserved(table: SymbolTable) -> None:
    """Wildcard imports are marked correctly."""
    table = SymbolTable.build_from_sources({"pkg/mod.py": "from utils import *\n"})

    [record] = table.lookup_exact("*")

    assert record.is_wildcard_import is True

"""Unit tests for the DependencyGraphBuilder (Issue 10).

Covers the acceptance criteria -- absolute imports, relative imports resolved to
absolute module paths, ``from X import *`` flagged, circular-import detection,
and edge metadata (source module, target module, import type, imported names) --
plus submodule from-imports, aliased imports, the source-based vs symbol-based
entry points, and the public :class:`DependencyEdge` contract.
"""

from __future__ import annotations

import pytest

from src.reporag.graph.dependency_graph import (
    DependencyEdge,
    DependencyGraphBuilder,
)
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def builder() -> DependencyGraphBuilder:
    """Provide a fresh DependencyGraphBuilder for each test."""
    return DependencyGraphBuilder()


def _edge(edges: list[DependencyEdge], target_module: str) -> DependencyEdge:
    """Return the single edge whose target is *target_module* (fails if absent)."""
    matches = [e for e in edges if e.target_module == target_module]
    assert matches, f"no edge -> {target_module} in {[str(e) for e in edges]}"
    assert len(matches) == 1, f"expected one -> {target_module}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Absolute imports
# ---------------------------------------------------------------------------


def test_absolute_plain_import(builder: DependencyGraphBuilder) -> None:
    """`import os` yields an external edge tagged ``import``."""
    edges = builder.build_from_sources({"m.py": "import os\n"})
    edge = _edge(edges, "os")
    assert edge.import_type == "import"
    assert edge.imported_names == []
    assert edge.is_external is True
    assert edge.resolved is False
    assert edge.target_file is None
    assert edge.source_module == "m"
    assert edge.line == 1


def test_absolute_from_import_records_names(builder: DependencyGraphBuilder) -> None:
    """`from flask import Flask, request` records the imported names."""
    edges = builder.build_from_sources({"m.py": "from flask import Flask, request\n"})
    edge = _edge(edges, "flask")
    assert edge.import_type == "from"
    assert sorted(edge.imported_names) == ["Flask", "request"]
    assert edge.is_external is True


def test_dotted_module_import(builder: DependencyGraphBuilder) -> None:
    """`import os.path` targets the full dotted module."""
    edges = builder.build_from_sources({"m.py": "import os.path\n"})
    edge = _edge(edges, "os.path")
    assert edge.import_type == "import"


def test_aliased_module_import(builder: DependencyGraphBuilder) -> None:
    """`import numpy as np` targets the source module, not the alias."""
    edges = builder.build_from_sources({"m.py": "import numpy as np\n"})
    edge = _edge(edges, "numpy")
    assert edge.import_type == "import"


def test_aliased_from_import_records_source_name(
    builder: DependencyGraphBuilder,
) -> None:
    """`from math import sin as s` targets ``math`` and records name ``sin``."""
    edges = builder.build_from_sources({"m.py": "from math import sin as s\n"})
    edge = _edge(edges, "math")
    assert edge.import_type == "from"
    assert edge.imported_names == ["sin"]


def test_multiple_targets_one_statement(builder: DependencyGraphBuilder) -> None:
    """`import os, sys` produces one edge per module."""
    edges = builder.build_from_sources({"m.py": "import os, sys\n"})
    assert {e.target_module for e in edges} == {"os", "sys"}


# ---------------------------------------------------------------------------
# Cross-file resolution
# ---------------------------------------------------------------------------


def test_cross_file_from_import_resolves(builder: DependencyGraphBuilder) -> None:
    """A from-import of a sibling project module resolves to its file."""
    sources = {
        "app.py": "from db import get_user\n",
        "db.py": "def get_user():\n    return None\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "db")
    assert edge.resolved is True
    assert edge.is_external is False
    assert edge.target_file == "db.py"
    assert edge.imported_names == ["get_user"]


def test_bare_import_resolves_to_canonical_module(
    builder: DependencyGraphBuilder,
) -> None:
    """A bare import resolves to the target file's canonical dotted name."""
    sources = {
        "pkg/app.py": "import helpers\n",
        "pkg/helpers.py": "x = 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.helpers")
    assert edge.source_module == "pkg.app"
    assert edge.resolved is True
    assert edge.target_file == "pkg/helpers.py"


def test_external_and_internal_mixed(builder: DependencyGraphBuilder) -> None:
    """A file mixing a project import and a stdlib import yields both edges."""
    sources = {
        "auth.py": "import hashlib\nfrom db import get_user\n",
        "db.py": "def get_user():\n    return None\n",
    }
    edges = builder.build_from_sources(sources)
    assert _edge(edges, "hashlib").is_external is True
    assert _edge(edges, "db").resolved is True


# ---------------------------------------------------------------------------
# Relative imports
# ---------------------------------------------------------------------------


def test_relative_import_resolves_to_absolute(builder: DependencyGraphBuilder) -> None:
    """`from .utils import helper` becomes an absolute dotted target module."""
    sources = {
        "pkg/app.py": "from .utils import helper\n",
        "pkg/utils.py": "def helper():\n    return 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.utils")
    assert edge.import_type == "relative"
    assert edge.is_relative is True
    assert edge.resolved is True
    assert edge.imported_names == ["helper"]


def test_relative_dot_import_submodule(builder: DependencyGraphBuilder) -> None:
    """`from . import sibling` targets the sibling submodule."""
    sources = {
        "pkg/app.py": "from . import sibling\n",
        "pkg/sibling.py": "x = 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.sibling")
    assert edge.import_type == "relative"
    assert edge.is_relative is True
    assert edge.resolved is True


def test_relative_parent_import(builder: DependencyGraphBuilder) -> None:
    """`from ..util import x` ascends one package level to an absolute target."""
    sources = {
        "pkg/sub/app.py": "from ..util import x\n",
        "pkg/util.py": "x = 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.util")
    assert edge.source_module == "pkg.sub.app"
    assert edge.is_relative is True
    assert edge.resolved is True


# ---------------------------------------------------------------------------
# Submodule vs member fan-out
# ---------------------------------------------------------------------------


def test_from_import_splits_submodule_and_member(
    builder: DependencyGraphBuilder,
) -> None:
    """A from-import fans out: submodule name -> its file, member name -> module."""
    sources = {
        "pkg/__init__.py": "helper = 1\n",
        "pkg/app.py": "from pkg import submod, helper\n",
        "pkg/submod.py": "y = 2\n",
    }
    edges = builder.build_from_sources(sources)
    sub_edge = _edge(edges, "pkg.submod")
    assert sub_edge.resolved is True
    assert sub_edge.imported_names == ["submod"]
    pkg_edge = _edge(edges, "pkg")
    assert pkg_edge.imported_names == ["helper"]
    assert pkg_edge.resolved is True  # pkg/__init__.py exists


# ---------------------------------------------------------------------------
# Wildcard imports
# ---------------------------------------------------------------------------


def test_wildcard_import_flagged(builder: DependencyGraphBuilder) -> None:
    """`from mod import *` is flagged as a wildcard with names ``['*']``."""
    sources = {
        "app.py": "from mod import *\n",
        "mod.py": "x = 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "mod")
    assert edge.import_type == "wildcard"
    assert edge.is_wildcard is True
    assert edge.imported_names == ["*"]
    assert edge.resolved is True


def test_relative_wildcard_is_wildcard_and_relative(
    builder: DependencyGraphBuilder,
) -> None:
    """`from .mod import *` is wildcard (primary label) and relative."""
    sources = {
        "pkg/app.py": "from .mod import *\n",
        "pkg/mod.py": "x = 1\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.mod")
    assert edge.import_type == "wildcard"
    assert edge.is_wildcard is True
    assert edge.is_relative is True


# ---------------------------------------------------------------------------
# Circular-import detection
# ---------------------------------------------------------------------------


def test_two_module_cycle_detected(builder: DependencyGraphBuilder) -> None:
    """Mutually importing modules form a detected cycle and edges are flagged."""
    sources = {
        "a.py": "from b import foo\n",
        "b.py": "from a import bar\n",
    }
    edges = builder.build_from_sources(sources)
    cycles = DependencyGraphBuilder.detect_cycles(edges)
    assert cycles == [["a", "b"]]
    assert _edge(edges, "b").in_cycle is True
    assert _edge(edges, "a").in_cycle is True


def test_three_module_cycle_detected(builder: DependencyGraphBuilder) -> None:
    """A -> B -> C -> A is reported as one cycle group of three modules."""
    sources = {
        "a.py": "from b import x\n",
        "b.py": "from c import y\n",
        "c.py": "from a import z\n",
    }
    edges = builder.build_from_sources(sources)
    assert DependencyGraphBuilder.detect_cycles(edges) == [["a", "b", "c"]]


def test_acyclic_graph_has_no_cycles(builder: DependencyGraphBuilder) -> None:
    """A linear import chain reports no cycles and flags no edges."""
    sources = {
        "a.py": "from b import x\n",
        "b.py": "from c import y\n",
        "c.py": "value = 1\n",
    }
    edges = builder.build_from_sources(sources)
    assert DependencyGraphBuilder.detect_cycles(edges) == []
    assert all(e.in_cycle is False for e in edges)


def test_external_imports_never_cycle(builder: DependencyGraphBuilder) -> None:
    """External modules cannot close a project cycle."""
    edges = builder.build_from_sources({"m.py": "import os\nimport sys\n"})
    assert DependencyGraphBuilder.detect_cycles(edges) == []


def test_cycle_paths_are_ordered_chains(builder: DependencyGraphBuilder) -> None:
    """`find_cycle_paths` returns an ordered chain that closes on itself."""
    sources = {
        "a.py": "from b import x\n",
        "b.py": "from c import y\n",
        "c.py": "from a import z\n",
    }
    edges = builder.build_from_sources(sources)
    paths = DependencyGraphBuilder.find_cycle_paths(edges)
    assert paths == [["a", "b", "c", "a"]]


def test_cycle_detected_via_symbol_api(builder: DependencyGraphBuilder) -> None:
    """Circular imports are detected through the issue's ``build`` API too."""
    extractor = SymbolExtractor()
    symbols_by_file = {
        "a.py": extractor.extract_from_source("from b import x\n", file_path="a.py"),
        "b.py": extractor.extract_from_source("from a import y\n", file_path="b.py"),
    }
    edges = builder.build(symbols_by_file)
    assert DependencyGraphBuilder.detect_cycles(edges) == [["a", "b"]]


# ---------------------------------------------------------------------------
# Entry points: symbol-based (issue API) and file-based
# ---------------------------------------------------------------------------


def test_build_from_symbols_by_file_api(builder: DependencyGraphBuilder) -> None:
    """The issue's ``build(symbols_by_file)`` API resolves cross-file imports."""
    extractor = SymbolExtractor()
    symbols_by_file = {
        "app.py": extractor.extract_from_source(
            "from db import get_user\n", file_path="app.py"
        ),
        "db.py": extractor.extract_from_source(
            "def get_user():\n    return None\n", file_path="db.py"
        ),
    }
    edges = builder.build(symbols_by_file)
    edge = _edge(edges, "db")
    assert edge.resolved is True
    assert edge.imported_names == ["get_user"]


def test_build_from_symbols_flat(builder: DependencyGraphBuilder) -> None:
    """The flat ``build_from_symbols`` wrapper groups symbols by file."""
    extractor = SymbolExtractor()
    symbols = extractor.extract_from_source(
        "from .utils import helper\n", file_path="pkg/app.py"
    ) + extractor.extract_from_source(
        "def helper():\n    pass\n", file_path="pkg/utils.py"
    )
    edges = builder.build_from_symbols(symbols)
    edge = _edge(edges, "pkg.utils")
    assert edge.is_relative is True
    assert edge.resolved is True


def test_build_from_files(tmp_path, builder: DependencyGraphBuilder) -> None:
    """``build_from_files`` parses files from disk and resolves imports."""
    (tmp_path / "app.py").write_text("from db import get_user\n", encoding="utf-8")
    (tmp_path / "db.py").write_text(
        "def get_user():\n    return None\n", encoding="utf-8"
    )
    edges = builder.build_from_files([tmp_path / "app.py", tmp_path / "db.py"])
    assert any(e.imported_names == ["get_user"] and e.resolved for e in edges)


def test_symbol_and_source_paths_agree_on_common_imports(
    builder: DependencyGraphBuilder,
) -> None:
    """The symbol-based and source-based paths agree on ordinary imports."""
    source = "import os\nfrom db import get_user\nfrom .utils import helper\n"
    files = {
        "pkg/app.py": source,
        "pkg/db.py": "def get_user():\n    return None\n",
        "pkg/utils.py": "def helper():\n    pass\n",
    }
    src_edges = builder.build_from_sources(files)

    extractor = SymbolExtractor()
    symbols_by_file = {
        path: extractor.extract_from_source(code, file_path=path)
        for path, code in files.items()
    }
    sym_edges = builder.build(symbols_by_file)

    def key(edges: list[DependencyEdge]) -> set[tuple[str, str, str]]:
        return {(e.source_module, e.target_module, e.import_type) for e in edges}

    assert key(src_edges) == key(sym_edges)


# ---------------------------------------------------------------------------
# Robustness and determinism
# ---------------------------------------------------------------------------


def test_syntax_error_tolerated(builder: DependencyGraphBuilder) -> None:
    """Broken source does not crash; valid imports are still extracted."""
    code = "import os\ndef broken(:\n    pass\nfrom sys import path\n"
    edges = builder.build_from_sources({"m.py": code})
    targets = {e.target_module for e in edges}
    assert "os" in targets


def test_import_inside_function_captured(builder: DependencyGraphBuilder) -> None:
    """An import nested inside a function body is still captured."""
    code = "def load():\n    import json\n    return json\n"
    edges = builder.build_from_sources({"m.py": code})
    assert _edge(edges, "json").import_type == "import"


def test_no_imports_yields_no_edges(builder: DependencyGraphBuilder) -> None:
    """A file with no imports produces no edges and does not crash."""
    edges = builder.build_from_sources({"m.py": "x = 1\ndef f():\n    return x\n"})
    assert edges == []


def test_deterministic_ordering(builder: DependencyGraphBuilder) -> None:
    """Edges are sorted by (source_file, line, target_module, import_type)."""
    sources = {"m.py": "import zzz\nimport aaa\nfrom bbb import c\n"}
    edges = builder.build_from_sources(sources)
    keys = [(e.source_file, e.line, e.target_module, e.import_type) for e in edges]
    assert keys == sorted(keys)


def test_aliased_dotted_import_ast_fidelity(builder: DependencyGraphBuilder) -> None:
    """The source-based path distinguishes ``import a.b as c`` from ``from a``."""
    plain = builder.build_from_sources({"m.py": "import a.b as c\n"})
    assert _edge(plain, "a.b").import_type == "import"

    frm = builder.build_from_sources({"m.py": "from a import b as c\n"})
    edge = _edge(frm, "a")
    assert edge.import_type == "from"
    assert edge.imported_names == ["b"]


# ---------------------------------------------------------------------------
# DependencyEdge contract
# ---------------------------------------------------------------------------


def test_edge_to_dict_is_json_primitive(builder: DependencyGraphBuilder) -> None:
    """``to_dict`` returns only JSON-primitive values (and string lists)."""
    edges = builder.build_from_sources({"m.py": "from flask import Flask\n"})
    payload = _edge(edges, "flask").to_dict()
    assert payload["source_module"] == "m"
    assert payload["target_module"] == "flask"
    assert payload["import_type"] == "from"
    assert payload["imported_names"] == ["Flask"]
    for value in payload.values():
        assert isinstance(value, str | int | bool | list | None)


def test_edge_resolved_external_mirror(builder: DependencyGraphBuilder) -> None:
    """``resolved`` and ``is_external`` are consistent inverses of each other."""
    edges = builder.build_from_sources(
        {"a.py": "from b import x\nimport os\n", "b.py": "x = 1\n"}
    )
    for edge in edges:
        assert edge.resolved is not edge.is_external

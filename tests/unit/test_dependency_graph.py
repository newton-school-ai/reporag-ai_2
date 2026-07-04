"""Unit tests for the DependencyGraphBuilder (Issue 10).

Covers the acceptance criteria: absolute, relative, and wildcard imports;
circular dependency detection; edge metadata formatting; and the public
entry points (`build_from_files`, `build_from_sources`, `build_from_symbols`).
"""

from __future__ import annotations

import pytest

from src.reporag.graph.dependency_graph import (
    DependencyEdge,
    DependencyGraphBuilder,
)


@pytest.fixture
def builder() -> DependencyGraphBuilder:
    """Provide a fresh DependencyGraphBuilder for each test."""
    return DependencyGraphBuilder()


def _edge(
    edges: list[DependencyEdge], source_module: str, target_module: str
) -> DependencyEdge:
    """Return the single edge matching source -> target (fails if absent)."""
    matches = [
        e
        for e in edges
        if e.source_module == source_module and e.target_module == target_module
    ]
    assert (
        matches
    ), f"no edge {source_module} -> {target_module} in {[str(e) for e in edges]}"
    assert (
        len(matches) == 1
    ), f"expected one {source_module} -> {target_module}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Absolute imports (intra-project)
# ---------------------------------------------------------------------------


def test_plain_import_intra_project(builder: DependencyGraphBuilder) -> None:
    """`import utils` resolves to the project file and yields an empty names list."""
    sources = {
        "utils.py": "def helper(): pass",
        "app.py": "import utils\nutils.helper()\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "app", "utils")
    assert edge.import_type == "import"
    assert edge.imported_names == []
    assert edge.resolved is True
    assert edge.target_file == "utils.py"
    assert edge.source_file == "app.py"


def test_from_import_intra_project(builder: DependencyGraphBuilder) -> None:
    """`from db import get_user` resolves to the module and records the name."""
    sources = {
        "db.py": "def get_user(): pass",
        "app.py": "from db import get_user\nget_user()\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "app", "db")
    assert edge.import_type == "from_import"
    assert edge.imported_names == ["get_user"]
    assert edge.target_file == "db.py"


def test_multiple_names_merge_into_one_edge(builder: DependencyGraphBuilder) -> None:
    """Multiple `from db import X` statements merge into a single edge."""
    sources = {
        "db.py": "def get_user(): pass\ndef get_role(): pass",
        "app.py": "from db import get_user\nfrom db import get_role\n",
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_module == "app"
    assert edge.target_module == "db"
    assert edge.imported_names == ["get_role", "get_user"]


def test_aliased_imports(builder: DependencyGraphBuilder) -> None:
    """`import X as Y` and `from M import X as Y` record the correct target module."""
    sources = {
        "math.py": "def sin(): pass",
        "utils.py": "def helper(): pass",
        "app.py": "import utils as u\nfrom math import sin as s\n",
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 2

    edge1 = _edge(edges, "app", "math")
    assert edge1.import_type == "from_import"
    assert edge1.imported_names == ["sin"]
    assert edge1.resolved is True

    edge2 = _edge(edges, "app", "utils")
    assert edge2.import_type == "import"
    assert edge2.imported_names == []
    assert edge2.resolved is True


# ---------------------------------------------------------------------------
# Relative imports
# ---------------------------------------------------------------------------


def test_relative_import_sibling(builder: DependencyGraphBuilder) -> None:
    """`from .utils import helper` resolves to a sibling module."""
    sources = {
        "pkg/utils.py": "def helper(): pass",
        "pkg/app.py": "from .utils import helper\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.app", "pkg.utils")
    assert edge.import_type == "from_import"
    assert edge.imported_names == ["helper"]
    assert edge.target_file == "pkg/utils.py"


def test_relative_import_parent(builder: DependencyGraphBuilder) -> None:
    """`from .. import core` resolves to the parent package."""
    sources = {
        "pkg/core.py": "def engine(): pass",
        "pkg/sub/app.py": "from .. import core\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "pkg.sub.app", "pkg.core")
    assert edge.target_file == "pkg/core.py"
    assert edge.import_type == "from_import"
    assert edge.imported_names == ["core"]


# ---------------------------------------------------------------------------
# Wildcard imports
# ---------------------------------------------------------------------------


def test_wildcard_import(builder: DependencyGraphBuilder) -> None:
    """`from config import *` is tagged wildcard and records `["*"]`."""
    sources = {
        "config.py": "DEBUG = True",
        "app.py": "from config import *\n",
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "app", "config")
    assert edge.import_type == "wildcard"
    assert edge.imported_names == ["*"]
    assert edge.target_file == "config.py"


# ---------------------------------------------------------------------------
# Circular dependencies
# ---------------------------------------------------------------------------


def test_circular_dependency_flagged(builder: DependencyGraphBuilder) -> None:
    """A -> B -> A sets is_circular=True on both edges."""
    sources = {
        "a.py": "from b import func_b\ndef func_a(): pass",
        "b.py": "from a import func_a\ndef func_b(): pass",
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 2
    edge1 = _edge(edges, "a", "b")
    edge2 = _edge(edges, "b", "a")
    assert edge1.is_circular is True
    assert edge2.is_circular is True


def test_non_circular_chain_not_flagged(builder: DependencyGraphBuilder) -> None:
    """A -> B -> C is not circular; is_circular=False."""
    sources = {
        "a.py": "import b",
        "b.py": "import c",
        "c.py": "x = 1",
    }
    edges = builder.build_from_sources(sources)
    for e in edges:
        assert e.is_circular is False


# ---------------------------------------------------------------------------
# External (unresolved) imports
# ---------------------------------------------------------------------------


def test_external_import_keeps_unresolved_edge(builder: DependencyGraphBuilder) -> None:
    """By default, third-party / built-in imports are retained but unresolved."""
    sources = {
        "app.py": "import os\nfrom flask import Flask\n",
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 2

    edge_os = _edge(edges, "app", "os")
    assert edge_os.resolved is False
    assert edge_os.target_file is None
    assert edge_os.import_type == "import"

    edge_flask = _edge(edges, "app", "flask")
    assert edge_flask.resolved is False
    assert edge_flask.imported_names == ["Flask"]


def test_external_imports_dropped_when_disabled(
    builder: DependencyGraphBuilder,
) -> None:
    """include_external=False drops unresolved external edges."""
    sources = {
        "utils.py": "x = 1",
        "app.py": "import os\nimport utils\n",
    }
    edges = builder.build_from_sources(sources, include_external=False)
    assert len(edges) == 1
    assert edges[0].target_module == "utils"


# ---------------------------------------------------------------------------
# Entry points and output contract
# ---------------------------------------------------------------------------


def test_build_from_files(builder: DependencyGraphBuilder) -> None:
    """End-to-end build over actual files works."""
    # We parse some of our own source files which use imports
    edges = builder.build_from_files(
        [
            "src/reporag/graph/dependency_graph.py",
            "src/reporag/graph/call_graph.py",
        ]
    )
    # The dependency graph builder itself imports _ModuleIndex from call_graph
    found = False
    for e in edges:
        if (
            e.source_module == "src.reporag.graph.dependency_graph"
            and e.target_module == "src.reporag.graph.call_graph"
        ):
            found = True
            assert e.resolved is True
            assert "_ModuleIndex" in e.imported_names
            break
    assert found, "Dependency on call_graph not found in dependency_graph"


def test_build_from_symbols_api(builder: DependencyGraphBuilder) -> None:
    """The lower-level build_from_symbols API works."""
    from src.reporag.ingestion.parser import ASTParser
    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    parser = ASTParser()
    extractor = SymbolExtractor(parser)

    sources = {
        "db.py": "def get_user(): pass",
        "app.py": "from db import get_user",
    }
    symbols = []
    for path, src in sources.items():
        tree = parser.parse(src, language="python")
        symbols.extend(extractor.extract_from_tree(tree, path, src, language="python"))

    edges = builder.build_from_symbols(symbols)
    assert len(edges) == 1
    assert edges[0].source_module == "app"
    assert edges[0].target_module == "db"


def test_edges_are_deterministically_ordered(builder: DependencyGraphBuilder) -> None:
    """Edges are sorted by (source_file, target_module)."""
    sources = {
        "z.py": "x = 1",
        "b.py": "import z",
        "a.py": "import z\nimport b",
    }
    edges = builder.build_from_sources(sources)
    keys = [(e.source_file, e.target_module) for e in edges]
    assert keys == sorted(keys)


def test_dependency_edge_to_dict_is_json_primitive() -> None:
    """DependencyEdge.to_dict emits only JSON-primitive values."""
    edge = DependencyEdge(
        source_module="a",
        target_module="b",
        source_file="a.py",
        target_file="b.py",
        import_type="from_import",
        imported_names=["func"],
        is_circular=False,
    )
    assert edge.resolved is True
    payload = edge.to_dict()
    assert payload["source_module"] == "a"
    assert payload["resolved"] is True
    assert payload["imported_names"] == ["func"]
    for value in payload.values():
        assert isinstance(value, str | int | bool | list | type(None))


def test_dependency_edge_repr() -> None:
    """Test the __repr__ of DependencyEdge."""
    edge1 = DependencyEdge("a", "b", "a.py", "import", [])
    assert "DependencyEdge(a -> b [import])" in repr(edge1)
    edge2 = DependencyEdge("a", "b", "a.py", "import", [], is_circular=True)
    assert "[circular]" in repr(edge2)


def test_build_from_files_errors(builder: DependencyGraphBuilder) -> None:
    """Test build_from_files with unsupported and unreadable files."""
    import contextlib
    import os
    from pathlib import Path

    d = Path("tests/unit/_tmp_test_errors")
    d.mkdir(parents=True, exist_ok=True)
    unsupported = d / "test.txt"
    unsupported.write_text("hello")
    unreadable = d / "unreadable.py"
    unreadable.write_text("x = 1")
    os.chmod(unreadable, 0o000)  # make unreadable

    try:
        edges = builder.build_from_files([unsupported, unreadable])
        assert len(edges) == 0
    finally:
        os.chmod(unreadable, 0o666)
        unreadable.unlink(missing_ok=True)
        unsupported.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            d.rmdir()


def test_module_key_none(builder: DependencyGraphBuilder) -> None:
    """Test symbols with no import source are skipped."""
    from src.reporag.ingestion.symbol_extractor import Symbol

    sym = Symbol(name="x", type="import", file_path="a.py", start_line=1, end_line=1)
    edges = builder.build_from_symbols([sym])
    assert len(edges) == 0

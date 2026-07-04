"""Unit tests for the DependencyGraphBuilder (Issue 10).

Covers the acceptance criteria -- absolute and relative import resolution,
star imports with a warning, circular import detection, and edge metadata
(source, target, import type, imported names) -- plus edge cases around
plain ``import x, y`` statements, aliasing, and unresolved external
imports.
"""

from __future__ import annotations

import pytest

from src.reporag.graph.dependency_graph import (
    CircularImportChain,
    DependencyEdge,
    DependencyGraphBuilder,
    DependencyGraphResult,
)
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def builder() -> DependencyGraphBuilder:
    """Provide a fresh DependencyGraphBuilder for each test."""
    return DependencyGraphBuilder()


def _edge(edges: list[DependencyEdge], source: str, target: str) -> DependencyEdge:
    """Return the single edge matching *source* -> *target* (fails if absent)."""
    matches = [e for e in edges if e.source == source and e.target == target]
    assert matches, f"no edge {source} -> {target} in {[str(e) for e in edges]}"
    assert len(matches) == 1, f"expected one {source} -> {target}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Absolute imports
# ---------------------------------------------------------------------------


def test_plain_import_resolves_to_project_file(builder: DependencyGraphBuilder) -> None:
    """``import db`` resolves to db.py when it's part of the project."""
    result = builder.build_from_sources({"app.py": "import db\n", "db.py": "x = 1\n"})
    edge = _edge(result.edges, "app.py", "db.py")
    assert edge.import_type == "import"
    assert edge.resolved is True
    assert edge.source_module == "app"
    assert edge.target_module == "db"
    assert edge.line == 1
    assert edge.is_relative is False


def test_dotted_import_resolves_nested_module(builder: DependencyGraphBuilder) -> None:
    """``import pkg.sub`` resolves to pkg/sub.py."""
    result = builder.build_from_sources(
        {
            "app.py": "import pkg.sub\n",
            "pkg/__init__.py": "",
            "pkg/sub.py": "x = 1\n",
        }
    )
    edge = _edge(result.edges, "app.py", "pkg/sub.py")
    assert edge.target_module == "pkg.sub"


def test_import_with_alias_records_alias(builder: DependencyGraphBuilder) -> None:
    """``import numpy as np`` records the (module, alias) pair."""
    result = builder.build_from_sources({"app.py": "import numpy as np\n"})
    edge = _edge(result.edges, "app.py", "numpy")
    assert edge.resolved is False
    assert edge.imported_names == [("numpy", "np")]


def test_comma_separated_import_produces_two_edges(
    builder: DependencyGraphBuilder,
) -> None:
    """``import a, b`` names two independent dependencies -- two edges."""
    result = builder.build_from_sources(
        {"app.py": "import a, b\n", "a.py": "", "b.py": ""}
    )
    assert _edge(result.edges, "app.py", "a.py")
    assert _edge(result.edges, "app.py", "b.py")


def test_from_import_groups_names_on_one_edge(builder: DependencyGraphBuilder) -> None:
    """``from x import a, b`` is one dependency edge carrying both names."""
    result = builder.build_from_sources(
        {"app.py": "from utils import a, b\n", "utils.py": ""}
    )
    edge = _edge(result.edges, "app.py", "utils.py")
    assert edge.import_type == "from_import"
    assert edge.imported_names == [("a", None), ("b", None)]


def test_from_import_with_alias(builder: DependencyGraphBuilder) -> None:
    """``from os import path as p`` records the (name, alias) pair."""
    result = builder.build_from_sources({"app.py": "from os import path as p\n"})
    edge = _edge(result.edges, "app.py", "os")
    assert edge.imported_names == [("path", "p")]


# ---------------------------------------------------------------------------
# Relative imports
# ---------------------------------------------------------------------------


def test_single_dot_relative_import_resolves_sibling(
    builder: DependencyGraphBuilder,
) -> None:
    """``from .sibling import Z`` resolves against the importing file's package."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .sibling import Z\n",
            "pkg/sibling.py": "Z = 1\n",
        }
    )
    edge = _edge(result.edges, "pkg/a.py", "pkg/sibling.py")
    assert edge.is_relative is True
    assert edge.relative_level == 1
    assert edge.imported_names == [("Z", None)]


def test_bare_dot_import_resolves_submodule_when_it_exists(
    builder: DependencyGraphBuilder,
) -> None:
    """``from . import sibling`` resolves to sibling.py when it exists.

    Real Python import semantics try the submodule first: ``from . import
    sibling`` binds the *module* ``pkg.sibling``, not an attribute of
    ``pkg/__init__.py``, whenever ``pkg/sibling.py`` exists. Bare ``.`` has
    no meaningful module to resolve on its own, so the single imported
    name is appended to it before resolution.
    """
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from . import sibling\n",
            "pkg/sibling.py": "x = 1\n",
        }
    )
    edge = _edge(result.edges, "pkg/a.py", "pkg/sibling.py")
    assert edge.is_relative is True
    assert edge.relative_level == 1
    assert edge.target_module == ".sibling"


def test_bare_dot_import_falls_back_to_unresolved_when_no_submodule(
    builder: DependencyGraphBuilder,
) -> None:
    """``from . import name`` with no matching submodule is left unresolved.

    Without a ``sibling.py`` file, ``name`` is presumably an attribute
    defined directly in ``pkg/__init__.py`` rather than a submodule; this
    builder doesn't inspect ``__init__.py`` contents, so it's reported as
    an unresolved (external-looking) import rather than guessed at.
    """
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from . import something\n",
        }
    )
    edge = _edge(result.edges, "pkg/a.py", ".something")
    assert edge.resolved is False


def test_double_dot_relative_import_ascends_a_level(
    builder: DependencyGraphBuilder,
) -> None:
    """``from ..sibling_pkg import x`` ascends one package level before resolving."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/a.py": "from ..sibling_pkg import x\n",
            "pkg/sibling_pkg/__init__.py": "",
        }
    )
    edge = _edge(result.edges, "pkg/sub/a.py", "pkg/sibling_pkg/__init__.py")
    assert edge.relative_level == 2


# ---------------------------------------------------------------------------
# Star imports
# ---------------------------------------------------------------------------


def test_star_import_is_flagged_and_resolved(builder: DependencyGraphBuilder) -> None:
    """``from utils import *`` resolves as an edge and raises a warning."""
    result = builder.build_from_sources(
        {"app.py": "from utils import *\n", "utils.py": ""}
    )
    edge = _edge(result.edges, "app.py", "utils.py")
    assert edge.is_wildcard is True
    assert edge.imported_names == [("*", None)]
    assert any("star import" in w for w in result.warnings)
    assert any("utils" in w for w in result.warnings)


def test_relative_star_import_is_flagged(builder: DependencyGraphBuilder) -> None:
    """A relative star import is both flagged and marked relative."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .utils import *\n",
            "pkg/utils.py": "",
        }
    )
    edge = _edge(result.edges, "pkg/a.py", "pkg/utils.py")
    assert edge.is_wildcard is True
    assert edge.is_relative is True
    assert result.warnings


# ---------------------------------------------------------------------------
# Circular imports
# ---------------------------------------------------------------------------


def test_direct_circular_import_detected(builder: DependencyGraphBuilder) -> None:
    """Two modules importing each other form a length-2 cycle."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .b import foo\n",
            "pkg/b.py": "from .a import bar\n",
        }
    )
    assert result.has_cycles
    assert len(result.cycles) == 1
    cycle = result.cycles[0]
    assert isinstance(cycle, CircularImportChain)
    assert set(cycle.files[:-1]) == {"pkg/a.py", "pkg/b.py"}
    assert cycle.files[0] == cycle.files[-1]


def test_three_module_circular_chain_detected(builder: DependencyGraphBuilder) -> None:
    """A -> B -> C -> A is detected as a single 3-node cycle."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .b import x\n",
            "pkg/b.py": "from .c import y\n",
            "pkg/c.py": "from .a import z\n",
        }
    )
    assert len(result.cycles) == 1
    assert set(result.cycles[0].files[:-1]) == {"pkg/a.py", "pkg/b.py", "pkg/c.py"}


def test_self_import_is_a_length_one_cycle(builder: DependencyGraphBuilder) -> None:
    """A module importing itself is reported as a self-cycle."""
    result = builder.build_from_sources(
        {"pkg/__init__.py": "", "pkg/a.py": "from pkg.a import thing\n"}
    )
    assert len(result.cycles) == 1
    assert result.cycles[0].files == ["pkg/a.py", "pkg/a.py"]


def test_no_cycle_for_acyclic_imports(builder: DependencyGraphBuilder) -> None:
    """A strictly acyclic import chain reports no cycles."""
    result = builder.build_from_sources(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .b import x\n",
            "pkg/b.py": "from .c import y\n",
            "pkg/c.py": "z = 1\n",
        }
    )
    assert result.has_cycles is False
    assert result.cycles == []


def test_unresolved_imports_do_not_create_false_cycles(
    builder: DependencyGraphBuilder,
) -> None:
    """External imports never appear in a reported cycle chain."""
    result = builder.build_from_sources({"app.py": "import os\nimport sys\n"})
    assert result.cycles == []


# ---------------------------------------------------------------------------
# Unresolved / external imports
# ---------------------------------------------------------------------------


def test_external_import_marked_unresolved_by_default(
    builder: DependencyGraphBuilder,
) -> None:
    """A stdlib import with no matching project file is still emitted."""
    result = builder.build_from_sources({"app.py": "import hashlib\n"})
    edge = _edge(result.edges, "app.py", "hashlib")
    assert edge.resolved is False


def test_include_external_false_drops_unresolved_edges(
    builder: DependencyGraphBuilder,
) -> None:
    """``include_external=False`` returns only intra-project edges."""
    result = builder.build_from_sources(
        {"app.py": "import hashlib\nimport db\n", "db.py": ""},
        include_external=False,
    )
    targets = {e.target for e in result.edges}
    assert "hashlib" not in targets
    assert "db.py" in targets


# ---------------------------------------------------------------------------
# Result contract / misc
# ---------------------------------------------------------------------------


def test_result_type_is_dependency_graph_result(
    builder: DependencyGraphBuilder,
) -> None:
    """The builder always returns a DependencyGraphResult."""
    result = builder.build_from_sources({"app.py": "import os\n"})
    assert isinstance(result, DependencyGraphResult)


def test_empty_source_produces_no_edges(builder: DependencyGraphBuilder) -> None:
    """A file with no imports produces no edges."""
    result = builder.build_from_sources({"app.py": "x = 1\n"})
    assert result.edges == []
    assert result.cycles == []


# ---------------------------------------------------------------------------
# build_from_symbols -- the Issue 7 dependency
# ---------------------------------------------------------------------------


def _symbols_for(sources: dict[str, str]) -> dict[str, list]:
    """Run Issue 7's SymbolExtractor over *sources*, keyed by file path."""
    extractor = SymbolExtractor()
    return {
        path: extractor.extract_from_source(src, language="python", file_path=path)
        for path, src in sources.items()
    }


def test_build_from_symbols_matches_ast_path_for_unaliased_imports(
    builder: DependencyGraphBuilder,
) -> None:
    """Plain and non-aliased from-imports resolve identically via Symbol input."""
    sources = {
        "pkg/__init__.py": "",
        "pkg/a.py": "from .b import foo\nimport os\n",
        "pkg/b.py": "from .a import bar\n",
    }
    via_symbols = builder.build_from_symbols(_symbols_for(sources))
    via_ast = builder.build_from_sources(sources)

    def _as_set(edges: list[DependencyEdge]) -> set[tuple]:
        return {
            (e.source, e.target, e.import_type, e.resolved, tuple(e.imported_names))
            for e in edges
        }

    assert _as_set(via_symbols.edges) == _as_set(via_ast.edges)
    assert len(via_symbols.cycles) == len(via_ast.cycles) == 1


def test_build_from_symbols_grouping_recovers_from_import_names(
    builder: DependencyGraphBuilder,
) -> None:
    """``from utils import a, b`` groups back into one edge via Symbol input."""
    sources = {"app.py": "from utils import a, b\n", "utils.py": ""}
    result = builder.build_from_symbols(_symbols_for(sources))
    edge = _edge(result.edges, "app.py", "utils.py")
    assert set(edge.imported_names) == {("a", None), ("b", None)}


def test_build_from_symbols_resolves_alias_to_from_import_when_module_matches(
    builder: DependencyGraphBuilder,
) -> None:
    """An aliased from-import resolves correctly when the module is a project file.

    ``from pkg.a import foo as f`` is ambiguous from the Symbol alone, but
    since ``pkg.a`` (not ``pkg.a.foo``) is a real project file, the
    disambiguation heuristic correctly falls back to the from-import
    reading.
    """
    sources = {
        "pkg/__init__.py": "",
        "pkg/a.py": "",
        "pkg/sub.py": "from pkg.a import foo as f\n",
    }
    result = builder.build_from_symbols(_symbols_for(sources))
    edge = _edge(result.edges, "pkg/sub.py", "pkg/a.py")
    assert edge.import_type == "from_import"
    assert edge.imported_names == [("foo", "f")]


def test_build_from_symbols_wildcard_still_flagged(
    builder: DependencyGraphBuilder,
) -> None:
    """A wildcard Symbol still produces a resolved edge plus a warning."""
    sources = {"app.py": "from utils import *\n", "utils.py": ""}
    result = builder.build_from_symbols(_symbols_for(sources))
    edge = _edge(result.edges, "app.py", "utils.py")
    assert edge.is_wildcard is True
    assert result.warnings


def test_build_from_symbols_documents_aliased_external_ambiguity(
    builder: DependencyGraphBuilder,
) -> None:
    """Known limitation: an aliased import of an external module can't be
    disambiguated from Symbol data alone, so the "plain import" reading is
    kept as a best-effort fallback (see build_from_symbols docstring).
    """
    sources = {"app.py": "from os import path as p\n"}
    result = builder.build_from_symbols(_symbols_for(sources))
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.resolved is False
    # Best-effort fallback reports the combined dotted path, not "os".
    assert edge.target == "os.path"
    assert edge.imported_names == [("os.path", "p")]


def test_build_from_symbols_skips_non_import_symbols(
    builder: DependencyGraphBuilder,
) -> None:
    """Function/class symbols from Issue 7 output are ignored, not errored on."""
    sources = {"app.py": "import os\n\ndef helper():\n    return 1\n"}
    result = builder.build_from_symbols(_symbols_for(sources))
    assert len(result.edges) == 1
    assert result.edges[0].target == "os"

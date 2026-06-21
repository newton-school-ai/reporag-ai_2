"""Unit tests for call graph builder.

Tests cover the call_graph module defined in src/reporag/graph/call_graph.py.
Currently validates the module stub; test bodies will be implemented in Issue 9.
"""

import importlib

import pytest


class TestCallGraphModuleExists:
    """Verify the call_graph module is importable and properly documented."""

    def test_call_graph_module_importable(self) -> None:
        """Call graph module can be imported without error."""
        mod = importlib.import_module("reporag.graph.call_graph")
        assert mod is not None

    def test_call_graph_module_has_docstring(self) -> None:
        """Call graph module has a module-level docstring."""
        mod = importlib.import_module("reporag.graph.call_graph")
        assert mod.__doc__ is not None
        assert "call" in mod.__doc__.lower()


@pytest.mark.skip(reason="Not yet implemented -- Issue 9")
class TestCallEdgeExtraction:
    """Call graph builder should extract caller-callee edges from AST."""

    def test_direct_function_call(self) -> None:
        """Detects a direct function call edge: caller -> callee."""

    def test_method_call(self) -> None:
        """Detects self.method() call edges within a class."""

    def test_chained_calls(self) -> None:
        """Detects chained method calls like obj.method1().method2()."""

    def test_constructor_call(self) -> None:
        """Detects constructor calls as edges to __init__."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 9")
class TestCallEdgeMetadata:
    """Call edges should carry metadata about the call site."""

    def test_edge_has_caller_and_callee(self) -> None:
        """Each edge records both caller and callee symbol names."""

    def test_edge_has_call_site_line(self) -> None:
        """Each edge records the line number of the call site."""

    def test_edge_has_file_path(self) -> None:
        """Each edge records the source file path."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 9")
class TestCrossFileResolution:
    """Call graph should resolve calls across files via imports."""

    def test_resolve_imported_function(self) -> None:
        """Resolves a call to a function imported from another module."""

    def test_unresolved_call_flagged(self) -> None:
        """Calls that cannot be resolved are flagged, not silently dropped."""

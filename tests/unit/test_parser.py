"""Unit tests for tree-sitter AST parser.

Tests cover the parser module interface defined in src/reporag/ingestion/parser.py.
Currently validates the module stub; test bodies will be implemented in Issue 6.
"""

import importlib

import pytest


class TestParserModuleExists:
    """Verify the parser module is importable and properly documented."""

    def test_parser_module_importable(self) -> None:
        """Parser module can be imported without error."""
        mod = importlib.import_module("reporag.ingestion.parser")
        assert mod is not None

    def test_parser_module_has_docstring(self) -> None:
        """Parser module has a module-level docstring."""
        mod = importlib.import_module("reporag.ingestion.parser")
        assert mod.__doc__ is not None
        assert "tree-sitter" in mod.__doc__.lower()


@pytest.mark.skip(reason="Not yet implemented -- Issue 6")
class TestParseEmptyFile:
    """Parser should handle empty source files gracefully."""

    def test_parse_empty_string(self) -> None:
        """Parsing an empty string returns an empty AST."""

    def test_parse_whitespace_only(self) -> None:
        """Parsing whitespace-only source returns an empty AST."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 6")
class TestParseFunctions:
    """Parser should extract function definitions from AST."""

    def test_parse_single_function(self) -> None:
        """Parsing a single function definition produces correct nodes."""

    def test_parse_async_function(self) -> None:
        """Parsing an async function definition produces correct nodes."""

    def test_parse_function_with_decorators(self) -> None:
        """Parsing decorated functions includes decorator metadata."""

    def test_parse_nested_functions(self) -> None:
        """Parsing nested function definitions captures hierarchy."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 6")
class TestParseClasses:
    """Parser should extract class definitions from AST."""

    def test_parse_class_with_methods(self) -> None:
        """Parsing a class extracts methods with parent class metadata."""

    def test_parse_nested_classes(self) -> None:
        """Parsing nested classes captures hierarchy."""

    def test_parse_class_with_inheritance(self) -> None:
        """Parsing class inheritance captures base classes."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 6")
class TestParseErrorHandling:
    """Parser should handle syntax errors gracefully."""

    def test_parse_syntax_error_returns_partial_ast(self) -> None:
        """Parsing a file with syntax errors returns partial AST."""

    def test_parse_flags_errors(self) -> None:
        """Parsing errors are flagged in the result metadata."""


@pytest.mark.skip(reason="Not yet implemented -- Issue 6")
class TestLanguageAgnosticInterface:
    """Parser should support multiple languages via a unified interface."""

    def test_parse_python(self) -> None:
        """Parser handles Python source files."""

    def test_parse_javascript(self) -> None:
        """Parser handles JavaScript source files."""

    def test_unsupported_language_raises(self) -> None:
        """Parser raises ValueError for unsupported languages."""

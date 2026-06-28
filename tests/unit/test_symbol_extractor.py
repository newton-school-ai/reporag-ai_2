"""Unit tests for src/reporag/ingestion/symbol_extractor.py (Issue 7).

Acceptance Criteria covered:
- [x] Extracts functions with: name, signature, docstring, decorators, line range
- [x] Extracts classes with: name, bases, methods, docstring, line range
- [x] Extracts imports: import X, from X import Y, from X import *
- [x] Handles nested functions, async functions, property decorators
- [x] Returns structured Symbol dataclass objects
- [x] Unit tests cover all symbol types
"""

from __future__ import annotations

import pytest

from src.reporag.ingestion.symbol_extractor import (
    ASYNC_FUNCTION,
    ASYNC_METHOD,
    CLASS,
    CLASS_METHOD,
    FUNCTION,
    IMPORT,
    METHOD,
    PROPERTY,
    STATIC_METHOD,
    Symbol,
    SymbolExtractor,
)

# ---------------------------------------------------------------------------
# Shared extractor fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor() -> SymbolExtractor:
    """One SymbolExtractor reused across the module (grammars loaded once)."""
    return SymbolExtractor()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def extract(source: str, extractor: SymbolExtractor) -> list[Symbol]:
    return extractor.extract_from_source(source, language="python")


def by_type(symbols: list[Symbol], sym_type: str) -> list[Symbol]:
    return [s for s in symbols if s.type == sym_type]


def by_name(symbols: list[Symbol], name: str) -> Symbol:
    for s in symbols:
        if s.name == name:
            return s
    raise KeyError(f"No symbol named {name!r} in {symbols}")


# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

SIMPLE_FUNCTION = """\
def hello(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}"
"""

ASYNC_FUNCTION_SRC = """\
import asyncio

async def fetch(url: str) -> dict:
    \"\"\"Fetch from url.\"\"\"
    await asyncio.sleep(0)
    return {}
"""

CLASS_SRC = """\
class Animal:
    \"\"\"Base animal class.\"\"\"

    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        return f"{self.name} speaks"
"""

CLASS_WITH_BASES = """\
class Dog(Animal, Serializable):
    \"\"\"A dog.\"\"\"

    def speak(self) -> str:
        return "Woof"
"""

DECORATED_FUNCTION = """\
import functools

def my_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    \"\"\"Greet someone.\"\"\"
    return f"Hi {name}"
"""

STATIC_AND_CLASS_METHODS = """\
class Utils:
    @staticmethod
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    @classmethod
    def create(cls) -> \"Utils\":
        \"\"\"Create instance.\"\"\"
        return cls()

    @property
    def value(self) -> int:
        \"\"\"The value.\"\"\"
        return 42
"""

ASYNC_METHOD_SRC = """\
class Service:
    async def connect(self) -> None:
        \"\"\"Connect to service.\"\"\"
        pass

    async def disconnect(self) -> None:
        pass

    def sync_op(self) -> None:
        pass
"""

NESTED_FUNCTION_SRC = """\
def outer(x: int) -> int:
    \"\"\"Outer function.\"\"\"

    def inner(y: int) -> int:
        \"\"\"Inner function.\"\"\"
        return x + y

    def another_inner() -> None:
        pass

    return inner(x)
"""

DEEPLY_NESTED_SRC = """\
def level_one():
    def level_two():
        def level_three():
            pass
"""

NESTED_IN_METHOD_SRC = """\
class Builder:
    def build(self) -> None:
        \"\"\"Build with helpers.\"\"\"

        def _validate(value):
            return value is not None

        def _transform(value):
            return str(value)
"""

NESTED_CLASS = """\
class Outer:
    \"\"\"Outer class.\"\"\"

    class Inner:
        \"\"\"Inner class.\"\"\"

        def inner_method(self) -> None:
            pass

    def outer_method(self) -> None:
        pass
"""

IMPORT_SRC = """\
import os
import sys
from pathlib import Path, PurePath
from typing import Optional, List
from os import *
from collections import OrderedDict as OD
"""

MIXED_SRC = """\
import os
from pathlib import Path

class Config:
    \"\"\"Config class.\"\"\"

    def load(self, path: str) -> dict:
        \"\"\"Load config from path.\"\"\"
        pass

    @staticmethod
    def default() -> \"Config\":
        return Config()

def run(config: Config) -> None:
    \"\"\"Run the app.\"\"\"
    pass

async def start() -> None:
    \"\"\"Start async.\"\"\"
    pass
"""

NO_DOCSTRING_SRC = """\
def no_doc(x):
    return x * 2

class NoDocs:
    def method(self):
        pass
"""

EMPTY_SRC = ""

SYNTAX_ERROR_SRC = """\
def broken(
    return 42
"""

MULTI_DECORATOR_SRC = """\
class View:
    @classmethod
    @some_other_decorator
    def dispatch(cls, request):
        pass
"""


# ---------------------------------------------------------------------------
# 1. Returns list of Symbol objects
# ---------------------------------------------------------------------------


def test_returns_list_of_symbol_objects(extractor: SymbolExtractor) -> None:
    """extract_from_source always returns a list."""
    result = extract(SIMPLE_FUNCTION, extractor)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(s, Symbol) for s in result)


# ---------------------------------------------------------------------------
# 2. Simple function extraction
# ---------------------------------------------------------------------------


def test_simple_function_name(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    funcs = by_type(symbols, FUNCTION)
    assert len(funcs) == 1
    assert funcs[0].name == "hello"


def test_simple_function_type(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    assert symbols[0].type == FUNCTION


def test_simple_function_signature(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    assert "(name: str)" in fn.signature
    assert "-> str" in fn.signature


def test_simple_function_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    assert fn.docstring == "Say hello."


def test_simple_function_line_range(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    assert fn.start_line == 1
    assert fn.end_line == 3


def test_simple_function_return_type(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    assert "str" in fn.return_type_hint


def test_simple_function_no_parent_class(extractor: SymbolExtractor) -> None:
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    assert fn.parent_class == ""


# ---------------------------------------------------------------------------
# 3. Async function
# ---------------------------------------------------------------------------


def test_async_function_type(extractor: SymbolExtractor) -> None:
    symbols = extract(ASYNC_FUNCTION_SRC, extractor)
    async_funcs = by_type(symbols, ASYNC_FUNCTION)
    assert len(async_funcs) == 1


def test_async_function_name(extractor: SymbolExtractor) -> None:
    symbols = extract(ASYNC_FUNCTION_SRC, extractor)
    fn = by_type(symbols, ASYNC_FUNCTION)[0]
    assert fn.name == "fetch"


def test_async_function_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(ASYNC_FUNCTION_SRC, extractor)
    fn = by_type(symbols, ASYNC_FUNCTION)[0]
    assert fn.docstring == "Fetch from url."


# ---------------------------------------------------------------------------
# 4. Class extraction
# ---------------------------------------------------------------------------


def test_class_is_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    classes = by_type(symbols, CLASS)
    assert len(classes) == 1


def test_class_name(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    cls = by_type(symbols, CLASS)[0]
    assert cls.name == "Animal"


def test_class_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    cls = by_type(symbols, CLASS)[0]
    assert cls.docstring == "Base animal class."


def test_class_line_range(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    cls = by_type(symbols, CLASS)[0]
    assert cls.start_line == 1
    assert cls.end_line >= 8


def test_class_methods_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    methods = by_type(symbols, METHOD)
    method_names = {m.name for m in methods}
    assert "__init__" in method_names
    assert "speak" in method_names


def test_method_parent_class(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    methods = by_type(symbols, METHOD)
    assert all(m.parent_class == "Animal" for m in methods)


# ---------------------------------------------------------------------------
# 5. Class with base classes
# ---------------------------------------------------------------------------


def test_class_with_bases_signature(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_WITH_BASES, extractor)
    cls = by_type(symbols, CLASS)[0]
    assert "Animal" in cls.signature
    assert "Serializable" in cls.signature


# ---------------------------------------------------------------------------
# 6. Decorated function
# ---------------------------------------------------------------------------


def test_decorated_function_type(extractor: SymbolExtractor) -> None:
    symbols = extract(DECORATED_FUNCTION, extractor)
    funcs = by_type(symbols, FUNCTION)
    greet = next((f for f in funcs if f.name == "greet"), None)
    assert greet is not None


def test_decorated_function_decorators(extractor: SymbolExtractor) -> None:
    symbols = extract(DECORATED_FUNCTION, extractor)
    greet = by_name(symbols, "greet")
    assert "my_decorator" in greet.decorators


def test_decorated_function_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(DECORATED_FUNCTION, extractor)
    greet = by_name(symbols, "greet")
    assert greet.docstring == "Greet someone."


# ---------------------------------------------------------------------------
# 7. Static / class methods / property
# ---------------------------------------------------------------------------


def test_static_method_type(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    statics = by_type(symbols, STATIC_METHOD)
    assert any(s.name == "add" for s in statics)


def test_static_method_decorator(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    add = by_name(symbols, "add")
    assert "staticmethod" in add.decorators


def test_class_method_type(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    cms = by_type(symbols, CLASS_METHOD)
    assert any(s.name == "create" for s in cms)


def test_class_method_decorator(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    create = by_name(symbols, "create")
    assert "classmethod" in create.decorators


def test_property_type(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    props = by_type(symbols, PROPERTY)
    assert any(s.name == "value" for s in props)


def test_property_decorator(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    val = by_name(symbols, "value")
    assert "property" in val.decorators


def test_static_method_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(STATIC_AND_CLASS_METHODS, extractor)
    add = by_name(symbols, "add")
    assert add.docstring == "Add two numbers."


# ---------------------------------------------------------------------------
# 8. Nested class
# ---------------------------------------------------------------------------


def test_nested_class_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(NESTED_CLASS, extractor)
    classes = by_type(symbols, CLASS)
    class_names = {c.name for c in classes}
    assert "Outer" in class_names
    assert "Inner" in class_names


def test_nested_class_parent_set(extractor: SymbolExtractor) -> None:
    symbols = extract(NESTED_CLASS, extractor)
    inner = by_name(symbols, "Inner")
    assert inner.parent_class == "Outer"


def test_nested_class_method_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(NESTED_CLASS, extractor)
    inner_method = by_name(symbols, "inner_method")
    assert inner_method.parent_class == "Inner"


# ---------------------------------------------------------------------------
# 9. Imports
# ---------------------------------------------------------------------------


def test_simple_import_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    import_names = {s.name for s in imports}
    assert "os" in import_names
    assert "sys" in import_names


def test_from_import_extracted(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    # from pathlib import Path, PurePath
    pathlib_import = next((s for s in imports if s.module == "pathlib"), None)
    assert pathlib_import is not None
    assert "Path" in pathlib_import.names
    assert "PurePath" in pathlib_import.names


def test_wildcard_import(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    wildcard = next((s for s in imports if s.module == "os" and "*" in s.names), None)
    assert wildcard is not None


def test_aliased_import(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    aliased = next((s for s in imports if "OD" in s.names), None)
    assert aliased is not None


def test_import_line_range(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    os_import = next((s for s in imports if s.name == "os" and s.module == "os"), None)
    assert os_import is not None
    assert os_import.start_line == 1
    assert os_import.end_line == 1


def test_import_module_field(extractor: SymbolExtractor) -> None:
    symbols = extract(IMPORT_SRC, extractor)
    imports = by_type(symbols, IMPORT)
    typing_import = next((s for s in imports if s.module == "typing"), None)
    assert typing_import is not None
    assert typing_import.module == "typing"


# ---------------------------------------------------------------------------
# 10. Mixed source (classes + functions + imports)
# ---------------------------------------------------------------------------


def test_mixed_source_symbol_types(extractor: SymbolExtractor) -> None:
    symbols = extract(MIXED_SRC, extractor)
    types = {s.type for s in symbols}
    assert IMPORT in types
    assert CLASS in types
    assert FUNCTION in types
    assert ASYNC_FUNCTION in types
    assert METHOD in types
    assert STATIC_METHOD in types


def test_mixed_source_order(extractor: SymbolExtractor) -> None:
    """Symbols are returned in source order (imports first, then class, then functions)."""
    symbols = extract(MIXED_SRC, extractor)
    import_lines = [s.start_line for s in symbols if s.type == IMPORT]
    class_lines = [s.start_line for s in symbols if s.type == CLASS]
    func_lines = [s.start_line for s in symbols if s.type in (FUNCTION, ASYNC_FUNCTION)]

    # All imports come before the class which comes before top-level functions
    assert max(import_lines) < min(class_lines)
    assert min(func_lines) > min(class_lines)


# ---------------------------------------------------------------------------
# 11. No docstring
# ---------------------------------------------------------------------------


def test_function_without_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(NO_DOCSTRING_SRC, extractor)
    fn = by_name(symbols, "no_doc")
    assert fn.docstring == ""


def test_class_without_docstring(extractor: SymbolExtractor) -> None:
    symbols = extract(NO_DOCSTRING_SRC, extractor)
    cls = by_name(symbols, "NoDocs")
    assert cls.docstring == ""


# ---------------------------------------------------------------------------
# 12. Empty source
# ---------------------------------------------------------------------------


def test_empty_source_returns_empty_list(extractor: SymbolExtractor) -> None:
    symbols = extract(EMPTY_SRC, extractor)
    assert symbols == []


# ---------------------------------------------------------------------------
# 13. Syntax error source (partial AST tolerance)
# ---------------------------------------------------------------------------


def test_syntax_error_source_does_not_raise(extractor: SymbolExtractor) -> None:
    """extract_from_source must not raise on broken source."""
    result = extract(SYNTAX_ERROR_SRC, extractor)
    assert isinstance(result, list)  # May be empty or partial -- must not crash


# ---------------------------------------------------------------------------
# 14. Multi-decorator method
# ---------------------------------------------------------------------------


def test_multi_decorator_method(extractor: SymbolExtractor) -> None:
    symbols = extract(MULTI_DECORATOR_SRC, extractor)
    dispatch = by_name(symbols, "dispatch")
    assert "classmethod" in dispatch.decorators
    assert "some_other_decorator" in dispatch.decorators


# ---------------------------------------------------------------------------
# 15. Symbol dataclass fields completeness
# ---------------------------------------------------------------------------


def test_symbol_has_all_required_fields(extractor: SymbolExtractor) -> None:
    """Every Symbol has all required fields with correct types."""
    symbols = extract(MIXED_SRC, extractor)
    for sym in symbols:
        assert isinstance(sym.name, str) and sym.name
        assert isinstance(sym.type, str) and sym.type
        assert isinstance(sym.file_path, str)
        assert isinstance(sym.start_line, int) and sym.start_line >= 1
        assert isinstance(sym.end_line, int) and sym.end_line >= sym.start_line
        assert isinstance(sym.signature, str)
        assert isinstance(sym.docstring, str)
        assert isinstance(sym.decorators, list)
        assert isinstance(sym.parent_class, str)
        assert isinstance(sym.return_type_hint, str)
        assert isinstance(sym.module, str)
        assert isinstance(sym.names, list)


# ---------------------------------------------------------------------------
# 16. extract_from_file (disk)
# ---------------------------------------------------------------------------


def test_extract_from_file(
    tmp_path: pytest.TempPathFactory, extractor: SymbolExtractor
) -> None:
    """extract_from_file reads from disk and returns the same symbols."""
    src_file = tmp_path / "module.py"
    src_file.write_text(SIMPLE_FUNCTION, encoding="utf-8")

    symbols = extractor.extract_from_file(str(src_file), language="python")
    assert len(symbols) >= 1
    assert symbols[0].name == "hello"
    assert symbols[0].file_path == str(src_file)


def test_extract_from_file_path_in_symbol(
    tmp_path: pytest.TempPathFactory, extractor: SymbolExtractor
) -> None:
    """file_path in each Symbol matches the file that was parsed."""
    src_file = tmp_path / "greet.py"
    src_file.write_text(SIMPLE_FUNCTION, encoding="utf-8")
    symbols = extractor.extract_from_file(str(src_file), language="python")
    assert all(s.file_path == str(src_file) for s in symbols)


# ---------------------------------------------------------------------------
# 17. Unsupported language returns empty list (not crash)
# ---------------------------------------------------------------------------


def test_unsupported_language_returns_empty(extractor: SymbolExtractor) -> None:
    """Non-Python languages return [] gracefully (JS extraction not yet implemented)."""
    result = extractor.extract_from_source("function hello() {}", language="javascript")
    assert result == []


# ---------------------------------------------------------------------------
# 18. Reuse extractor across calls
# ---------------------------------------------------------------------------


def test_extractor_reusable(extractor: SymbolExtractor) -> None:
    """Same extractor instance produces correct results across multiple calls."""
    r1 = extract(SIMPLE_FUNCTION, extractor)
    r2 = extract(CLASS_SRC, extractor)
    r3 = extract(IMPORT_SRC, extractor)

    assert r1[0].name == "hello"
    assert by_type(r2, CLASS)[0].name == "Animal"
    assert any(s.type == IMPORT for s in r3)


# ---------------------------------------------------------------------------
# 19. Method signature and return type
# ---------------------------------------------------------------------------


def test_method_signature_includes_params(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    init = by_name(symbols, "__init__")
    assert "name: str" in init.signature


def test_method_return_type(extractor: SymbolExtractor) -> None:
    symbols = extract(CLASS_SRC, extractor)
    init = by_name(symbols, "__init__")
    assert "None" in init.return_type_hint


# ---------------------------------------------------------------------------
# 20. Symbol __repr__
# ---------------------------------------------------------------------------


def test_symbol_repr(extractor: SymbolExtractor) -> None:
    """Symbol repr is human-readable."""
    symbols = extract(SIMPLE_FUNCTION, extractor)
    fn = by_type(symbols, FUNCTION)[0]
    r = repr(fn)
    assert "function" in r
    assert "hello" in r
    assert "[" in r and "]" in r


# ---------------------------------------------------------------------------
# 21. Async method type (async def inside a class -> ASYNC_METHOD)
# ---------------------------------------------------------------------------


def test_async_method_type(extractor: SymbolExtractor) -> None:
    """async def inside a class produces ASYNC_METHOD, not METHOD."""
    symbols = extract(ASYNC_METHOD_SRC, extractor)
    connect = by_name(symbols, "connect")
    assert connect.type == ASYNC_METHOD


def test_async_method_multiple(extractor: SymbolExtractor) -> None:
    """All async methods in a class are classified as ASYNC_METHOD."""
    symbols = extract(ASYNC_METHOD_SRC, extractor)
    async_methods = by_type(symbols, ASYNC_METHOD)
    names = {s.name for s in async_methods}
    assert "connect" in names
    assert "disconnect" in names


def test_sync_method_unaffected(extractor: SymbolExtractor) -> None:
    """Sync methods in the same class remain type METHOD."""
    symbols = extract(ASYNC_METHOD_SRC, extractor)
    sync_op = by_name(symbols, "sync_op")
    assert sync_op.type == METHOD


def test_async_method_docstring(extractor: SymbolExtractor) -> None:
    """Async method docstring is extracted correctly."""
    symbols = extract(ASYNC_METHOD_SRC, extractor)
    connect = by_name(symbols, "connect")
    assert connect.docstring == "Connect to service."


def test_async_method_parent_class(extractor: SymbolExtractor) -> None:
    """Async method has parent_class set to the enclosing class name."""
    symbols = extract(ASYNC_METHOD_SRC, extractor)
    connect = by_name(symbols, "connect")
    assert connect.parent_class == "Service"


# ---------------------------------------------------------------------------
# 22. Nested function extraction
# ---------------------------------------------------------------------------


def test_nested_function_extracted(extractor: SymbolExtractor) -> None:
    """Functions defined inside another function are extracted."""
    symbols = extract(NESTED_FUNCTION_SRC, extractor)
    names = {s.name for s in symbols}
    assert "outer" in names
    assert "inner" in names
    assert "another_inner" in names


def test_nested_function_type(extractor: SymbolExtractor) -> None:
    """Nested functions have type FUNCTION (not method)."""
    symbols = extract(NESTED_FUNCTION_SRC, extractor)
    inner = by_name(symbols, "inner")
    assert inner.type == FUNCTION


def test_nested_function_docstring(extractor: SymbolExtractor) -> None:
    """Docstring of a nested function is extracted."""
    symbols = extract(NESTED_FUNCTION_SRC, extractor)
    inner = by_name(symbols, "inner")
    assert inner.docstring == "Inner function."


def test_nested_function_no_parent_class(extractor: SymbolExtractor) -> None:
    """Nested functions have parent_class='' (they are not class members)."""
    symbols = extract(NESTED_FUNCTION_SRC, extractor)
    inner = by_name(symbols, "inner")
    assert inner.parent_class == ""


def test_deeply_nested_functions_extracted(extractor: SymbolExtractor) -> None:
    """Functions nested 3 levels deep are all extracted."""
    symbols = extract(DEEPLY_NESTED_SRC, extractor)
    names = {s.name for s in symbols}
    assert "level_one" in names
    assert "level_two" in names
    assert "level_three" in names


def test_nested_function_inside_method(extractor: SymbolExtractor) -> None:
    """Helper functions defined inside a class method are extracted."""
    symbols = extract(NESTED_IN_METHOD_SRC, extractor)
    names = {s.name for s in symbols}
    assert "_validate" in names
    assert "_transform" in names

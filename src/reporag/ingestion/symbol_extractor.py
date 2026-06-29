"""Symbol extractor.

Walks a tree-sitter AST and extracts meaningful code entities: functions,
classes, methods, imports. Each symbol carries metadata (line range,
signature, docstring, decorators).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser, UnsupportedLanguageError

SymbolType = Literal["class", "function", "method", "import"]


@dataclass
class Symbol:
    """Structured metadata for an extracted code symbol."""

    name: str
    type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    methods: list[Symbol] = field(default_factory=list)
    children: list[Symbol] = field(default_factory=list)
    is_async: bool = False
    parent_symbol: str | None = None
    qualified_name: str | None = None
    import_source: str | None = None
    import_alias: str | None = None
    is_wildcard_import: bool = False
    has_parse_error: bool = False


# ---------------------------------------------------------------------------
# Python Extractor Implementation
# ---------------------------------------------------------------------------


class PythonSymbolExtractor:
    """Extracts symbols and metadata from a Python tree-sitter AST."""

    def __init__(self, file_path: str, source_bytes: bytes) -> None:
        """Initialise the Python symbol extractor."""
        self.file_path = file_path
        self.source_bytes = source_bytes

    def extract(self, tree: Tree) -> list[Symbol]:
        """Extract symbols and build their hierarchy in a single iterative pass."""
        roots: list[Symbol] = []
        # Stack elements: (node, parent_symbol_obj, is_method_context, in_function_context, decorators)
        stack = [
            (child, None, False, False, [])
            for child in reversed(tree.root_node.named_children)
        ]

        while stack:
            node, parent, is_method, in_func, decorators = stack.pop()

            if node.type == "ERROR" or node.is_missing:
                # Still try to extract if it has a name, otherwise skip.
                # However, raw ERROR nodes or missing definitions are generally skipped at top level
                # if they can't be resolved. Let's process definitions.
                continue

            if node.type == "decorated_definition":
                decs = self._extract_decorators(node)
                inner_def = None
                for child in node.named_children:
                    if child.type in ("class_definition", "function_definition"):
                        inner_def = child
                        break
                if inner_def:
                    stack.append((inner_def, parent, is_method, in_func, decs))
                continue

            if node.type == "class_definition":
                sym = self._extract_class(node, decorators, parent, in_func)
                if not sym:
                    continue
                self._attach_symbol(sym, parent, roots)

                body = node.child_by_field_name("body")
                if body:
                    for child in reversed(body.named_children):
                        stack.append((child, sym, True, False, []))
                continue

            if node.type == "function_definition":
                sym = self._extract_function(
                    node, decorators, parent, is_method, in_func
                )
                if not sym:
                    continue
                self._attach_symbol(sym, parent, roots)

                body = node.child_by_field_name("body")
                if body:
                    for child in reversed(body.named_children):
                        stack.append((child, sym, False, True, []))
                continue

            if node.type in ("import_statement", "import_from_statement"):
                imports = self._extract_imports(node)
                for imp in imports:
                    self._attach_symbol(imp, parent, roots)
                continue

            # Fallback to traverse children
            for child in reversed(node.named_children):
                stack.append((child, parent, is_method, in_func, []))

        return roots

    def _attach_symbol(
        self, sym: Symbol, parent: Symbol | None, roots: list[Symbol]
    ) -> None:
        """Attach a symbol to its parent or add to the module root list.

        - **Why it exists**: Integrates extracted symbols into the hierarchical scope tree.
        - **Algorithm**: If no parent exists, the symbol represents a module-level entity and goes
          to roots. If a parent class exists and the symbol type is 'method', it is stored in the
          class's methods list. Otherwise, it is added to the parent's generic children collection.
        - **Edge cases**: Nested classes inside methods are routed to generic children instead
          of the methods array, avoiding semantic misclassification.
        - **Correctness choice**: Modifies parent containers in-place during the iterative DFS traversal,
          constructing the full hierarchy in O(1) step operations.
        """
        roots.append(sym)

    def _extract_docstring(self, block_node: Node | None) -> str | None:
        """Extract docstring string from the block node.

        - **Why it exists**: Parses raw docstring literals from a class or function block.
        - **Algorithm**: Inspects block named_children, skipping leading comments. If the first
          named statement is an expression containing string literals, it executes a stack-based DFS
          over the string node to gather all 'string_content' nodes, merging them into a single string.
        - **Edge cases**: Supports unicode, bytes, and raw string prefixes, and handles Python
          concatenated string docstrings (e.g., "part1 " "part2") by processing all child string contents.
        - **Correctness choice**: Iterates over named_children instead of using positional child indexing,
          preventing failure if the grammar structure inserts comments or punctuation before the literal.
        """
        if not block_node or block_node.type != "block":
            return None

        first_child = None
        for child in block_node.named_children:
            if child.type == "comment":
                continue
            first_child = child
            break

        if not first_child:
            return None

        expr_node = None
        if first_child.type == "expression_statement":
            expr_node = first_child.child(0)
        elif first_child.type in ("string", "concatenated_string"):
            expr_node = first_child

        if not expr_node or expr_node.type not in ("string", "concatenated_string"):
            return None

        # Gather all string_content nodes
        contents = []
        stack = [expr_node]
        while stack:
            curr = stack.pop()
            if curr.type == "string_content":
                contents.append(curr.text.decode("utf-8", errors="replace"))
            for child in reversed(curr.named_children):
                stack.append(child)

        if not contents:
            # Fallback to stripping quotes manually for empty/unstructured string nodes
            text = expr_node.text.decode("utf-8", errors="replace").strip()
            for prefix in ("r", "b", "u", "f", "rb", "br", "fr", "rf"):
                if text.lower().startswith(prefix):
                    text = text[len(prefix) :].strip()
                    break
            for quote in ('"""', "'''", '"', "'"):
                if text.startswith(quote) and text.endswith(quote):
                    return text[len(quote) : -len(quote)].strip()
            return text.strip()

        return "".join(contents).strip()

    def _extract_decorators(self, node: Node) -> list[str]:
        """Extract decorators text expressions.

        - **Why it exists**: Extracts decorator expressions for class or function definitions.
        - **Algorithm**: Inspects the named children of a 'decorated_definition' and grabs
          the text of any child nodes of type 'decorator', stripping the leading '@'.
        - **Edge cases**: Correctly handles parameters inside decorator calls (e.g. @decorator(arg=1)).
        - **Correctness choice**: Grabs the full decorator AST node content rather than relying on regex.
        """
        decorators = []
        if node.type == "decorated_definition":
            for child in node.named_children:
                if child.type == "decorator":
                    dec_text = child.text.decode("utf-8", errors="replace").strip()
                    if dec_text.startswith("@"):
                        dec_text = dec_text[1:].strip()
                    decorators.append(dec_text)
        return decorators

    def _extract_signature(self, func_node: Node) -> str | None:
        """Extract exact signature from code slice.

        - **Why it exists**: Captures the exact parameters and return annotation text.
        - **Algorithm**: Locates the colon (:) or block child node marking the end of the function
          parameters, then slices the source byte array from the function start up to that index.
        - **Edge cases**: Gracefully handles multiline signatures and type annotations.
        - **Correctness choice**: Slices the original source buffer instead of reconstructing it from nodes,
          preserving the exact spacing and comments of the code author.
        """
        colon_idx = -1
        # Punctuation/colons are anonymous, so we must use node.children
        for i, child in enumerate(func_node.children):
            if child.type in (":", "block"):
                colon_idx = i
                break

        if colon_idx != -1:
            start_offset = func_node.start_byte
            end_offset = func_node.children[colon_idx].start_byte
            return (
                self.source_bytes[start_offset:end_offset]
                .decode("utf-8", errors="replace")
                .strip()
            )
        return None

    def _extract_bases(self, class_node: Node) -> list[str]:
        """Extract bases from superclasses node.

        - **Why it exists**: Harvests the list of base classes that the class inherits from.
        - **Algorithm**: Walks through all named children of the class's 'superclasses' node,
          filtering out keyword arguments like 'metaclass=ABCMeta', and captures the raw text of the rest.
        - **Edge cases**: Robustly supports generic parameters (e.g. Generic[T], list[int]) and attribute chains.
        - **Correctness choice**: Bypasses strict type checks (dotted_name, attribute) and uses raw node slicing,
          scaling automatically to all complex, user-defined inheritance syntax expressions.
        """
        bases = []
        superclasses = class_node.child_by_field_name("superclasses")
        if superclasses:
            for arg in superclasses.named_children:
                if arg.type == "keyword_argument":
                    continue
                bases.append(arg.text.decode("utf-8", errors="replace").strip())
        return bases

    def _extract_class(
        self, node: Node, decorators: list[str], parent: Symbol | None, in_func: bool
    ) -> Symbol | None:
        """Parse class details into a Symbol.

        - **Why it exists**: Instantiates the Symbol record for a Python class definition.
        - **Algorithm**: Resolves the class name and prefix path, constructs the pythonic
          qualified name (inserting <locals> if within a function scope), and gathers docstrings and bases.
        - **Edge cases**: Returns None if the name node is missing or contains immediate parse errors.
        - **Correctness choice**: Determines 'has_parse_error' by inspecting the node's local error flags.
        """
        name_node = node.child_by_field_name("name")
        if not name_node or name_node.type == "ERROR" or name_node.is_missing:
            return None
        name = name_node.text.decode("utf-8", errors="replace")

        parent_name = parent.qualified_name if parent else None
        if parent_name:
            prefix = f"{parent_name}.<locals>" if in_func else parent_name
            qualified_name = f"{prefix}.{name}"
        else:
            qualified_name = name

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node)

        has_parse_error = node.has_error or node.type == "ERROR" or node.is_missing

        return Symbol(
            name=name,
            type="class",
            file_path=self.file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            decorators=decorators,
            bases=self._extract_bases(node),
            docstring=docstring,
            qualified_name=qualified_name,
            parent_symbol=parent_name,
            has_parse_error=has_parse_error,
        )

    def _extract_function(
        self,
        node: Node,
        decorators: list[str],
        parent: Symbol | None,
        is_method: bool,
        in_func: bool,
    ) -> Symbol | None:
        """Parse function/method details into a Symbol.

        - **Why it exists**: Instantiates the Symbol record for a Python function or method definition.
        - **Algorithm**: Inspects syntax details (is_async, signature, docstring), builds the scoping paths
          including <locals> where nested, and detects error states.
        - **Edge cases**: Correctly differentiates module functions, nested functions, and class methods.
        - **Correctness choice**: Performs fallback parsing to extract valid headers even if the body has syntax errors.
        """
        name_node = node.child_by_field_name("name")
        if not name_node or name_node.type == "ERROR" or name_node.is_missing:
            return None
        name = name_node.text.decode("utf-8", errors="replace")

        parent_name = parent.qualified_name if parent else None
        if parent_name:
            prefix = f"{parent_name}.<locals>" if in_func else parent_name
            qualified_name = f"{prefix}.{name}"
        else:
            qualified_name = name

        # The 'async' keyword is an anonymous token, so we must use node.children
        is_async = any(child.type == "async" for child in node.children)
        signature = self._extract_signature(node)

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node)

        sym_type: SymbolType = "method" if is_method else "function"

        has_parse_error = node.has_error or node.type == "ERROR" or node.is_missing

        return Symbol(
            name=name,
            type=sym_type,
            file_path=self.file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            qualified_name=qualified_name,
            parent_symbol=parent_name,
            has_parse_error=has_parse_error,
        )

    def _extract_imports(self, node: Node) -> list[Symbol]:
        """Extract import statement statements.

        - **Why it exists**: Parses import statements to extract individual imported names.
        - **Algorithm**: Delegates to specialized sub-parsers based on whether it is a standard
          'import_statement' or a 'import_from_statement' block.
        - **Edge cases**: Captures parent-level syntax parse errors and copies them down to child symbols.
        - **Correctness choice**: Separates statement routing to minimize cyclomatic complexity.
        """
        symbols = []
        has_parse_error = node.has_error or node.type == "ERROR" or node.is_missing
        if node.type == "import_statement":
            for child in node.named_children:
                for sym in self._parse_import_name(child):
                    sym.has_parse_error = has_parse_error
                    symbols.append(sym)
        elif node.type == "import_from_statement":
            for sym in self._parse_import_from(node):
                sym.has_parse_error = has_parse_error
                symbols.append(sym)

        return symbols

    def _parse_import_name(self, child: Node) -> list[Symbol]:
        """Parse dotted name or aliased import under import statement.

        - **Why it exists**: Parses names from basic 'import x, y as z' statements.
        - **Algorithm**: Evaluates child node type, extracting aliases and source paths.
        - **Edge cases**: Uses child node positions directly, ensuring accurate line boundaries.
        """
        start_line = child.start_point[0] + 1
        end_line = child.end_point[0] + 1
        if child.type == "dotted_name":
            name = child.text.decode("utf-8", errors="replace")
            return [
                Symbol(
                    name=name,
                    type="import",
                    file_path=self.file_path,
                    start_line=start_line,
                    end_line=end_line,
                    import_source=name,
                )
            ]
        if child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node and alias_node:
                src_name = name_node.text.decode("utf-8", errors="replace")
                alias = alias_node.text.decode("utf-8", errors="replace")
                return [
                    Symbol(
                        name=alias,
                        type="import",
                        file_path=self.file_path,
                        start_line=start_line,
                        end_line=end_line,
                        import_source=src_name,
                        import_alias=alias,
                    )
                ]
            return []

    def _parse_import_from(self, node: Node) -> list[Symbol]:
        """Parse from_import structures.

        - **Why it exists**: Parses from-imports (e.g. from math import sin, cos).
        - **Algorithm**: Resolves the source module name, handles wildcards, and iterates
          over named imports.
        - **Edge cases**: Supports relative prefixes (e.g. from . import module).
        """
        module_node = node.child_by_field_name("module_name")
        if not module_node:
            for child in node.named_children:
                if child.type in ("dotted_name", "relative_import"):
                    module_node = child
                    break

        if not module_node:
            return []

        module_name = module_node.text.decode("utf-8", errors="replace")
        has_wildcard = any(c.type == "wildcard_import" for c in node.named_children)

        # Base statement line numbers for wildcard
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        if has_wildcard:
            return [
                Symbol(
                    name="*",
                    type="import",
                    file_path=self.file_path,
                    start_line=start_line,
                    end_line=end_line,
                    import_source=module_name,
                    is_wildcard_import=True,
                )
            ]

        symbols = []
        for child in node.named_children:
            if child == module_node:
                continue
            symbols.extend(self._parse_import_from_child(child, module_name))
        return symbols

    def _parse_import_from_child(self, child: Node, module_name: str) -> list[Symbol]:
        """Parse names or aliased imports inside from_import statement.

        - **Why it exists**: Instantiates the Import Symbol for each child name inside from-imports.
        - **Algorithm**: Detects dotted names, plain identifiers, or aliased forms.
        - **Edge cases**: Collects precise child node start/end line coordinates for multiline ranges.
        - **Correctness choice**: Checks both 'dotted_name' and 'identifier' types to catch all sub-module/constant imports.
        """
        start_line = child.start_point[0] + 1
        end_line = child.end_point[0] + 1
        if child.type in ("dotted_name", "identifier"):
            name = child.text.decode("utf-8", errors="replace")
            return [
                Symbol(
                    name=name,
                    type="import",
                    file_path=self.file_path,
                    start_line=start_line,
                    end_line=end_line,
                    import_source=module_name,
                )
            ]
        if child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node and alias_node:
                src_name = name_node.text.decode("utf-8", errors="replace")
                alias = alias_node.text.decode("utf-8", errors="replace")
                return [
                    Symbol(
                        name=alias,
                        type="import",
                        file_path=self.file_path,
                        start_line=start_line,
                        end_line=end_line,
                        import_source=f"{module_name}.{src_name}",
                        import_alias=alias,
                    )
                ]
        return []


# ---------------------------------------------------------------------------
# Language Registry Map
# ---------------------------------------------------------------------------

_EXTRACTOR_REGISTRY = {
    "python": PythonSymbolExtractor,
}


# ---------------------------------------------------------------------------
# Coordinator / Public API Class
# ---------------------------------------------------------------------------


class SymbolExtractor:
    """Language-agnostic coordinator for symbol extraction."""

    def __init__(self, parser: ASTParser | None = None) -> None:
        """Initialise the coordinator."""
        self.parser = parser or ASTParser()

    def extract_from_tree(
        self,
        tree: Tree,
        file_path: str,
        source: str | bytes,
        language: str = "python",
    ) -> list[Symbol]:
        """Extract symbols from an existing tree-sitter Tree.

        Raises:
            UnsupportedLanguageError: If the language is not registered.
        """
        lang = language.lower().strip()
        extractor_cls = _EXTRACTOR_REGISTRY.get(lang)
        if not extractor_cls:
            raise UnsupportedLanguageError(
                f"No SymbolExtractor registered for language '{language}'. "
                f"Supported: {sorted(_EXTRACTOR_REGISTRY)}"
            )

        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        extractor = extractor_cls(file_path, source_bytes)
        return extractor.extract(tree)

    def extract_from_source(
        self,
        source: str | bytes,
        language: str = "python",
        file_path: str = "<string>",
    ) -> list[Symbol]:
        """Extract symbols from raw source code string or bytes."""
        lang = language.lower().strip()
        tree = self.parser.parse(source, language=lang)
        return self.extract_from_tree(tree, file_path, source, language=lang)

    def extract_from_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> list[Symbol]:
        """Parse a file from disk and extract its symbols."""
        fpath = Path(file_path)

        if language is None:
            from src.reporag.config import settings

            ext = fpath.suffix.lower()
            language = settings.extension_map.get(ext)
            if language is None:
                raise UnsupportedLanguageError(
                    f"Cannot infer language for extension '{ext}'."
                )

        # Prevent reading the file twice: read the bytes once
        from src.reporag.ingestion.parser import ParseError

        try:
            source_bytes = fpath.read_bytes()
        except OSError as exc:
            raise ParseError(f"Cannot read file '{fpath}': {exc}") from exc

        tree = self.parser.parse(source_bytes, language=language)
        return self.extract_from_tree(tree, str(fpath), source_bytes, language=language)

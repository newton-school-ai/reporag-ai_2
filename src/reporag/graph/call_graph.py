"""AST-aware call graph builder.

Walks ASTs to identify function call expressions and resolves them to
target symbols. Builds directed edges: caller -> callee with call site
metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Node, Tree

from reporag.ingestion.parser import ASTParser
from reporag.ingestion.symbol_extractor import Symbol, SymbolExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallEdge:
    """A directed edge in the call graph representing a function or method call.

    Attributes:
        caller: Fully qualified name of the calling function/method, or "<module>"
                for module-level calls.
        callee: Fully qualified name of the resolved target symbol being called.
        file_path: The workspace-relative file path containing the call site.
        line: 1-based line number of the call expression.
    """

    caller: str
    callee: str
    file_path: str
    line: int


@dataclass
class FileContext:
    """Stores definition, import, and syntax tree cache for a single file."""

    file_path: str
    source_bytes: bytes
    tree: Tree
    symbols: list[Symbol]
    definitions: dict[str, Symbol] = field(default_factory=dict)
    imports: dict[str, Symbol] = field(default_factory=dict)
    wildcards: list[Symbol] = field(default_factory=list)


def _get_possible_module_names(file_path: str) -> list[str]:
    """Determine possible dotted module names for a given file path.

    Handles standard file paths, 'src/' prefix removal, and package __init__.py files.
    """
    p = Path(file_path)
    parts = list(p.parts)
    if not parts:
        return []

    # Handle __init__.py representing the package itself
    parts = parts[:-1] if parts[-1] == "__init__.py" else list(p.with_suffix("").parts)

    if not parts:
        return []

    dotted1 = ".".join(parts)
    res = [dotted1]
    # If the workspace path starts with src/, it might also be imported without it
    if parts[0] == "src":
        dotted2 = ".".join(parts[1:])
        res.append(dotted2)

    return res


def _flatten_attribute(node: Node) -> list[Node] | None:
    """Flatten an attribute node chain (e.g. self.obj.method) into a list of constituent nodes."""
    if node.type == "identifier":
        return [node]
    elif node.type == "attribute":
        obj = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if obj and attr:
            prefix = _flatten_attribute(obj)
            if prefix is not None:
                return prefix + [attr]
    return None


def _get_called_name(node: Node) -> str | None:
    """Retrieve the simple name of the function or method being called."""
    if node.type == "identifier":
        return node.text.decode("utf-8", errors="replace")
    elif node.type == "attribute":
        attr = node.child_by_field_name("attribute")
        if attr:
            return attr.text.decode("utf-8", errors="replace")
    return None


class CallGraphBuilder:
    """Builds a call graph by walking tree-sitter ASTs and resolving targets."""

    def __init__(self, parser: ASTParser | None = None) -> None:
        """Initialize the CallGraphBuilder."""
        self.parser = parser or ASTParser()
        self.extractor = SymbolExtractor(self.parser)
        self.module_to_file: dict[str, str] = {}
        self.global_definitions: dict[str, tuple[Symbol, FileContext]] = {}
        self.global_methods: dict[str, list[tuple[str, FileContext]]] = {}

    def build_from_sources(self, sources: dict[str, str]) -> list[CallEdge]:
        """Build the call graph from in-memory source files.

        Args:
            sources: Dict mapping file paths to their source code strings.

        Returns:
            A list of CallEdge objects.
        """
        contexts: dict[str, FileContext] = {}
        self.module_to_file.clear()

        # Build FileContext and module_to_file lookup index
        for fpath, source in sources.items():
            source_bytes = source.encode("utf-8")
            try:
                tree = self.parser.parse(source_bytes, language="python")
                symbols = self.extractor.extract_from_tree(
                    tree, fpath, source_bytes, language="python"
                )
            except Exception as exc:
                logger.warning("Failed to parse/extract symbols for %s: %s", fpath, exc)
                continue

            ctx = FileContext(
                file_path=fpath, source_bytes=source_bytes, tree=tree, symbols=symbols
            )
            for sym in symbols:
                if sym.type == "import":
                    if sym.is_wildcard_import:
                        ctx.wildcards.append(sym)
                    else:
                        ctx.imports[sym.name] = sym
                else:
                    if sym.qualified_name:
                        ctx.definitions[sym.qualified_name] = sym

            contexts[fpath] = ctx

            # Map possible module names to this file path
            for mod_name in _get_possible_module_names(fpath):
                self.module_to_file[mod_name] = fpath

        return self._resolve_all_calls(contexts)

    def build_from_files(self, file_paths: list[str | Path]) -> list[CallEdge]:
        """Build the call graph from files on the local disk.

        Args:
            file_paths: List of absolute or workspace-relative file paths.

        Returns:
            A list of CallEdge objects.
        """
        sources: dict[str, str] = {}
        for path in file_paths:
            fpath = Path(path)
            try:
                sources[str(path)] = fpath.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Failed to read file %s: %s", path, exc)
                continue
        return self.build_from_sources(sources)

    def _resolve_all_calls(self, contexts: dict[str, FileContext]) -> list[CallEdge]:
        """Find and resolve all function and method call edges."""
        call_info_list: list[dict[str, Any]] = []
        edges: list[CallEdge] = []

        # Build global definitions and methods index for fast O(1) lookups
        self.global_definitions.clear()
        self.global_methods.clear()
        for ctx in contexts.values():
            for qname, sym in ctx.definitions.items():
                self.global_definitions[qname] = (sym, ctx)
                if sym.type == "method":
                    method_name = sym.name
                    if method_name not in self.global_methods:
                        self.global_methods[method_name] = []
                    self.global_methods[method_name].append((qname, ctx))

        # Find all raw call expressions
        for fpath, ctx in contexts.items():
            file_calls: list[dict[str, Any]] = []
            self._find_calls(ctx.tree.root_node, ctx, None, file_calls, False)
            for call in file_calls:
                call["file_path"] = fpath
                call["context"] = ctx
            call_info_list.extend(file_calls)

        # Resolve targets for each call expression
        for call in call_info_list:
            func_node = call["func_node"]
            line = call["line"]
            caller = call["caller"]
            file_path = call["file_path"]
            ctx = call["context"]

            flat_nodes = _flatten_attribute(func_node)
            resolved_callee = None

            if flat_nodes is not None:
                parts = [n.text.decode("utf-8", errors="replace") for n in flat_nodes]
                if parts[0] == "self" and len(parts) >= 2:
                    # Resolve self.method name
                    caller_parts = caller.split(".")
                    class_qual = None
                    for i in range(len(caller_parts), 0, -1):
                        prefix = ".".join(caller_parts[:i])
                        if (
                            prefix in ctx.definitions
                            and ctx.definitions[prefix].type == "class"
                        ):
                            class_qual = prefix
                            break

                    if class_qual:
                        resolved_callee = self._resolve_method_in_class(
                            parts[1], class_qual, ctx, contexts
                        )
                        if not resolved_callee:
                            resolved_callee = f"{class_qual}.{parts[1]}"
                else:
                    name = ".".join(parts)
                    resolved_callee = self._resolve_symbol_name(
                        name, ctx, caller, contexts
                    )

                    # Handled method calls on objects: fallback to global method lookup
                    if not resolved_callee and len(parts) == 2:
                        resolved_callee = self._resolve_method_globally(
                            parts[1], ctx, contexts
                        )
            else:
                called_name = _get_called_name(func_node)
                if called_name:
                    resolved_callee = self._resolve_method_globally(
                        called_name, ctx, contexts
                    )

            if not resolved_callee:
                resolved_callee = func_node.text.decode("utf-8", errors="replace")

            edges.append(
                CallEdge(
                    caller=caller,
                    callee=resolved_callee,
                    file_path=file_path,
                    line=line,
                )
            )

        return edges

    def _find_calls(
        self,
        node: Node,
        file_context: FileContext,
        current_function: str | None,
        results: list[dict[str, Any]],
        in_func: bool = False,
    ) -> None:
        """Recursively walk the AST and record caller-callee candidate locations."""
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8", errors="replace")
                if current_function:
                    prefix = (
                        f"{current_function}.<locals>" if in_func else current_function
                    )
                    func_qual = f"{prefix}.{name}"
                else:
                    func_qual = name

                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._find_calls(
                            child, file_context, func_qual, results, in_func=True
                        )
                return

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8", errors="replace")
                if current_function:
                    prefix = (
                        f"{current_function}.<locals>" if in_func else current_function
                    )
                    class_qual = f"{prefix}.{name}"
                else:
                    class_qual = name

                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._find_calls(
                            child, file_context, class_qual, results, in_func=False
                        )
                return

        elif node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                line = node.start_point[0] + 1
                results.append(
                    {
                        "func_node": func_node,
                        "line": line,
                        "caller": current_function or "<module>",
                    }
                )

        for child in node.children:
            self._find_calls(child, file_context, current_function, results, in_func)

    def _resolve_module_to_file(
        self, module_name: str, current_file: str, contexts: dict[str, FileContext]
    ) -> str | None:
        """Resolve a dotted module name to a workspace file path."""
        # Relative import resolution
        if module_name.startswith("."):
            dots_count = 0
            while dots_count < len(module_name) and module_name[dots_count] == ".":
                dots_count += 1
            remainder = module_name[dots_count:]

            parts = list(Path(current_file).parts)
            dir_parts = parts[:-1]

            go_up = dots_count - 1
            if go_up > 0:
                dir_parts = dir_parts[:-go_up] if go_up <= len(dir_parts) else []

            if remainder:
                dir_parts.extend(remainder.split("."))

            target_mod = ".".join(dir_parts)
            if target_mod in self.module_to_file:
                return self.module_to_file[target_mod]
            return None

        # Absolute import check
        if module_name in self.module_to_file:
            return self.module_to_file[module_name]

        # Directory-prefix fallback for same-package absolute imports
        parts = list(Path(current_file).parts)
        if len(parts) > 1:
            dir_mod = ".".join(parts[:-1])
            combined_mod = f"{dir_mod}.{module_name}"
            if combined_mod in self.module_to_file:
                return self.module_to_file[combined_mod]

        return None

    def _resolve_symbol_name(
        self,
        name: str,
        file_context: FileContext,
        caller_qual: str | None,
        contexts: dict[str, FileContext],
    ) -> str | None:
        """Resolve a name (optionally dot-qualified) to a fully qualified symbol name."""
        parts = name.split(".")
        base = parts[0]

        def is_defined_in(qname: str, ctx: FileContext) -> bool:
            return qname in ctx.definitions

        # 1. Local and enclosing scopes
        if caller_qual:
            scopes = []
            caller_parts = caller_qual.split(".")
            for i in range(len(caller_parts), 0, -1):
                sub_prefix = ".".join(caller_parts[:i])
                scopes.append(f"{sub_prefix}.<locals>.{base}")
                scopes.append(f"{sub_prefix}.{base}")

            for candidate in scopes:
                suffix = ".".join(parts[1:])
                full_candidate = f"{candidate}.{suffix}" if suffix else candidate
                if is_defined_in(full_candidate, file_context):
                    return full_candidate

        # 2. Module level scope in the same file
        suffix = ".".join(parts[1:])
        module_candidate = f"{base}.{suffix}" if suffix else base
        if is_defined_in(module_candidate, file_context):
            return module_candidate

        # 3. Import resolution
        if base in file_context.imports:
            imp_symbol = file_context.imports[base]
            import_src = imp_symbol.import_source
            if not import_src:
                return None

            # Try to resolve import_src as module directly
            target_file = self._resolve_module_to_file(
                import_src, file_context.file_path, contexts
            )
            if target_file:
                target_name = imp_symbol.name
                if imp_symbol.import_alias:
                    src_parts = import_src.split(".")
                    target_name = src_parts[-1]

                suffix = ".".join(parts[1:])
                qname_in_target = f"{target_name}.{suffix}" if suffix else target_name
                target_ctx = contexts[target_file]
                if is_defined_in(qname_in_target, target_ctx):
                    return qname_in_target
            else:
                # Dotted import (e.g. from pkg.module import symbol)
                src_parts = import_src.split(".")
                for i in range(len(src_parts) - 1, 0, -1):
                    parent_mod = ".".join(src_parts[:i])
                    target_file = self._resolve_module_to_file(
                        parent_mod, file_context.file_path, contexts
                    )
                    if target_file:
                        symbol_path = src_parts[i:]
                        target_name = ".".join(symbol_path)
                        suffix = ".".join(parts[1:])
                        qname_in_target = (
                            f"{target_name}.{suffix}" if suffix else target_name
                        )
                        target_ctx = contexts[target_file]
                        if is_defined_in(qname_in_target, target_ctx):
                            return qname_in_target
                        break

        # 4. Wildcard imports
        for wild_sym in file_context.wildcards:
            wild_src = wild_sym.import_source
            if wild_src:
                target_file = self._resolve_module_to_file(
                    wild_src, file_context.file_path, contexts
                )
                if target_file:
                    target_ctx = contexts[target_file]
                    suffix = ".".join(parts[1:])
                    qname_in_target = f"{base}.{suffix}" if suffix else base
                    if is_defined_in(qname_in_target, target_ctx):
                        return qname_in_target

        return None

    def _find_class_symbol(
        self,
        class_qual: str,
        file_context: FileContext,
        contexts: dict[str, FileContext],
    ) -> Symbol | None:
        """Find the Symbol object matching class_qual prefix."""
        if class_qual in file_context.definitions:
            sym = file_context.definitions[class_qual]
            if sym.type == "class":
                return sym

        resolved = self._resolve_symbol_name(class_qual, file_context, None, contexts)
        if resolved and resolved in self.global_definitions:
            sym, _ = self.global_definitions[resolved]
            if sym.type == "class":
                return sym
        return None

    def _resolve_method_in_class(
        self,
        method_name: str,
        class_qual: str,
        file_context: FileContext,
        contexts: dict[str, FileContext],
        visited_classes: set[str] | None = None,
    ) -> str | None:
        """Resolve a method name in a class definition, checking inheritance and preventing loops."""
        if visited_classes is None:
            visited_classes = set()

        if class_qual in visited_classes:
            return None
        visited_classes.add(class_qual)

        class_sym = self._find_class_symbol(class_qual, file_context, contexts)
        if not class_sym:
            return None

        class_file = class_sym.file_path
        class_ctx = contexts.get(class_file)
        if class_ctx:
            candidate = f"{class_qual}.{method_name}"
            if candidate in class_ctx.definitions:
                return candidate

        # Check superclass bases recursively
        for base in class_sym.bases:
            base_ctx = contexts.get(class_file) or file_context
            resolved_base = self._resolve_symbol_name(base, base_ctx, None, contexts)
            if resolved_base:
                res = self._resolve_method_in_class(
                    method_name, resolved_base, base_ctx, contexts, visited_classes
                )
                if res:
                    return res

        return None

    def _resolve_method_globally(
        self,
        method_name: str,
        file_context: FileContext,
        contexts: dict[str, FileContext],
    ) -> str | None:
        """Fallback lookup to match method calls to a candidate definition globally."""
        candidates = self.global_methods.get(method_name)
        if not candidates:
            return None

        # Prefer method candidates defined in the same file
        same_file = [
            q for q, ctx in candidates if ctx.file_path == file_context.file_path
        ]
        if same_file:
            return same_file[0]

        # Prefer candidates whose module is imported in this file
        for q, ctx in candidates:
            for mod in _get_possible_module_names(ctx.file_path):
                if mod in file_context.imports:
                    return q

        # Fallback to unique matches
        if len(candidates) == 1:
            return candidates[0][0]

        return None

"""Symbol extractor.

Walks a tree-sitter AST and extracts meaningful code entities: functions,
classes, methods, imports. Each symbol carries metadata (line range,
signature, docstring, decorators).
"""

# TODO: Implement in Issue 7
# - Walk AST to find function_definition, class_definition, import nodes
# - Extract: name, type, file_path, start_line, end_line, signature
# - Extract: docstring, decorators, parent_class (if method), return_type_hint
# - Handle: nested functions, static/class methods, async, property decorators
# - Return list of Symbol dataclass objects

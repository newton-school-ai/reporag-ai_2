"""Tree-sitter AST parser.

Parses source files into tree-sitter ASTs. Supports Python (extensible
to JS/TS). Handles parse errors gracefully with partial ASTs.
"""

# TODO: Implement in Issue 6
# - Load tree-sitter grammar for target language
# - Parse source string -> tree-sitter Tree
# - Walk tree, return structured node data (type, text, start/end lines)
# - Handle syntax errors (return partial AST, flag errors)
# - Language-agnostic interface: Parser.parse(source, language)

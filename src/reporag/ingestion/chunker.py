"""Semantic code chunker.

AST-aware chunking that respects function/class boundaries. Never splits
a function mid-body. Large functions are split at logical points with
function signature overlap.
"""

# TODO: Implement in Issue 8
# - Chunk at AST-node boundaries (functions, classes, top-level blocks)
# - Configurable max chunk size (tokens via tiktoken)
# - Split large functions at logical points (between statements)
# - Overlap: include function signature in continuation chunks
# - Each chunk: file_path, start_line, end_line, parent_symbol, language, token_count

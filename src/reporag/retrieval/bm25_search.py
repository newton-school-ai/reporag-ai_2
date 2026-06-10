"""BM25 sparse keyword search.

Queries the BM25 index with code-aware tokenization. Excels at finding
exact identifier matches that vector search may miss.
"""

# TODO: Implement in Issue 17
# - Load pre-built BM25 index
# - Tokenize query using same code-aware tokenizer as indexing
# - Return top-k results with BM25 scores
# - Boost exact function/class name matches (configurable boost factor)
# - Return RetrievalResult objects (same schema as vector search)

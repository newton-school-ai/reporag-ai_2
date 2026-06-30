"""Hybrid index builder.

Creates and populates both a Qdrant vector collection and a BM25 sparse
index. Includes a code-aware tokenizer that splits camelCase and snake_case
identifiers for better keyword matching.
"""

# TODO: Implement in Issue 15
# - Create Qdrant collection with payload schema (file, lines, symbol, language)
# - Upsert code + doc embeddings with metadata payloads
# - Build BM25 index from tokenized code (rank-bm25)
# - Code-aware tokenizer: split camelCase, snake_case, operators
# - Support incremental updates (add new files without full rebuild)

"""Vector semantic search.

Queries Qdrant with an embedded query vector. Returns top-k results with
scores and metadata payloads. Supports filtering by language, file path,
and symbol type.
"""

# TODO: Implement in Issue 16
# - Embed query using code_embedder or doc_embedder (auto-detect)
# - Search Qdrant collection with query vector, top_k, optional filters
# - Search code and doc collections separately, merge results
# - Return RetrievalResult objects with: score, file_path, lines, symbol, chunk_text

"""Cross-encoder reranker.

Scores (query, chunk) pairs using a cross-encoder model. The cross-encoder
sees both query and document together, producing more accurate relevance
scores than bi-encoder retrieval.
"""

# TODO: Implement in Issue 19
# - Load cross-encoder model (ms-marco-MiniLM-L-6-v2)
# - rerank(query, candidates, top_k) -> reranked list
# - Score each (query, candidate.chunk_text) pair
# - Sort by rerank_score descending, return top_k
# - Batch scoring for efficiency

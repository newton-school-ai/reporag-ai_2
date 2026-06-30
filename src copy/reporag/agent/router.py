"""Strategy router.

Routes each sub-query to the optimal retrieval strategy: graph, vector,
bm25, or hybrid. Routing based on sub-query characteristics with
LLM-assisted classification and rule-based fallback.
"""

# TODO: Implement in Issue 22
# - route(sub_query) -> Strategy (graph / vector / bm25 / hybrid)
# - Rules: identifier mention -> BM25, structural ("what calls X") -> graph,
#   semantic ("how does X work") -> vector, ambiguous -> hybrid
# - LLM-assisted routing for complex cases
# - Rule-based fallback when LLM is unavailable

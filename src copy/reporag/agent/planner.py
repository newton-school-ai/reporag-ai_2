"""Agentic query planner.

Contains the query classifier and query decomposer. Classifies queries
into simple-lookup / multi-hop / exploratory, then decomposes complex
queries into ordered sub-queries using a LangGraph state machine.
"""

# TODO: Implement in Issues 20, 21
#
# QueryClassifier (Issue 20):
# - LLM-based classification with few-shot examples
# - Categories: simple-lookup, multi-hop, exploratory
# - Returns (query_type, confidence)
# - Low confidence (<threshold) falls back to multi-hop
#
# QueryDecomposer (Issue 21):
# - LangGraph state machine for decomposition
# - Input: complex query + repo context (modules, key symbols)
# - Output: ordered list of SubQuery objects with dependency edges
# - Each SubQuery: text, expected_answer_type, context_from (prior IDs)
# - Handles queries that do not need decomposition (single step)

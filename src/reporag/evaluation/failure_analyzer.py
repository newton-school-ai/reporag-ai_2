"""Failure mode classifier.

Classifies RAG pipeline failures into categories: retrieval failure
(relevant chunks not retrieved), generation failure (LLM hallucination
or irrelevance), and context overflow (too much/too little context).
"""

# TODO: Implement in Issue 37
# - analyze(query, retrieved, answer, ground_truth) -> FailureAnalysis
# - Categories: retrieval_failure, generation_failure, context_overflow, success
# - Retrieval failure: relevant chunks not in retrieved set
# - Generation failure: answer not faithful to retrieved context
# - Context overflow: context too large (truncation lost key info)
# - Return: category, confidence, explanation, suggested_fix

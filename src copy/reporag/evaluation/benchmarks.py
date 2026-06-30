"""Code QA benchmark suites.

Runs evaluation against real open-source repositories with ground-truth
Q&A pairs. Reports aggregate and per-query metrics with comparison
across retrieval strategies.
"""

# TODO: Implement in Issue 37
# - Load eval dataset from examples/eval_dataset.json
# - Run retrieval for each query, compare against ground truth
# - Compute aggregate metrics (mean, std across queries)
# - Comparison mode: vector-only vs BM25-only vs hybrid vs hybrid+rerank
# - Output as JSON + human-readable table (via rich)

"""Sub-query executor.

Executes sub-queries in dependency order, passing context from earlier
steps to later ones. Orchestrates the retrieval engine based on the
strategy assigned by the router.
"""

# TODO: Implement in Issue 22
# - execute(plan) -> dict[step_id, list[RetrievalResult]]
# - Topological sort sub-queries by dependency edges
# - Execute each sub-query using the retrieval engine + assigned strategy
# - Pass retrieved context from prior steps as additional context
# - Handle failures: retry once, then skip and continue

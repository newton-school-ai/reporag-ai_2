"""Prompt builder with code-aware templates.

Builds LLM prompts with code context, citation format instructions, and
few-shot examples. Different templates for simple-lookup, multi-hop, and
exploratory query types.
"""

# TODO: Implement in Issue 24
# - Template per query type: simple-lookup, multi-hop, exploratory
# - System prompt: role, citation format [file:start-end], examples
# - Code context block from context_assembler
# - Sub-query answers injected for multi-hop (from prior steps)
# - Total prompt token count validation against model limit

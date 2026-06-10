"""Context assembler.

Transforms raw retrieval results into a structured, deduplicated context
block for the LLM prompt. Orders chunks by file and line, merges overlaps,
and truncates to fit the context window.
"""

# TODO: Implement in Issue 23
# - Order chunks by file_path, then start_line
# - Deduplicate overlapping chunks (merge if overlap > 50%)
# - Format: "## file_path (lines N-M)\n```python\n...\n```"
# - Truncate to max_tokens, prioritizing highest-ranked chunks

"""Line-level citation extraction.

Parses LLM response text for citation markers [file:start_line-end_line],
validates each citation against the retrieved context, and returns
structured Citation objects.
"""

# TODO: Implement in Issue 25
# - Parse response for [file_path:start_line-end_line] markers
# - Validate each citation: does the file + line range exist in context?
# - Flag invalid citations (hallucinated file/line references)
# - Return: list of Citation(file_path, start_line, end_line, snippet, valid)
# - Compute citation coverage: cited_claims / total_claims

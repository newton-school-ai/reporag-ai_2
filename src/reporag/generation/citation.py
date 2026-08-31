"""Line-level citation extraction.

Parses LLM response text for citation markers [file:start_line-end_line],
validates each citation against the retrieved context, and returns
structured Citation objects.
"""

import re
from dataclasses import dataclass

from reporag.retrieval.vector_search import RetrievalResult


@dataclass
class Citation:
    """Represents a validated line-level citation from the LLM.

    Attributes:
        file_path (str): The referenced file path.
        start_line (int): The starting line number.
        end_line (int): The ending line number.
        snippet (str): The exact code snippet from the context if valid.
        valid (bool): True if the cited lines exist in the provided context chunks.
    """

    file_path: str
    start_line: int
    end_line: int
    snippet: str
    valid: bool


def compute_citation_coverage(text: str, citations: list[Citation]) -> float:
    """Compute citation coverage metric: cited_claims / total_claims.

    Args:
        text (str): The generated answer text.
        citations (List[Citation]): The list of extracted citations.

    Returns:
        float: Coverage score between 0.0 and 1.0.
    """
    if not text.strip():
        return 0.0

    # Remove markdown code blocks before counting claim sentences
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if s.strip() and not s.strip().startswith("#")
    ]

    if not sentences:
        return 1.0 if citations else 0.0

    pattern = re.compile(r"\[([^\]:]+):(\d+)-(\d+)\]")
    cited_count = sum(1 for sentence in sentences if pattern.search(sentence))

    return min(1.0, cited_count / len(sentences))


class CitationExtractor:
    """Extracts and validates line-level citations from generated text."""

    def __init__(self) -> None:
        """Initialize the citation extractor with regex patterns."""
        # Matches [path/to/file.py:10-25]
        self.pattern = re.compile(r"\[([^\]:]+):(\d+)-(\d+)\]")

    def extract(
        self, text: str, context_chunks: list[RetrievalResult]
    ) -> list[Citation]:
        """Extract citations from text and validate against context.

        Args:
            text (str): The raw response text from the LLM.
            context_chunks (List[RetrievalResult]): The chunks provided in the prompt.

        Returns:
            List[Citation]: A list of extracted and validated citations.
        """
        citations = []
        matches = self.pattern.findall(text)

        # Build a lookup for easier validation
        # Map file_path -> list of (start, end, chunk_text)
        context_lookup = {}
        for chunk in context_chunks:
            if chunk.file_path not in context_lookup:
                context_lookup[chunk.file_path] = []

            start = chunk.start_line if chunk.start_line is not None else -1
            end = chunk.end_line if chunk.end_line is not None else float("inf")
            context_lookup[chunk.file_path].append((start, end, chunk))

        for file_path, start_str, end_str in matches:
            raw_start = int(start_str)
            raw_end = int(end_str)
            start_line = min(raw_start, raw_end)
            end_line = max(raw_start, raw_end)

            valid = False
            snippet = ""

            if file_path in context_lookup:
                # Check if lines intersect with any chunk
                for c_start, c_end, chunk in context_lookup[file_path]:
                    if c_start == -1 or max(start_line, c_start) <= min(
                        end_line, c_end
                    ):
                        valid = True

                        if chunk.start_line is not None and chunk.end_line is not None:
                            lines = chunk.chunk_text.split("\n")
                            rel_start = max(0, start_line - chunk.start_line)
                            rel_end = min(len(lines), end_line - chunk.start_line + 1)
                            snippet_lines = lines[rel_start:rel_end]
                            snippet = "\n".join(snippet_lines)
                        else:
                            snippet = chunk.chunk_text
                        break

            citations.append(
                Citation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    snippet=snippet,
                    valid=valid,
                )
            )

        return citations

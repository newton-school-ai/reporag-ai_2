"""Prompt builder with code-aware templates.

Builds LLM prompts with code context, citation format instructions, and
few-shot examples. Different templates for simple-lookup, multi-hop, and
exploratory query types.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Query type type definition
QueryType = Literal["simple-lookup", "multi-hop", "exploratory"]
_VALID_QUERY_TYPES: frozenset[str] = frozenset(
    {"simple-lookup", "multi-hop", "exploratory"}
)

# Default token budget for full prompt
_DEFAULT_MAX_TOKENS = 8000
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class BuiltPrompt:
    """The result of building a prompt for the LLM.

    Attributes:
        system_prompt: System prompt instructions and citation rules.
        user_prompt: Formatted user prompt containing context, sub-query answers, and query.
        full_prompt: Complete concatenated prompt text ready for token counting / inspection.
        query_type: The query type classification used for template selection.
        total_tokens: Approximate token count of the full prompt.
        truncated: True if context was truncated to fit within max_tokens.
    """

    system_prompt: str
    user_prompt: str
    full_prompt: str
    query_type: QueryType
    total_tokens: int
    truncated: bool


def _approximate_tokens(text: str) -> int:
    """Approximate the token count of a string using char-per-token ratio."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Citation Instructions & Few-Shot Templates
# ---------------------------------------------------------------------------

_CITATION_INSTRUCTIONS = """
CITATION INSTRUCTION:
You MUST cite your sources using exact line-level citation markers in the format [file_path:start_line-end_line].
Example: "The authentication function validates credentials [src/auth.py:10-25]."
Do not make statements about code without citing the file and line range where the implementation resides.
"""

_SIMPLE_LOOKUP_SYSTEM_PROMPT = f"""You are an expert codebase assistant specializing in direct code lookup and navigation.
Your task is to answer the user's question concisely and accurately based on the provided code context.

{_CITATION_INSTRUCTIONS}
Be direct and precise. Identify the exact file, symbol, class, or function definition requested.
"""

_MULTI_HOP_SYSTEM_PROMPT = f"""You are an expert codebase assistant specializing in multi-step code reasoning and component tracing.
Your task is to synthesize information across multiple files and components to answer complex queries.

{_CITATION_INSTRUCTIONS}

FEW-SHOT EXAMPLE:
User Query: How does user login flow connect to the database session?
Step-by-Step Reasoning:
1. The login endpoint is handled by `login_user` [src/api/auth.py:15-30].
2. `login_user` authenticates credentials via `verify_password` [src/auth/crypto.py:40-52].
3. Upon validation, it requests an async DB session from `get_db` [src/db/session.py:12-25] to fetch the user record [src/db/models.py:100-120].
Answer:
The login flow starts at `login_user` [src/api/auth.py:15-30], which validates credentials [src/auth/crypto.py:40-52] and acquires a database session via `get_db` [src/db/session.py:12-25] to retrieve user attributes [src/db/models.py:100-120].
"""

_EXPLORATORY_SYSTEM_PROMPT = f"""You are an expert codebase architect assistant specializing in codebase exploration and architectural overviews.
Your task is to provide a comprehensive structural overview, detailing key components, patterns, and data flow.

{_CITATION_INSTRUCTIONS}

FEW-SHOT EXAMPLE:
User Query: Explain the architecture of the vector search pipeline.
Answer:
The vector search pipeline consists of three primary layers:
1. **Embedding Generation**: Source code files are parsed and converted into embedding vectors using `CodeEmbedder` [src/embedding/code_embedder.py:15-45].
2. **Vector Storage**: Embeddings are indexed into Qdrant collections managed by `IndexBuilder` [src/embedding/index_builder.py:30-80].
3. **Similarity Retrieval**: Queries are executed against Qdrant via `VectorSearch.search` [src/retrieval/vector_search.py:50-95], returning scored `RetrievalResult` items.
"""


class PromptBuilder:
    """Builds LLM prompts with code context, citation format instructions, and templates.

    Args:
        max_tokens: Maximum token budget for the assembled prompt (default: 8000).
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}")
        self.max_tokens = max_tokens

    def build(
        self,
        query: str,
        query_type: QueryType = "simple-lookup",
        context: Any | str | None = None,
        sub_query_answers: Sequence[str | dict[str, Any]] | None = None,
    ) -> BuiltPrompt:
        """Build a structured LLM prompt for the specified query and context.

        Args:
            query: The user's input question or prompt.
            query_type: One of 'simple-lookup', 'multi-hop', or 'exploratory'.
            context: Code context block (assembled string, object with .text, or None).
            sub_query_answers: Optional step answers from prior decomposition steps (for multi-hop).

        Returns:
            A :class:`BuiltPrompt` containing system prompt, user prompt, and total token metadata.

        Raises:
            ValueError: If query_type is not one of 'simple-lookup', 'multi-hop', 'exploratory'.
        """
        if query_type not in _VALID_QUERY_TYPES:
            raise ValueError(
                f"Invalid query_type {query_type!r}. Must be one of {sorted(_VALID_QUERY_TYPES)}"
            )

        # Select system prompt template
        if query_type == "simple-lookup":
            system_prompt = _SIMPLE_LOOKUP_SYSTEM_PROMPT
        elif query_type == "multi-hop":
            system_prompt = _MULTI_HOP_SYSTEM_PROMPT
        else:
            system_prompt = _EXPLORATORY_SYSTEM_PROMPT

        # Extract context text
        context_text_attr = getattr(context, "text", None)
        if isinstance(context_text_attr, str):
            context_text = context_text_attr
        elif isinstance(context, str):
            context_text = context
        else:
            context_text = ""

        # Format sub-query answers if provided
        sub_answers_text = ""
        if sub_query_answers:
            formatted_steps: list[str] = []
            for idx, item in enumerate(sub_query_answers, 1):
                if isinstance(item, dict):
                    ans = item.get("answer", str(item))
                    sq = item.get("sub_query", "")
                    step_str = (
                        f"Step {idx}: {sq}\nAnswer: {ans}"
                        if sq
                        else f"Step {idx}: {ans}"
                    )
                else:
                    step_str = f"Step {idx}: {item}"
                formatted_steps.append(step_str)
            sub_answers_text = "\n\n".join(formatted_steps)

        # Build user prompt components
        user_prompt_sections: list[str] = []

        if context_text.strip():
            user_prompt_sections.append(
                f"--- CODE CONTEXT ---\n{context_text.strip()}\n--- END CONTEXT ---"
            )

        if sub_answers_text.strip():
            user_prompt_sections.append(
                f"--- PRIOR STEP ANSWERS ---\n{sub_answers_text.strip()}\n--- END PRIOR ANSWERS ---"
            )

        user_prompt_sections.append(f"Query: {query.strip()}")
        user_prompt = "\n\n".join(user_prompt_sections)

        # Combine into full prompt
        full_prompt = (
            f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{user_prompt.strip()}"
        )
        total_tokens = _approximate_tokens(full_prompt)
        truncated = False

        # Truncate context if budget exceeded
        if total_tokens > self.max_tokens and context_text.strip():
            logger.warning(
                "Prompt tokens (%d) exceed max_tokens (%d). Truncating context.",
                total_tokens,
                self.max_tokens,
            )

            # Available budget for context text including wrapper headers
            wrapper_overhead = _approximate_tokens(
                "\n--- CODE CONTEXT ---\n\n--- END CONTEXT ---"
            )
            overhead_text = f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n" + (
                "\n\n".join(
                    [
                        s
                        for s in user_prompt_sections
                        if not s.startswith("--- CODE CONTEXT ---")
                    ]
                )
            )
            overhead_tokens = _approximate_tokens(overhead_text) + wrapper_overhead
            available_context_tokens = max(0, self.max_tokens - overhead_tokens)
            available_chars = available_context_tokens * _CHARS_PER_TOKEN

            while available_chars > 0:
                truncated_context_text = context_text[:available_chars].rstrip()
                if truncated_context_text:
                    user_prompt_sections[0] = (
                        f"--- CODE CONTEXT ---\n{truncated_context_text}\n--- END CONTEXT ---"
                    )
                else:
                    user_prompt_sections.pop(0)

                user_prompt = "\n\n".join(user_prompt_sections)
                full_prompt = (
                    f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{user_prompt.strip()}"
                )
                total_tokens = _approximate_tokens(full_prompt)
                if total_tokens <= self.max_tokens:
                    break
                available_chars -= 10

            if (
                available_chars <= 0
                and user_prompt_sections
                and user_prompt_sections[0].startswith("--- CODE CONTEXT ---")
            ):
                user_prompt_sections.pop(0)
                user_prompt = "\n\n".join(user_prompt_sections)
                full_prompt = (
                    f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{user_prompt.strip()}"
                )
                total_tokens = _approximate_tokens(full_prompt)

            truncated = True

        return BuiltPrompt(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt.strip(),
            full_prompt=full_prompt.strip(),
            query_type=query_type,
            total_tokens=total_tokens,
            truncated=truncated,
        )

"""Prompt builder with code-aware templates.

Builds LLM prompts with code context, citation format instructions, and
few-shot examples. Different templates for simple-lookup, multi-hop, and
exploratory query types.
"""

from typing import Literal

from reporag.ingestion.chunker import count_tokens


class PromptTooLargeError(ValueError):
    """Raised when the constructed prompt exceeds the maximum token limit."""


class PromptBuilder:
    """Builds prompts for the LLM based on query type, context, and sub-queries."""

    # Base instructions included in all templates
    SYSTEM_INSTRUCTIONS = (
        "You are an expert software engineer answering questions about a codebase.\n"
        "Use the provided code context to answer the user's query.\n"
        "IMPORTANT CITATION RULES:\n"
        "You MUST cite your sources using the format [file_path:start_line-end_line].\n"
        "For example: 'The auth flow is handled in [src/auth.py:10-25].'\n"
        "Do NOT use markdown links for citations, only the exact bracket format above."
    )

    SIMPLE_LOOKUP_TEMPLATE = (
        "Answer directly and concisely.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )

    MULTI_HOP_TEMPLATE = (
        "This is a complex query that requires combining multiple pieces of information.\n"
        "Use the sub-query answers to help synthesize your final answer.\n\n"
        "Example of a good cited answer:\n"
        "The login flow starts in [api/routes.py:5-15] which calls the authentication service.\n"
        "The service then verifies the token using the secret key in [config/settings.py:20-22].\n\n"
        "Context:\n{context}\n\n"
        "Sub-query answers:\n{sub_query_answers}\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )

    EXPLORATORY_TEMPLATE = (
        "This is an exploratory query. Provide a comprehensive overview and explain how different parts interact.\n\n"
        "Example of a good cited answer:\n"
        "The architecture consists of three main layers:\n"
        "1. API Layer: Handled by [api/main.py:10-50], this routes requests.\n"
        "2. Business Logic: Found in [core/services.py:5-100], this processes data.\n"
        "3. Database: Managed by [db/models.py:1-40], this stores entities.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )

    def __init__(self, max_tokens: int = 100_000) -> None:
        """Initialize the prompt builder.

        Args:
            max_tokens: Maximum allowed tokens for the fully constructed prompt.
        """
        self.max_tokens = max_tokens

    def build(
        self,
        query: str,
        query_type: Literal["simple-lookup", "multi-hop", "exploratory"],
        context: str,
        sub_query_answers: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Build the structured messages for the LLM.

        Args:
            query: The user's query.
            query_type: One of 'simple-lookup', 'multi-hop', 'exploratory'.
            context: The assembled context blocks.
            sub_query_answers: Optional list of previous sub-query answers (for multi-hop).

        Returns:
            A list of structured messages (system and user) ready for chat-based LLMs.

        Raises:
            ValueError: If the query_type is invalid.
            PromptTooLargeError: If the final prompt exceeds max_tokens.
        """
        if not context:
            context = "No context found."

        if sub_query_answers:
            sub_answers_str = "\n".join(f"- {ans}" for ans in sub_query_answers)
        else:
            sub_answers_str = "None provided."

        if query_type == "simple-lookup":
            user_prompt = self.SIMPLE_LOOKUP_TEMPLATE.format(
                context=context, query=query
            )
        elif query_type == "multi-hop":
            user_prompt = self.MULTI_HOP_TEMPLATE.format(
                context=context,
                sub_query_answers=sub_answers_str,
                query=query,
            )
        elif query_type == "exploratory":
            user_prompt = self.EXPLORATORY_TEMPLATE.format(context=context, query=query)
        else:
            raise ValueError(f"Invalid query_type: '{query_type}'")

        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

        # Calculate total tokens including rough overhead for message structure
        # (approx 4 tokens per message for formatting)
        total_tokens = sum(count_tokens(msg["content"]) + 4 for msg in messages)

        if total_tokens > self.max_tokens:
            raise PromptTooLargeError(
                f"Constructed prompt size ({total_tokens} tokens) exceeds "
                f"maximum allowed context window ({self.max_tokens} tokens)."
            )

        return messages

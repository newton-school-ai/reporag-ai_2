"""Unit tests for AnswerGenerator (Issue 25)."""

from __future__ import annotations

import pytest

from reporag.generation.generator import AnswerGenerator, GenerationResult
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.vector_search import RetrievalResult


def test_generator_init_defaults() -> None:
    gen = AnswerGenerator()
    assert gen.provider in ("openai", "anthropic")
    assert gen.model is not None


def test_generator_init_custom() -> None:
    gen = AnswerGenerator(provider="anthropic", model="claude-3-5-sonnet", timeout=10.0)
    assert gen.provider == "anthropic"
    assert gen.model == "claude-3-5-sonnet"
    assert gen.timeout == 10.0


def test_generator_invalid_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        AnswerGenerator(provider="invalid_llm")


def test_generator_with_mock_client() -> None:
    def mock_llm(prompt: str) -> str:
        return "The login flow authenticates credentials [src/auth.py:10-20]."

    context = [
        RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
            symbol_name=None,
            chunk_text="def login(): pass",
            metadata={},
        )
    ]

    gen = AnswerGenerator(provider="openai", model="gpt-4o", llm_client=mock_llm)
    result = gen.generate("How does login work?", context_chunks=context)

    assert isinstance(result, GenerationResult)
    assert "login flow authenticates" in result.answer_text
    assert len(result.citations) == 1
    assert result.citations[0].file_path == "src/auth.py"
    assert result.citations[0].valid is True
    assert result.coverage == 1.0
    assert result.provider == "openai"
    assert result.model == "gpt-4o"


def test_generator_error_handling() -> None:
    def mock_llm_error(prompt: str) -> str:
        raise TimeoutError("API request timed out after 30s")

    gen = AnswerGenerator(provider="openai", llm_client=mock_llm_error)
    result = gen.generate("Test prompt")

    assert "Error generating answer" in result.answer_text
    assert "timed out" in result.answer_text
    assert result.coverage == 0.0
    assert len(result.citations) == 0


def test_generator_with_built_prompt() -> None:
    builder = PromptBuilder()
    prompt = builder.build(query="Find auth function", query_type="simple-lookup")

    def mock_llm(prompt_str: str) -> str:
        assert "Find auth function" in prompt_str
        return "Found auth function [src/auth.py:5-10]."

    gen = AnswerGenerator(llm_client=mock_llm)
    result = gen.generate(prompt)

    assert "Found auth function" in result.answer_text

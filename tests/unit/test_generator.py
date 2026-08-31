"""Unit tests for answer generator."""

from unittest.mock import MagicMock, patch

import pytest
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
from httpx import Request, Response
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from reporag.generation.generator import AnswerGenerator, GenerationResult
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.vector_search import RetrievalResult


def make_chunk(file_path: str, start: int, end: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        score=0.9,
        file_path=file_path,
        start_line=start,
        end_line=end,
        symbol_name=None,
        chunk_text=text,
        metadata={},
    )


# ---------------------------------------------------------------------------
# OpenAI generation tests
# ---------------------------------------------------------------------------


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_success(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Here is an answer [src/test.py:1-5]."
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    chunks = [
        make_chunk("src/test.py", 1, 10, "def test():\n    pass\n    return True\n")
    ]

    result = gen.generate("What is the test?", chunks)
    assert isinstance(result, GenerationResult)
    assert result.answer_text == "Here is an answer [src/test.py:1-5]."
    assert len(result.citations) == 1
    assert result.citations[0].valid is True
    assert result.citations[0].file_path == "src/test.py"

    # Verify parameters sent to OpenAI client
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["timeout"] == 60.0


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_with_built_prompt(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer text [src/test.py:1-2]."
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    prompt = PromptBuilder().build_prompt("Where is login?")
    chunks = [make_chunk("src/test.py", 1, 10, "def test(): pass")]

    result = gen.generate(prompt, chunks)
    assert result.answer_text == "Answer text [src/test.py:1-2]."
    assert len(result.citations) == 1

    # Check messages structure includes system and user roles
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_connection_error(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAIConnectionError(
        request=Request("GET", "http://test")
    )
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    with pytest.raises(RuntimeError, match="Connection error to openai API"):
        gen.generate("Test prompt", [])


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_timeout_error(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAITimeoutError(
        request=Request("GET", "http://test")
    )
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    with pytest.raises(RuntimeError, match="Timeout connecting to openai API"):
        gen.generate("Test prompt", [])


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_rate_limit_error(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
        message="Rate limit exceeded",
        response=Response(status_code=429, request=Request("GET", "http://test")),
        body=None,
    )
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    with pytest.raises(RuntimeError, match="Rate limit exceeded for openai API"):
        gen.generate("Test prompt", [])


@patch("reporag.generation.generator.OpenAI")
def test_generator_openai_unexpected_error(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ValueError(
        "Corrupt response payload"
    )
    mock_openai.return_value = mock_client

    gen = AnswerGenerator(provider="openai", model="gpt-4o")
    with pytest.raises(RuntimeError, match="Unexpected error calling openai API"):
        gen.generate("Test prompt", [])


# ---------------------------------------------------------------------------
# Anthropic generation tests
# ---------------------------------------------------------------------------


@patch("reporag.generation.generator.Anthropic")
def test_generator_anthropic_success_with_built_prompt(mock_anthropic):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Anthropic answer [src/test.py:1-2]."
    mock_response.content = [mock_text_block]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    gen = AnswerGenerator(provider="anthropic", model="claude-3-5-sonnet")
    prompt = PromptBuilder().build_prompt("Explain auth architecture")
    chunks = [make_chunk("src/test.py", 1, 10, "def test(): pass")]

    result = gen.generate(prompt, chunks)
    assert result.answer_text == "Anthropic answer [src/test.py:1-2]."
    assert len(result.citations) == 1
    assert result.citations[0].valid is True

    # Verify anthropic payload parameters
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-3-5-sonnet"
    assert call_kwargs["max_tokens"] == 4096
    assert "system" in call_kwargs


@patch("reporag.generation.generator.Anthropic")
def test_generator_anthropic_connection_error(mock_anthropic):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = AnthropicConnectionError(
        request=Request("GET", "http://test")
    )
    mock_anthropic.return_value = mock_client

    gen = AnswerGenerator(provider="anthropic", model="claude-3-5-sonnet")
    with pytest.raises(RuntimeError, match="Connection error to anthropic API"):
        gen.generate("Test prompt", [])


@patch("reporag.generation.generator.Anthropic")
def test_generator_anthropic_timeout_error(mock_anthropic):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = AnthropicTimeoutError(
        request=Request("GET", "http://test")
    )
    mock_anthropic.return_value = mock_client

    gen = AnswerGenerator(provider="anthropic", model="claude-3-5-sonnet")
    with pytest.raises(RuntimeError, match="Timeout connecting to anthropic API"):
        gen.generate("Test prompt", [])


@patch("reporag.generation.generator.Anthropic")
def test_generator_anthropic_rate_limit_error(mock_anthropic):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = AnthropicRateLimitError(
        message="Anthropic rate limit",
        response=Response(status_code=429, request=Request("GET", "http://test")),
        body=None,
    )
    mock_anthropic.return_value = mock_client

    gen = AnswerGenerator(provider="anthropic", model="claude-3-5-sonnet")
    with pytest.raises(RuntimeError, match="Rate limit exceeded for anthropic API"):
        gen.generate("Test prompt", [])


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------


def test_generator_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider: cohere"):
        AnswerGenerator(provider="cohere")

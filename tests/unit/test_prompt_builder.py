import pytest

from reporag.generation.prompt_builder import PromptBuilder, PromptTooLargeError


def test_prompt_builder_init():
    builder = PromptBuilder()
    assert "file_path:start_line-end_line" in builder.SYSTEM_INSTRUCTIONS
    assert builder.max_tokens == 100_000


def test_simple_lookup_prompt():
    builder = PromptBuilder()
    prompt = builder.build(
        query="How does auth work?",
        query_type="simple-lookup",
        context="## src/auth.py\ndef login(): pass",
    )
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"

    assert "You are an expert software engineer" in prompt[0]["content"]
    assert "file_path:start_line-end_line" in prompt[0]["content"]

    assert "## src/auth.py" in prompt[1]["content"]
    assert "How does auth work?" in prompt[1]["content"]
    assert "Answer directly and concisely" in prompt[1]["content"]
    assert "Sub-query answers" not in prompt[1]["content"]


def test_multi_hop_prompt_with_sub_queries():
    builder = PromptBuilder()
    prompt = builder.build(
        query="What uses the database?",
        query_type="multi-hop",
        context="## src/db.py\nclass DB: pass",
        sub_query_answers=["The API uses it.", "The background jobs use it."],
    )
    assert len(prompt) == 2
    assert "file_path:start_line-end_line" in prompt[0]["content"]

    assert "This is a complex query" in prompt[1]["content"]
    assert "## src/db.py" in prompt[1]["content"]
    assert "What uses the database?" in prompt[1]["content"]
    assert "- The API uses it." in prompt[1]["content"]
    assert "- The background jobs use it." in prompt[1]["content"]
    # Check for few-shot example
    assert "The login flow starts in [api/routes.py:5-15]" in prompt[1]["content"]


def test_exploratory_prompt():
    builder = PromptBuilder()
    prompt = builder.build(
        query="Explain the architecture",
        query_type="exploratory",
        context="## src/main.py\nrun()",
    )
    assert len(prompt) == 2
    assert "file_path:start_line-end_line" in prompt[0]["content"]

    assert "This is an exploratory query" in prompt[1]["content"]
    assert "## src/main.py" in prompt[1]["content"]
    assert "Explain the architecture" in prompt[1]["content"]
    # Check for few-shot example
    assert "The architecture consists of three main layers" in prompt[1]["content"]


def test_empty_context():
    builder = PromptBuilder()
    prompt = builder.build(
        query="Where is the config?",
        query_type="simple-lookup",
        context="",
    )
    assert "No context found." in prompt[1]["content"]
    assert "Where is the config?" in prompt[1]["content"]


def test_invalid_query_type():
    builder = PromptBuilder()
    with pytest.raises(ValueError, match="Invalid query_type: 'unknown'"):
        builder.build(
            query="test",
            query_type="unknown",  # type: ignore[arg-type]
            context="context",
        )


def test_multi_hop_without_sub_queries():
    builder = PromptBuilder()
    prompt = builder.build(
        query="test",
        query_type="multi-hop",
        context="ctx",
    )
    assert "None provided." in prompt[1]["content"]


def test_prompt_exceeds_token_limit():
    # Set a small token limit to easily trigger the exception
    builder = PromptBuilder(max_tokens=50)
    with pytest.raises(
        PromptTooLargeError, match="exceeds maximum allowed context window"
    ):
        builder.build(
            query="This query and context will definitely exceed fifty tokens because the templates themselves are quite long and detailed.",
            query_type="simple-lookup",
            context="## src/foo.py\ndef long_function_name_that_adds_tokens():\n    pass",
        )


def test_prompt_fits_token_limit():
    # 500 is enough for the template + simple query, but small enough to verify we enforce limits
    builder = PromptBuilder(max_tokens=500)
    prompt = builder.build(
        query="short query",
        query_type="simple-lookup",
        context="short context",
    )
    assert len(prompt) == 2


def test_large_sub_query_answers_exceed_limit():
    builder = PromptBuilder(max_tokens=200)
    with pytest.raises(PromptTooLargeError):
        builder.build(
            query="test",
            query_type="multi-hop",
            context="ctx",
            sub_query_answers=["A very long answer " * 50],
        )

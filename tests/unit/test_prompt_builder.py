"""Unit tests for PromptBuilder (Issue 24)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from reporag.generation.prompt_builder import BuiltPrompt, PromptBuilder


@dataclass
class DummyContext:
    text: str


def test_prompt_builder_simple_lookup() -> None:
    builder = PromptBuilder()
    prompt = builder.build(
        query="Where is the authenticate function defined?",
        query_type="simple-lookup",
        context="## src/auth.py (lines 10-25)\n```python\ndef authenticate(): pass\n```",
    )

    assert isinstance(prompt, BuiltPrompt)
    assert prompt.query_type == "simple-lookup"
    assert "CITATION INSTRUCTION:" in prompt.system_prompt
    assert "[file_path:start_line-end_line]" in prompt.system_prompt
    assert "direct code lookup" in prompt.system_prompt
    assert "--- CODE CONTEXT ---" in prompt.user_prompt
    assert "src/auth.py" in prompt.user_prompt
    assert "Query: Where is the authenticate function defined?" in prompt.user_prompt
    assert not prompt.truncated
    assert prompt.total_tokens > 0


def test_prompt_builder_multi_hop_with_sub_query_answers() -> None:
    builder = PromptBuilder()
    sub_answers = [
        "Step 1: Auth module defined in src/auth.py [src/auth.py:1-20]",
        {
            "sub_query": "Where is DB session acquired?",
            "answer": "get_db in src/db.py [src/db.py:10-30]",
        },
    ]
    prompt = builder.build(
        query="How does login authenticate and connect to DB?",
        query_type="multi-hop",
        context="## src/auth.py (lines 1-20)\n```python\ndef login(): pass\n```",
        sub_query_answers=sub_answers,
    )

    assert prompt.query_type == "multi-hop"
    assert "FEW-SHOT EXAMPLE:" in prompt.system_prompt
    assert "multi-step code reasoning" in prompt.system_prompt
    assert "--- PRIOR STEP ANSWERS ---" in prompt.user_prompt
    assert "Step 1: Auth module defined in src/auth.py" in prompt.user_prompt
    assert "Step 2: Where is DB session acquired?" in prompt.user_prompt
    assert "Answer: get_db in src/db.py" in prompt.user_prompt
    assert "Query: How does login authenticate and connect to DB?" in prompt.user_prompt


def test_prompt_builder_exploratory() -> None:
    builder = PromptBuilder()
    prompt = builder.build(
        query="Explain the overall system architecture.",
        query_type="exploratory",
        context="## src/main.py (lines 1-50)\n```python\napp = FastAPI()\n```",
    )

    assert prompt.query_type == "exploratory"
    assert "FEW-SHOT EXAMPLE:" in prompt.system_prompt
    assert "codebase architect assistant" in prompt.system_prompt
    assert "--- CODE CONTEXT ---" in prompt.user_prompt
    assert "Query: Explain the overall system architecture." in prompt.user_prompt


def test_prompt_builder_object_context() -> None:
    builder = PromptBuilder()
    dummy = DummyContext(
        text="## src/models.py (lines 1-10)\n```python\nclass User: pass\n```"
    )
    prompt = builder.build(
        query="Show the User class",
        query_type="simple-lookup",
        context=dummy,
    )

    assert "class User: pass" in prompt.user_prompt


def test_prompt_builder_none_context() -> None:
    builder = PromptBuilder()
    prompt = builder.build(
        query="What is the project name?",
        query_type="simple-lookup",
        context=None,
    )

    assert "--- CODE CONTEXT ---" not in prompt.user_prompt
    assert "Query: What is the project name?" in prompt.user_prompt


def test_prompt_builder_truncation_on_budget_exceeded() -> None:
    # Use max_tokens smaller than full prompt (system prompt ~164 tokens + context 800 tokens)
    builder = PromptBuilder(max_tokens=250)
    large_context = (
        "## src/huge.py (lines 1-1000)\n```python\n" + ("x = 1\n" * 500) + "```"
    )
    prompt = builder.build(
        query="Summarize huge file",
        query_type="simple-lookup",
        context=large_context,
    )

    assert prompt.truncated
    assert prompt.total_tokens <= 250


def test_prompt_builder_invalid_query_type() -> None:
    builder = PromptBuilder()
    with pytest.raises(ValueError, match="Invalid query_type"):
        builder.build(
            query="Test query",
            query_type="invalid-type",  # type: ignore[arg-type]
        )


def test_prompt_builder_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        PromptBuilder(max_tokens=0)

"""Unit tests for planner module."""

import json
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from reporag.agent.planner import QueryClassifier, QueryType


def make_mock_llm(response_dict: dict) -> Callable[[str], str]:
    """Helper to create a mock LLM callable returning a JSON string."""

    def mock_llm(prompt: str) -> str:
        return json.dumps(response_dict)

    return mock_llm


def make_mock_llm_raw(response_str: str) -> Callable[[str], str]:
    """Helper to create a mock LLM callable returning exactly the given string."""

    def mock_llm(prompt: str) -> str:
        return response_str

    return mock_llm


class TestQueryClassifier:

    def test_empty_query(self) -> None:
        """Empty query should fast-path to MULTI_HOP without calling LLM."""
        mock_client = MagicMock()
        classifier = QueryClassifier(llm_client=mock_client)
        result = classifier.classify("   ")
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0
        mock_client.assert_not_called()

    @pytest.mark.parametrize(
        "query, expected_type, confidence",
        [
            ("Where is authenticate defined?", QueryType.SIMPLE_LOOKUP, 0.95),
            ("Where is X defined?", QueryType.SIMPLE_LOOKUP, 0.90),
            ("Show me the auth middleware.", QueryType.SIMPLE_LOOKUP, 0.85),
            ("How does auth work end-to-end?", QueryType.MULTI_HOP, 0.92),
            ("How does a request go from API to DB?", QueryType.MULTI_HOP, 0.92),
            ("What calls function Y?", QueryType.MULTI_HOP, 0.91),
            (
                "Explain the architecture of the ingestion pipeline.",
                QueryType.EXPLORATORY,
                0.95,
            ),
            ("What is the overall testing strategy?", QueryType.EXPLORATORY, 0.85),
            ("Explain the architecture", QueryType.EXPLORATORY, 0.90),
            ("Give me a high-level overview.", QueryType.EXPLORATORY, 0.80),
        ],
    )
    def test_acceptance_criteria_queries(
        self, query: str, expected_type: QueryType, confidence: float
    ) -> None:
        """Test the specific queries mentioned in the AC and prompt."""
        mock_client = make_mock_llm(
            {"query_type": expected_type.value, "confidence": confidence}
        )
        classifier = QueryClassifier(llm_client=mock_client)

        result = classifier.classify(query)
        assert result.query_type == expected_type
        assert result.confidence == confidence

    def test_low_confidence_fallback(self) -> None:
        """If confidence is below threshold, it must fallback to multi-hop."""
        mock_client = make_mock_llm({"query_type": "simple-lookup", "confidence": 0.45})
        classifier = QueryClassifier(llm_client=mock_client)

        result = classifier.classify("Tell me about the code")
        # Overridden by fallback
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0

    def test_invalid_json_fallback(self) -> None:
        """If the LLM returns invalid JSON, fallback safely."""
        mock_client = make_mock_llm_raw("This is not json.")
        classifier = QueryClassifier(llm_client=mock_client)

        result = classifier.classify("What is X?")
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0

    def test_hallucinated_label_fallback(self) -> None:
        """If the LLM returns a query_type not in the Enum, fallback safely."""
        mock_client = make_mock_llm({"query_type": "magic-lookup", "confidence": 0.99})
        classifier = QueryClassifier(llm_client=mock_client)

        result = classifier.classify("What is X?")
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0

    def test_missing_confidence_fallback(self) -> None:
        """If the LLM forgets the confidence field, fallback safely."""
        mock_client = make_mock_llm_raw('{"query_type": "simple-lookup"}')
        classifier = QueryClassifier(llm_client=mock_client)

        result = classifier.classify("What is X?")
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0

    def test_network_timeout_fallback(self) -> None:
        """If the LLM client raises an exception, fallback safely."""

        def timeout_client(prompt: str) -> str:
            raise TimeoutError("Connection timed out")

        classifier = QueryClassifier(llm_client=timeout_client)
        result = classifier.classify("What is X?")

        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.0

    @patch("openai.Client")
    def test_ensure_loaded_openai(self, mock_openai_client: MagicMock) -> None:
        """Ensure the classifier correctly initializes the OpenAI client when configured."""
        with patch("reporag.agent.planner.settings.llm_provider", "openai"):
            classifier = QueryClassifier()
            classifier._ensure_loaded()

            mock_openai_client.assert_called_once()
            assert classifier._llm_client is mock_openai_client.return_value

    @patch("anthropic.Anthropic")
    def test_ensure_loaded_anthropic(self, mock_anthropic_client: MagicMock) -> None:
        """Ensure the classifier correctly initializes the Anthropic client when configured."""
        with patch("reporag.agent.planner.settings.llm_provider", "anthropic"):
            classifier = QueryClassifier()
            classifier._ensure_loaded()

            mock_anthropic_client.assert_called_once()
            assert classifier._llm_client is mock_anthropic_client.return_value

    def test_unsupported_client_raises(self) -> None:
        """A misconfigured llm_client that is not callable/OpenAI/Anthropic must raise, not fall back."""
        classifier = QueryClassifier(llm_client=object())
        with pytest.raises(RuntimeError, match="Unsupported llm_client type"):
            classifier.classify("Where is X defined?")

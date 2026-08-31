"""LLM answer generator.

Calls the configured LLM (OpenAI or Anthropic) with the built prompt and
returns the raw response for citation extraction.
"""

from dataclasses import dataclass
from typing import Union

from anthropic import Anthropic
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError

from reporag.config import settings
from reporag.generation.citation import Citation, CitationExtractor
from reporag.retrieval.vector_search import RetrievalResult

try:
    from reporag.generation.prompt_builder import BuiltPrompt
except ImportError:
    BuiltPrompt = None


@dataclass
class GenerationResult:
    """The result of an LLM generation pass.

    Attributes:
        answer_text (str): The raw text response from the LLM.
        citations (List[Citation]): The citations extracted and validated from the answer.
    """

    answer_text: str
    citations: list[Citation]


class AnswerGenerator:
    """Generates answers using an LLM and extracts citations."""

    def __init__(self, provider: str = None, model: str = None) -> None:
        """Initialize the AnswerGenerator.

        Args:
            provider (str, optional): The LLM provider (openai or anthropic). Defaults to settings.llm_provider.
            model (str, optional): The model name. Defaults to settings active model.
        """
        self.provider = provider or settings.llm_provider
        self.model = model or (
            settings.openai_model
            if self.provider == "openai"
            else settings.anthropic_model
        )
        self.extractor = CitationExtractor()

        api_key = settings.active_llm_api_key.get_secret_value()

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self, prompt: Union[str, "BuiltPrompt"], context_chunks: list[RetrievalResult]
    ) -> GenerationResult:
        """Generate an answer from the LLM and extract citations.

        Args:
            prompt (Union[str, BuiltPrompt]): The prompt to send to the LLM.
            context_chunks (List[RetrievalResult]): The retrieved chunks used in the prompt.

        Returns:
            GenerationResult: The generated answer and validated citations.

        Raises:
            RuntimeError: If there are connection errors, timeouts, or rate limits.
        """
        if hasattr(prompt, "system") and hasattr(prompt, "user"):
            system_prompt = prompt.system
            user_prompt = prompt.user
        elif hasattr(prompt, "system") and hasattr(prompt, "text"):
            system_prompt = prompt.system
            user_prompt = prompt.text
        else:
            system_prompt = ""
            user_prompt = str(prompt)

        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, timeout=60.0
                )
                answer_text = response.choices[0].message.content

            elif self.provider == "anthropic":
                messages = [{"role": "user", "content": user_prompt}]
                kwargs = {
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": messages,
                    "timeout": 60.0,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self.client.messages.create(**kwargs)
                answer_text = response.content[0].text

        except (OpenAITimeoutError, AnthropicTimeoutError) as e:
            raise RuntimeError(
                f"Timeout connecting to {self.provider} API: {str(e)}"
            ) from e
        except (OpenAIRateLimitError, AnthropicRateLimitError) as e:
            raise RuntimeError(
                f"Rate limit exceeded for {self.provider} API: {str(e)}"
            ) from e
        except (OpenAIConnectionError, AnthropicConnectionError) as e:
            raise RuntimeError(
                f"Connection error to {self.provider} API: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error calling {self.provider} API: {str(e)}"
            ) from e

        citations = self.extractor.extract(answer_text, context_chunks)

        return GenerationResult(answer_text=answer_text, citations=citations)

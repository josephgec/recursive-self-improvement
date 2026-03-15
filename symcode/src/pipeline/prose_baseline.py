"""Prose chain-of-thought baseline solver."""

from __future__ import annotations

from typing import Any

from src.pipeline.answer_extractor import AnswerExtractor, ExtractedAnswer
from src.pipeline.prompts import PromptManager
from src.utils.logging import get_logger

logger = get_logger("prose_baseline")


class ProseBaseline:
    """Solve math problems using prose chain-of-thought reasoning.

    Single-shot: no retries.  Extracts answer from \\boxed{} in the
    LLM response.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        prompt_manager: PromptManager | None = None,
        mock: bool = False,
    ):
        self.config = config or {}
        self.prompt_manager = prompt_manager or PromptManager()
        self.extractor = AnswerExtractor()
        self.mock = mock

        model_cfg = self.config.get("model", {})
        self.provider = "mock" if mock else model_cfg.get("provider", "openai")
        self.model_name = model_cfg.get("name", "gpt-4o")
        self.temperature = model_cfg.get("temperature", 0.0)
        self.max_tokens = model_cfg.get("max_tokens", 4096)

        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        model_cfg = self.config.get("model", {})
        if self.provider == "openai":
            import openai
            import os
            api_key = os.environ.get(
                model_cfg.get("api_key_env", "OPENAI_API_KEY"), ""
            )
            self._client = openai.OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            import anthropic
            import os
            api_key = os.environ.get(
                model_cfg.get("api_key_env", "ANTHROPIC_API_KEY"), ""
            )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call LLM and return text content."""
        if self.provider == "mock" or self.mock:
            return self._mock_response(messages)

        if self.provider == "openai":
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        if self.provider == "anthropic":
            client = self._get_client()
            system_msg = ""
            chat_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg += m["content"] + "\n"
                else:
                    chat_messages.append(m)
            response = client.messages.create(
                model=self.model_name,
                system=system_msg.strip(),
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.provider}")

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        """Generate a mock prose response for testing."""
        return (
            "Let me solve this step by step.\n\n"
            "Working through the problem...\n\n"
            "The answer is \\boxed{42}."
        )

    def solve(self, problem: str) -> tuple[str | None, str]:
        """Solve a problem using prose reasoning.

        Returns:
            (answer_string_or_None, full_response_text)
        """
        messages = self.prompt_manager.build_prose_prompt(problem)
        response = self._call_llm(messages)

        extracted = self.extractor.extract_from_prose(response)
        if extracted is not None:
            return extracted.normalized, response

        return None, response

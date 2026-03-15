"""LLM code generation for SymCode pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.pipeline.output_parser import CodeBlockParser
from src.pipeline.prompts import PromptManager
from src.pipeline.router import TaskType
from src.utils.logging import get_logger

logger = get_logger("code_generator")


@dataclass
class GenerationResult:
    """Result of a code generation call."""

    code: str
    raw_response: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class LLMClient(Protocol):
    """Protocol for LLM clients (OpenAI-compatible)."""

    def chat_completions_create(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]: ...


def _make_openai_client(config: dict[str, Any]) -> Any:
    """Create an OpenAI client from config."""
    import openai

    api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
    return openai.OpenAI(api_key=api_key)


def _make_anthropic_client(config: dict[str, Any]) -> Any:
    """Create an Anthropic client from config."""
    import anthropic

    api_key = os.environ.get(config.get("api_key_env", "ANTHROPIC_API_KEY"), "")
    return anthropic.Anthropic(api_key=api_key)


def _call_openai(
    client: Any,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Call OpenAI chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    return {
        "content": choice.message.content,
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
    }


def _call_anthropic(
    client: Any,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Call Anthropic messages API."""
    # Separate system from user/assistant messages
    system_msg = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg += m["content"] + "\n"
        else:
            chat_messages.append(m)

    response = client.messages.create(
        model=model,
        system=system_msg.strip(),
        messages=chat_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "content": response.content[0].text,
        "prompt_tokens": getattr(response.usage, "input_tokens", 0),
        "completion_tokens": getattr(response.usage, "output_tokens", 0),
    }


def _call_mock(
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Mock LLM for testing -- returns a simple SymPy program."""
    # Extract the problem from the last user message
    problem = ""
    for m in reversed(messages):
        if m["role"] == "user":
            problem = m["content"]
            break

    code = (
        "from sympy import *\n\n"
        "# Mock solution\n"
        "answer = 42\n"
        'print(f"Answer: {answer}")\n'
    )
    return {
        "content": f"```python\n{code}```",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


class SymCodeGenerator:
    """Generate SymPy code solutions via LLM."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        prompt_manager: PromptManager | None = None,
        cache_dir: str | Path | None = None,
        mock: bool = False,
    ):
        self.config = config or {}
        self.prompt_manager = prompt_manager or PromptManager()
        self.parser = CodeBlockParser()
        self.mock = mock

        model_cfg = self.config.get("model", {})
        self.provider = "mock" if mock else model_cfg.get("provider", "openai")
        self.model_name = model_cfg.get("name", "gpt-4o")
        self.temperature = model_cfg.get("temperature", 0.0)
        self.max_tokens = model_cfg.get("max_tokens", 4096)

        # Disk cache
        if cache_dir is None:
            cache_dir = Path(self.config.get("project", {}).get(
                "output_dir", "data/results"
            )) / "generation_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # LLM client (lazy init)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        model_cfg = self.config.get("model", {})
        if self.provider == "openai":
            self._client = _make_openai_client(model_cfg)
        elif self.provider == "anthropic":
            self._client = _make_anthropic_client(model_cfg)
        return self._client

    # ── caching ─────────────────────────────────────────────────────

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        blob = json.dumps(messages, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def _cache_get(self, key: str) -> GenerationResult | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return GenerationResult(
                code=data["code"],
                raw_response=data["raw_response"],
                model=data.get("model", ""),
                cached=True,
            )
        return None

    def _cache_put(self, key: str, result: GenerationResult) -> None:
        path = self.cache_dir / f"{key}.json"
        data = {
            "code": result.code,
            "raw_response": result.raw_response,
            "model": result.model,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── generation ──────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Dispatch LLM call based on provider."""
        if self.provider == "mock" or self.mock:
            return _call_mock(messages)
        elif self.provider == "openai":
            return _call_openai(
                self._get_client(),
                messages,
                self.model_name,
                self.temperature,
                self.max_tokens,
            )
        elif self.provider == "anthropic":
            return _call_anthropic(
                self._get_client(),
                messages,
                self.model_name,
                self.temperature,
                self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        problem: str,
        task_type: TaskType | None = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Generate SymPy code for a math problem.

        Args:
            problem: The math problem text.
            task_type: Optional task type for few-shot selection.
            use_cache: Whether to check/store in disk cache.

        Returns:
            GenerationResult with extracted code.
        """
        messages = self.prompt_manager.build_symcode_prompt(problem, task_type)

        # Check cache
        if use_cache:
            key = self._cache_key(messages)
            cached = self._cache_get(key)
            if cached is not None:
                logger.info("Cache hit for generation")
                return cached

        # Call LLM
        response = self._call_llm(messages)
        raw = response["content"]

        # Parse code from response
        code = self.parser.parse(raw)

        result = GenerationResult(
            code=code,
            raw_response=raw,
            model=self.model_name,
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
        )

        # Store in cache
        if use_cache:
            self._cache_put(key, result)

        return result

    def generate_correction(
        self,
        problem: str,
        prev_code: str,
        feedback: str,
        attempt: int = 1,
        max_attempts: int = 3,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Generate a corrected version of previously failed code.

        Args:
            problem: Original problem text.
            prev_code: The code that failed.
            feedback: Structured error feedback.
            attempt: Current attempt number.
            max_attempts: Maximum number of attempts.
            use_cache: Whether to check/store in disk cache.

        Returns:
            GenerationResult with corrected code.
        """
        messages = self.prompt_manager.build_retry_prompt(
            problem, prev_code, feedback, attempt, max_attempts
        )

        if use_cache:
            key = self._cache_key(messages)
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        response = self._call_llm(messages)
        raw = response["content"]
        code = self.parser.parse(raw)

        result = GenerationResult(
            code=code,
            raw_response=raw,
            model=self.model_name,
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
        )

        if use_cache:
            self._cache_put(key, result)

        return result

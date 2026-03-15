"""Task execution with LLM integration."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Task:
    """A task for the agent to solve."""

    task_id: str = ""
    question: str = ""
    expected_answer: str = ""
    domain: str = "general"
    category: str = ""
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from executing a task."""

    task: Task
    response: str = ""
    extracted_answer: str = ""
    correct: bool = False
    reasoning_mode: str = "cot"
    latency: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str: ...


class MockLLMClient:
    """Mock LLM client for testing without API keys."""

    def __init__(self, default_response: str = "") -> None:
        self._responses: list[str] = []
        self._default_response = default_response
        self._call_count = 0
        self._call_log: list[dict[str, Any]] = []

    def add_response(self, response: str) -> None:
        self._responses.append(response)

    def add_responses(self, responses: list[str]) -> None:
        self._responses.extend(responses)

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        self._call_log.append({"prompt": prompt, "system_prompt": system_prompt, **kwargs})
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        if self._default_response:
            return self._default_response
        # Try to extract a plausible answer from the prompt
        return self._auto_respond(prompt)

    def _auto_respond(self, prompt: str) -> str:
        """Generate a plausible mock response."""
        # For math-like questions, try to extract numbers
        if any(kw in prompt.lower() for kw in ["solve", "calculate", "compute", "what is", "find"]):
            return "Let me think step by step.\n\nThe answer is 42."
        if "deliberat" in prompt.lower() or "modif" in prompt.lower():
            return (
                '{"action": "modify", "target": "prompt_strategy", '
                '"description": "Improve system prompt for clarity", '
                '"code": "def prepare_prompt(self, task, examples):\\n'
                "    return f'Solve: {task.question}'\\n\", "
                '"risk": "low", "rationale": "Simple prompt improvement"}'
            )
        return "I'll work through this step by step.\n\nThe answer is: unknown"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)


def create_llm_client(provider: str = "mock", **kwargs: Any) -> LLMClient:
    """Factory for LLM clients."""
    if provider == "mock":
        mock_kwargs = {k: v for k, v in kwargs.items() if k in ("default_response",)}
        return MockLLMClient(**mock_kwargs)
    elif provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()

            class OpenAIClient:
                def __init__(self, model: str = "gpt-4o") -> None:
                    self.model = model
                    self.client = client

                def generate(self, prompt: str, system_prompt: str = "", **kw: Any) -> str:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=kw.get("temperature", 0.7),
                        max_tokens=kw.get("max_tokens", 4096),
                    )
                    return resp.choices[0].message.content or ""

            return OpenAIClient(model=kwargs.get("model", "gpt-4o"))
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAI client: {e}")
    elif provider == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic()

            class AnthropicClient:
                def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
                    self.model = model
                    self.client = client

                def generate(self, prompt: str, system_prompt: str = "", **kw: Any) -> str:
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=kw.get("max_tokens", 4096),
                        system=system_prompt or "You are a helpful assistant.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.content[0].text

            return AnthropicClient(model=kwargs.get("model", "claude-sonnet-4-20250514"))
        except Exception as e:
            raise RuntimeError(f"Failed to create Anthropic client: {e}")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


class TaskExecutor:
    """Executes tasks using the meta-learning algorithm and LLM."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm = llm_client
        self.config = config or {}

    def execute(
        self,
        task: Task,
        system_prompt: str = "",
        few_shot_examples: list[dict[str, Any]] | None = None,
        reasoning_mode: str = "cot",
        history: list[TaskResult] | None = None,
    ) -> TaskResult:
        """Execute a single task."""
        start = time.time()
        try:
            prompt = self._build_prompt(task, few_shot_examples or [], reasoning_mode)
            response = self.llm.generate(prompt, system_prompt=system_prompt)
            extracted = self._extract_answer(response)
            correct = self._check_answer(extracted, task.expected_answer)
            latency = time.time() - start
            return TaskResult(
                task=task,
                response=response,
                extracted_answer=extracted,
                correct=correct,
                reasoning_mode=reasoning_mode,
                latency=latency,
            )
        except Exception as e:
            return TaskResult(
                task=task,
                error=str(e),
                latency=time.time() - start,
            )

    def execute_batch(
        self,
        tasks: list[Task],
        system_prompt: str = "",
        few_shot_examples: list[dict[str, Any]] | None = None,
        reasoning_mode: str = "cot",
    ) -> list[TaskResult]:
        """Execute a batch of tasks."""
        results: list[TaskResult] = []
        for task in tasks:
            result = self.execute(task, system_prompt, few_shot_examples, reasoning_mode)
            results.append(result)
        return results

    def execute_validation(
        self,
        tasks: list[Task],
        system_prompt: str = "",
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> list[TaskResult]:
        """Execute validation tasks (always use cot mode)."""
        return self.execute_batch(tasks, system_prompt, few_shot_examples, reasoning_mode="cot")

    def _build_prompt(
        self,
        task: Task,
        examples: list[dict[str, Any]],
        reasoning_mode: str,
    ) -> str:
        parts: list[str] = []

        if examples:
            parts.append("Here are some examples:")
            for ex in examples:
                parts.append(f"Q: {ex.get('question', '')}")
                parts.append(f"A: {ex.get('answer', '')}")
            parts.append("")

        if reasoning_mode == "cot":
            parts.append("Think step by step to solve this problem.")
        elif reasoning_mode == "code":
            parts.append("Write code to solve this problem, then give the answer.")
        elif reasoning_mode == "decompose":
            parts.append("Break this problem into smaller parts and solve each.")
        else:
            parts.append("Solve this problem directly.")

        parts.append(f"\nQuestion: {task.question}")
        parts.append("\nProvide your final answer after 'The answer is:'")

        return "\n".join(parts)

    def _extract_answer(self, response: str) -> str:
        """Extract the answer from an LLM response."""
        # Look for "The answer is: X" pattern
        patterns = [
            r"[Tt]he answer is:?\s*(.+?)(?:\n|$)",
            r"[Aa]nswer:?\s*(.+?)(?:\n|$)",
            r"= (.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip().rstrip(".")
        # Fallback: last line
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""

    def _check_answer(self, extracted: str, expected: str) -> bool:
        """Check if the extracted answer matches the expected answer."""
        if not extracted or not expected:
            return False
        # Normalize
        e1 = extracted.lower().strip().rstrip(".")
        e2 = expected.lower().strip().rstrip(".")
        if e1 == e2:
            return True
        # Try numeric comparison
        try:
            return abs(float(e1) - float(e2)) < 1e-6
        except (ValueError, TypeError):
            pass
        # Substring match
        return e2 in e1 or e1 in e2

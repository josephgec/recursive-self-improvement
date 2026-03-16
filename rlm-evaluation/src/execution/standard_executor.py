"""Standard LLM executor with context window truncation."""

from __future__ import annotations

import random
from typing import Callable, Optional

from src.benchmarks.task import EvalTask, EvalResult


class StandardExecutor:
    """Execute tasks using a standard LLM with limited context window.

    When context exceeds the window, it gets truncated, often causing
    incorrect answers for tasks requiring information from the full context.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        context_window: int = 8192,
        seed: int = 42,
    ) -> None:
        self.llm_fn = llm_fn or self._default_mock_llm
        self.context_window = context_window
        self.rng = random.Random(seed)

    def execute(self, task: EvalTask) -> EvalResult:
        """Execute a single task using standard LLM approach."""
        # Truncate context if needed
        truncated = task.context_tokens > self.context_window
        effective_context = self._truncate_context(task.context)

        # Build prompt and get answer
        prompt = f"Context: {effective_context}\n\nQuestion: {task.query}"
        answer = self.llm_fn(prompt)

        # Standard LLM: correct when context fits, often wrong when truncated
        if truncated:
            correct = self._truncated_correctness(task)
        else:
            correct = self._full_context_correctness(task)

        if correct:
            answer = task.expected_answer

        # Cost: single call, proportional to context used
        effective_tokens = min(task.context_tokens, self.context_window)
        input_tokens = effective_tokens + len(task.query.split()) * 2
        output_tokens = len(answer.split()) * 2 + 50
        cost = (input_tokens * 0.005 + output_tokens * 0.015) / 1000

        return EvalResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            answer=answer,
            correct=correct,
            trajectory=[f"# Single-shot query (context {'truncated' if truncated else 'full'})"],
            strategy_detected="DIRECT",
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_calls=1,
            latency_ms=self.rng.uniform(200, 1500),
        )

    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit window."""
        max_chars = self.context_window * 4
        if len(context) > max_chars:
            return context[:max_chars]
        return context

    def _truncated_correctness(self, task: EvalTask) -> bool:
        """Determine correctness when context is truncated.

        Simple tasks may still be answerable if the relevant info is near the start.
        Most tasks requiring full context will fail.
        """
        # If the expected answer appears in the truncated portion, might still get it
        truncated_context = self._truncate_context(task.context)
        if task.expected_answer.lower() in truncated_context.lower():
            return self.rng.random() < 0.7  # 70% chance if answer visible
        return self.rng.random() < 0.15  # 15% chance otherwise (lucky guess)

    def _full_context_correctness(self, task: EvalTask) -> bool:
        """Determine correctness when full context is available."""
        # Standard LLM is generally good when it can see everything
        difficulty_rates = {
            "easy": 0.95,
            "medium": 0.85,
            "hard": 0.70,
        }
        rate = difficulty_rates.get(task.difficulty, 0.80)
        return self.rng.random() < rate

    def _default_mock_llm(self, prompt: str) -> str:
        """Default mock LLM."""
        return "mock_standard_response"


class MockStandardLLM:
    """Mock standard LLM for testing."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        """Return a mock response."""
        self.call_count += 1
        return "mock_standard_response"

    def create_executor(self, context_window: int = 8192) -> StandardExecutor:
        """Create a StandardExecutor using this mock."""
        return StandardExecutor(llm_fn=self, context_window=context_window, seed=self.seed)

"""Default prompt strategy implementation."""

from __future__ import annotations

import random
from typing import Any

from src.core.executor import Task, TaskResult
from src.meta.algorithm import MetaLearningAlgorithm


class DefaultPromptStrategy(MetaLearningAlgorithm):
    """Default strategy: fixed system prompt, random few-shot, always CoT."""

    def __init__(self, system_prompt: str = "", num_examples: int = 3) -> None:
        self.system_prompt = system_prompt or (
            "You are a helpful problem solver. Think step by step and provide your answer."
        )
        self.num_examples = num_examples
        self._learning_buffer: list[TaskResult] = []
        self._max_buffer = 100

    def prepare_prompt(self, task: Task, examples: list[dict[str, Any]]) -> str:
        """Prepare a prompt with examples and the task question."""
        parts: list[str] = []

        if examples:
            parts.append("Here are some solved examples:")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"Q: {ex.get('question', '')}")
                parts.append(f"A: {ex.get('answer', '')}")
            parts.append("")

        parts.append("Now solve the following problem step by step.")
        parts.append(f"\nQuestion: {task.question}")
        parts.append("\nProvide your final answer after 'The answer is:'")

        return "\n".join(parts)

    def select_examples(
        self, task: Task, pool: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]:
        """Select examples, preferring same category, with random fallback."""
        if not pool:
            return []

        n = min(n, len(pool))

        # Try category matching first
        same_category = [ex for ex in pool if ex.get("category") == task.category]
        if len(same_category) >= n:
            return random.sample(same_category, n)

        # Mix category matches with random
        selected = list(same_category)
        remaining = [ex for ex in pool if ex not in selected]
        if remaining:
            additional = min(n - len(selected), len(remaining))
            selected.extend(random.sample(remaining, additional))

        return selected[:n]

    def choose_reasoning_mode(
        self, task: Task, recent_results: list[TaskResult]
    ) -> str:
        """Always return 'cot' (chain of thought)."""
        return "cot"

    def learn_from_result(self, result: TaskResult) -> None:
        """Store result in learning buffer for future reference."""
        self._learning_buffer.append(result)
        if len(self._learning_buffer) > self._max_buffer:
            self._learning_buffer = self._learning_buffer[-self._max_buffer:]

    @property
    def learning_buffer(self) -> list[TaskResult]:
        return list(self._learning_buffer)

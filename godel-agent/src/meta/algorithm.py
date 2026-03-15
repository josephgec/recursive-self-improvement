"""Abstract base class for meta-learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.executor import Task, TaskResult


class MetaLearningAlgorithm(ABC):
    """Base class for modifiable meta-learning algorithms."""

    @abstractmethod
    def prepare_prompt(self, task: Task, examples: list[dict[str, Any]]) -> str:
        """Prepare the prompt for a task with few-shot examples."""
        ...

    @abstractmethod
    def select_examples(
        self, task: Task, pool: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]:
        """Select few-shot examples for a task."""
        ...

    @abstractmethod
    def choose_reasoning_mode(
        self, task: Task, recent_results: list[TaskResult]
    ) -> str:
        """Choose reasoning mode for a task."""
        ...

    @abstractmethod
    def learn_from_result(self, result: TaskResult) -> None:
        """Update internal state based on a task result."""
        ...

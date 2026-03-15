"""Reasoning strategy selection."""

from __future__ import annotations

from src.core.executor import Task, TaskResult


class ReasoningStrategy:
    """Chooses a reasoning mode based on task and recent results.

    Available modes: "cot", "code", "direct", "decompose".
    Initially always returns "cot".
    """

    def choose(self, task: Task, recent_results: list[TaskResult] | None = None) -> str:
        """Choose a reasoning mode for the given task.

        Default implementation always returns "cot".
        This is intended to be modified by the self-improvement loop.
        """
        return "cot"

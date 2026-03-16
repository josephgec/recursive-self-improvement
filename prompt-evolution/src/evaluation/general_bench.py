"""Placeholder for general domain benchmarks.

This module provides a stub interface for future domain-agnostic benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GeneralTask:
    """A general benchmark task."""

    task_id: str
    domain: str
    question: str
    expected_answer: str
    difficulty: str = "medium"


class GeneralBenchmark:
    """Placeholder benchmark for non-financial domains.

    To be extended for general-purpose prompt evaluation.
    """

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self._tasks: List[GeneralTask] = []

    def generate_tasks(self, n: int = 10) -> List[GeneralTask]:
        """Generate placeholder tasks."""
        self._tasks = [
            GeneralTask(
                task_id=f"gen_{i}",
                domain=self.domain,
                question=f"Sample question {i} for domain '{self.domain}'",
                expected_answer=f"answer_{i}",
            )
            for i in range(n)
        ]
        return list(self._tasks)

    def to_eval_tasks(
        self, tasks: Optional[List[GeneralTask]] = None
    ) -> List[Dict]:
        """Convert to eval task dicts."""
        if tasks is None:
            tasks = self._tasks
        return [
            {
                "task_id": t.task_id,
                "question": t.question,
                "expected_answer": t.expected_answer,
                "domain": t.domain,
            }
            for t in tasks
        ]

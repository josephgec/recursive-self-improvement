"""Validation runner for testing modifications."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.core.executor import Task, TaskResult, TaskExecutor
from src.validation.suite import ValidationSuite


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool = False
    pass_rate: float = 0.0
    baseline_rate: float = 0.0
    delta: float = 0.0
    total: int = 0
    correct: int = 0
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "baseline_rate": self.baseline_rate,
            "delta": self.delta,
            "total": self.total,
            "correct": self.correct,
        }


class ValidationRunner:
    """Runs validation tasks and checks pass rate."""

    def __init__(
        self,
        executor: TaskExecutor,
        suite: ValidationSuite,
        min_pass_rate: float = 0.90,
        performance_threshold: float = -0.05,
    ) -> None:
        self.executor = executor
        self.suite = suite
        self.min_pass_rate = min_pass_rate
        self.performance_threshold = performance_threshold
        self._baseline_rate: float | None = None

    def set_baseline(self, system_prompt: str = "") -> float:
        """Establish baseline performance on the validation suite."""
        tasks = self.suite.get_tasks()
        if not tasks:
            self._baseline_rate = 1.0
            return 1.0

        results = self.executor.execute_validation(tasks, system_prompt=system_prompt)
        correct = sum(1 for r in results if r.correct)
        self._baseline_rate = correct / len(results) if results else 0.0
        return self._baseline_rate

    def run(self, system_prompt: str = "") -> ValidationResult:
        """Run full validation suite."""
        tasks = self.suite.get_tasks()
        if not tasks:
            return ValidationResult(passed=True, pass_rate=1.0, baseline_rate=self._baseline_rate or 1.0)

        results = self.executor.execute_validation(tasks, system_prompt=system_prompt)
        return self._evaluate(results)

    def run_quick(self, system_prompt: str = "", sample_fraction: float = 0.2) -> ValidationResult:
        """Run a quick validation on a 20% sample."""
        tasks = self.suite.get_tasks()
        if not tasks:
            return ValidationResult(passed=True, pass_rate=1.0, baseline_rate=self._baseline_rate or 1.0)

        sample_size = max(1, int(len(tasks) * sample_fraction))
        sample = random.sample(tasks, min(sample_size, len(tasks)))
        results = self.executor.execute_validation(sample, system_prompt=system_prompt)
        return self._evaluate(results)

    def _evaluate(self, results: list[TaskResult]) -> ValidationResult:
        """Evaluate validation results."""
        if not results:
            return ValidationResult(passed=True, pass_rate=1.0, baseline_rate=self._baseline_rate or 1.0)

        correct = sum(1 for r in results if r.correct)
        pass_rate = correct / len(results)
        baseline = self._baseline_rate or pass_rate

        delta = pass_rate - baseline
        passed = pass_rate >= self.min_pass_rate and delta >= self.performance_threshold

        return ValidationResult(
            passed=passed,
            pass_rate=pass_rate,
            baseline_rate=baseline,
            delta=delta,
            total=len(results),
            correct=correct,
            results=results,
        )

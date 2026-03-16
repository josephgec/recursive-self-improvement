"""Analyze strategy failure modes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.benchmarks.task import EvalResult


@dataclass
class FailureCase:
    """A single failure case."""
    task_id: str
    strategy: str
    category: str
    error_type: str
    description: str


class StrategyFailureModeAnalyzer:
    """Analyze how and why strategies fail."""

    def categorize_failures(
        self,
        results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[FailureCase]]:
        """Categorize failures by error type.

        Error types:
        - truncation: context was too large
        - wrong_strategy: strategy mismatch for category
        - incomplete_search: didn't find all info
        - aggregation_error: wrong aggregation
        - reasoning_error: logical error
        - unknown: other
        """
        failures: Dict[str, List[FailureCase]] = defaultdict(list)

        for r in results:
            if r.correct:
                continue

            category = "unknown"
            if task_categories and r.task_id in task_categories:
                category = task_categories[r.task_id]
            else:
                for cat in ["retrieval", "aggregation", "counting", "reasoning", "needle", "refactoring", "bug_fix"]:
                    if cat in r.task_id:
                        category = cat
                        break

            error_type = self._classify_error(r, category)
            case = FailureCase(
                task_id=r.task_id,
                strategy=r.strategy_detected or "unknown",
                category=category,
                error_type=error_type,
                description=self._describe_failure(r, error_type),
            )
            failures[error_type].append(case)

        return dict(failures)

    def strategy_misapplication(
        self,
        results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
    ) -> List[FailureCase]:
        """Find cases where the wrong strategy was used for a category."""
        # Expected strategies per category
        expected: Dict[str, List[str]] = {
            "retrieval": ["PEEK_THEN_GREP", "DIRECT"],
            "needle": ["PEEK_THEN_GREP", "ITERATIVE_SEARCH"],
            "aggregation": ["MAP_REDUCE", "ITERATIVE_SEARCH"],
            "counting": ["ITERATIVE_SEARCH", "MAP_REDUCE"],
            "reasoning": ["HIERARCHICAL", "HYBRID"],
            "refactoring": ["PEEK_THEN_GREP", "DIRECT"],
            "bug_fix": ["DIRECT", "PEEK_THEN_GREP"],
        }

        misapplied: List[FailureCase] = []
        for r in results:
            if r.correct:
                continue

            category = "unknown"
            if task_categories and r.task_id in task_categories:
                category = task_categories[r.task_id]
            else:
                for cat in expected:
                    if cat in r.task_id:
                        category = cat
                        break

            if category in expected:
                strategy = r.strategy_detected or "unknown"
                if strategy not in expected[category]:
                    misapplied.append(FailureCase(
                        task_id=r.task_id,
                        strategy=strategy,
                        category=category,
                        error_type="wrong_strategy",
                        description=f"Used {strategy} for {category}; expected one of {expected[category]}",
                    ))

        return misapplied

    def failure_rate_by_strategy(
        self,
        results: List[EvalResult],
    ) -> Dict[str, float]:
        """Compute failure rate per strategy."""
        total: Dict[str, int] = defaultdict(int)
        failures: Dict[str, int] = defaultdict(int)

        for r in results:
            strategy = r.strategy_detected or "unknown"
            total[strategy] += 1
            if not r.correct:
                failures[strategy] += 1

        return {
            s: failures[s] / total[s] if total[s] > 0 else 0.0
            for s in total
        }

    def _classify_error(self, result: EvalResult, category: str) -> str:
        """Classify the type of error."""
        if result.error:
            return "runtime_error"

        trajectory_str = " ".join(result.trajectory).lower()

        if "truncat" in trajectory_str:
            return "truncation"

        if category in ("aggregation", "counting") and result.answer:
            return "aggregation_error"

        if category in ("reasoning",) and result.answer:
            return "reasoning_error"

        if "grep" in trajectory_str or "search" in trajectory_str:
            return "incomplete_search"

        return "unknown"

    def _describe_failure(self, result: EvalResult, error_type: str) -> str:
        """Generate a human-readable failure description."""
        descriptions = {
            "truncation": f"Context truncated, likely lost critical info. Task: {result.task_id}",
            "wrong_strategy": f"Strategy {result.strategy_detected} was suboptimal for this task type.",
            "incomplete_search": f"Search did not find all relevant information.",
            "aggregation_error": f"Aggregation produced wrong result: got '{result.answer}'.",
            "reasoning_error": f"Reasoning chain produced wrong conclusion: got '{result.answer}'.",
            "runtime_error": f"Runtime error: {result.error}",
            "unknown": f"Failed with answer '{result.answer}' (strategy: {result.strategy_detected}).",
        }
        return descriptions.get(error_type, f"Unknown failure for task {result.task_id}")

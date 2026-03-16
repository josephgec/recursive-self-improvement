"""RLMMetrics: compute evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.session import SessionResult
from src.strategies.detector import StrategyDetector, Strategy


@dataclass
class MetricResult:
    """A single metric computation result."""
    name: str
    value: float
    details: Dict[str, Any]


class RLMMetrics:
    """Compute various metrics over session results."""

    def __init__(self) -> None:
        self.detector = StrategyDetector()

    def accuracy(
        self,
        results: List[SessionResult],
        expected: List[str],
        exact: bool = True,
    ) -> MetricResult:
        """Compute accuracy: fraction of results matching expected answers."""
        if not results:
            return MetricResult(name="accuracy", value=0.0, details={"n": 0})

        correct = 0
        details_list: List[Dict[str, Any]] = []
        for sr, exp in zip(results, expected):
            actual = str(sr.result) if sr.result else ""
            if exact:
                match = actual.strip() == exp.strip()
            else:
                match = exp.strip().lower() in actual.strip().lower()
            if match:
                correct += 1
            details_list.append({"expected": exp, "actual": actual, "match": match})

        return MetricResult(
            name="accuracy",
            value=correct / len(results),
            details={"n": len(results), "correct": correct, "per_task": details_list},
        )

    def cost_per_query(
        self,
        results: List[SessionResult],
        cost_per_iteration: float = 0.01,
    ) -> MetricResult:
        """Estimate cost per query based on iteration count."""
        if not results:
            return MetricResult(name="cost_per_query", value=0.0, details={"n": 0})
        total_iters = sum(r.total_iterations for r in results)
        total_cost = total_iters * cost_per_iteration
        avg_cost = total_cost / len(results)
        return MetricResult(
            name="cost_per_query",
            value=avg_cost,
            details={
                "total_iterations": total_iters,
                "total_cost": total_cost,
                "avg_iterations": total_iters / len(results),
            },
        )

    def context_utilization(
        self,
        results: List[SessionResult],
    ) -> MetricResult:
        """Estimate how much of the context was explored (via code patterns)."""
        if not results:
            return MetricResult(name="context_utilization", value=0.0, details={"n": 0})

        scores: List[float] = []
        for sr in results:
            all_code = " ".join(c for step in sr.trajectory for c in step.code_blocks)
            score = 0.0
            if "peek(" in all_code:
                score += 0.2
            if "grep(" in all_code or "search(" in all_code:
                score += 0.3
            if "chunk(" in all_code:
                score += 0.3
            if "CONTEXT" in all_code:
                score += 0.2
            scores.append(min(1.0, score))

        avg = sum(scores) / len(scores)
        return MetricResult(
            name="context_utilization",
            value=avg,
            details={"per_session": scores},
        )

    def recursion_depth_distribution(
        self,
        results: List[SessionResult],
    ) -> MetricResult:
        """Distribution of recursion depths."""
        if not results:
            return MetricResult(name="recursion_depth_distribution", value=0.0, details={})
        depths = [r.depth for r in results]
        counter = Counter(depths)
        avg_depth = sum(depths) / len(depths)
        return MetricResult(
            name="recursion_depth_distribution",
            value=avg_depth,
            details={"distribution": dict(counter), "max_depth": max(depths)},
        )

    def strategy_distribution(
        self,
        results: List[SessionResult],
    ) -> MetricResult:
        """Distribution of detected strategies."""
        if not results:
            return MetricResult(name="strategy_distribution", value=0.0, details={})
        strategies: List[str] = []
        for sr in results:
            cls = self.detector.classify(sr.trajectory)
            strategies.append(cls.strategy.value)
        counter = Counter(strategies)
        return MetricResult(
            name="strategy_distribution",
            value=len(counter),
            details={"distribution": dict(counter)},
        )

    def accuracy_per_dollar(
        self,
        results: List[SessionResult],
        expected: List[str],
        cost_per_iteration: float = 0.01,
    ) -> MetricResult:
        """Accuracy divided by average cost per query."""
        acc = self.accuracy(results, expected)
        cost = self.cost_per_query(results, cost_per_iteration)
        if cost.value == 0:
            ratio = 0.0
        else:
            ratio = acc.value / cost.value
        return MetricResult(
            name="accuracy_per_dollar",
            value=ratio,
            details={"accuracy": acc.value, "cost": cost.value},
        )

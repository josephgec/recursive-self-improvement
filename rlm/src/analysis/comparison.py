"""RLMComparisonAnalyzer: compare accuracy, cost, and scaling between approaches."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.session import SessionResult
from src.evaluation.metrics import RLMMetrics, MetricResult


class RLMComparisonAnalyzer:
    """Compare two sets of session results across multiple dimensions."""

    def __init__(self, metrics: Optional[RLMMetrics] = None) -> None:
        self.metrics = metrics or RLMMetrics()

    def accuracy_comparison(
        self,
        results_a: List[SessionResult],
        results_b: List[SessionResult],
        expected: List[str],
        labels: tuple[str, str] = ("A", "B"),
    ) -> Dict[str, Any]:
        """Compare accuracy between two approaches."""
        acc_a = self.metrics.accuracy(results_a, expected, exact=False)
        acc_b = self.metrics.accuracy(results_b, expected, exact=False)
        return {
            labels[0]: {"accuracy": acc_a.value, "details": acc_a.details},
            labels[1]: {"accuracy": acc_b.value, "details": acc_b.details},
            "difference": acc_a.value - acc_b.value,
        }

    def cost_comparison(
        self,
        results_a: List[SessionResult],
        results_b: List[SessionResult],
        labels: tuple[str, str] = ("A", "B"),
    ) -> Dict[str, Any]:
        """Compare cost between two approaches."""
        cost_a = self.metrics.cost_per_query(results_a)
        cost_b = self.metrics.cost_per_query(results_b)
        return {
            labels[0]: {"cost_per_query": cost_a.value, "details": cost_a.details},
            labels[1]: {"cost_per_query": cost_b.value, "details": cost_b.details},
            "difference": cost_a.value - cost_b.value,
        }

    def context_scaling_comparison(
        self,
        results_by_size_a: Dict[int, List[SessionResult]],
        results_by_size_b: Dict[int, List[SessionResult]],
        expected_by_size: Dict[int, List[str]],
        labels: tuple[str, str] = ("A", "B"),
    ) -> Dict[str, Any]:
        """Compare how two approaches scale with context size."""
        sizes = sorted(set(results_by_size_a.keys()) | set(results_by_size_b.keys()))
        comparison: Dict[str, Any] = {"sizes": sizes, "per_size": {}}
        for size in sizes:
            ra = results_by_size_a.get(size, [])
            rb = results_by_size_b.get(size, [])
            exp = expected_by_size.get(size, [])
            entry: Dict[str, Any] = {}
            if ra and exp:
                entry[labels[0]] = {
                    "accuracy": self.metrics.accuracy(ra, exp, exact=False).value,
                    "cost": self.metrics.cost_per_query(ra).value,
                }
            if rb and exp:
                entry[labels[1]] = {
                    "accuracy": self.metrics.accuracy(rb, exp, exact=False).value,
                    "cost": self.metrics.cost_per_query(rb).value,
                }
            comparison["per_size"][size] = entry
        return comparison

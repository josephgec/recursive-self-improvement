"""BenchmarkRunner: run benchmarks and comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.core.session import RLMSession, SessionResult
from src.core.config import DEFAULT_CONFIG
from src.evaluation.metrics import RLMMetrics, MetricResult
from src.recursion.depth_controller import DepthController


@dataclass
class ComparisonResult:
    """Result of comparing two approaches on a benchmark."""
    benchmark_name: str
    approach_a: str
    approach_b: str
    results_a: List[SessionResult]
    results_b: List[SessionResult]
    metrics_a: Dict[str, MetricResult]
    metrics_b: Dict[str, MetricResult]

    def summary(self) -> str:
        lines = [f"Comparison: {self.approach_a} vs {self.approach_b} on {self.benchmark_name}"]
        for name in self.metrics_a:
            va = self.metrics_a[name].value
            vb = self.metrics_b.get(name, MetricResult(name, 0.0, {})).value
            lines.append(f"  {name}: {va:.4f} vs {vb:.4f}")
        return "\n".join(lines)


@dataclass
class ScalingResult:
    """Result of context-scaling experiments."""
    context_sizes: List[int]
    results_per_size: Dict[int, List[SessionResult]]
    metrics_per_size: Dict[int, Dict[str, MetricResult]]

    def summary(self) -> str:
        lines = ["Context Scaling Results"]
        for size in self.context_sizes:
            metrics = self.metrics_per_size.get(size, {})
            acc = metrics.get("accuracy", MetricResult("accuracy", 0.0, {})).value
            cost = metrics.get("cost_per_query", MetricResult("cost", 0.0, {})).value
            lines.append(f"  Size {size}: accuracy={acc:.4f}, cost={cost:.4f}")
        return "\n".join(lines)


class BenchmarkRunner:
    """Run benchmarks, comparisons, and scaling experiments."""

    def __init__(self, metrics: Optional[RLMMetrics] = None) -> None:
        self.metrics = metrics or RLMMetrics()

    def run_tasks(
        self,
        tasks: List[Any],
        llm: Any,
        max_iterations: int = 10,
    ) -> List[SessionResult]:
        """Run a list of tasks through RLM sessions."""
        results: List[SessionResult] = []
        for task in tasks:
            dc = DepthController(max_iterations=max_iterations)
            session = RLMSession(
                llm=llm,
                max_iterations=max_iterations,
                depth_controller=dc,
            )
            sr = session.run(query=task.query, context=task.context)
            results.append(sr)
        return results

    def run_comparison(
        self,
        tasks: List[Any],
        llm_a: Any,
        llm_b: Any,
        benchmark_name: str = "benchmark",
        max_iterations: int = 10,
    ) -> ComparisonResult:
        """Compare two LLMs on the same set of tasks."""
        expected = [t.expected_answer for t in tasks]

        results_a = self.run_tasks(tasks, llm_a, max_iterations)
        results_b = self.run_tasks(tasks, llm_b, max_iterations)

        metrics_a = self._compute_metrics(results_a, expected)
        metrics_b = self._compute_metrics(results_b, expected)

        return ComparisonResult(
            benchmark_name=benchmark_name,
            approach_a="llm_a",
            approach_b="llm_b",
            results_a=results_a,
            results_b=results_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
        )

    def run_context_scaling(
        self,
        task_factory: Callable[[int], Any],
        context_sizes: List[int],
        llm: Any,
        max_iterations: int = 10,
    ) -> ScalingResult:
        """Run a task at different context sizes."""
        results_per_size: Dict[int, List[SessionResult]] = {}
        metrics_per_size: Dict[int, Dict[str, MetricResult]] = {}

        for size in context_sizes:
            task = task_factory(size)
            session_results = self.run_tasks([task], llm, max_iterations)
            results_per_size[size] = session_results
            expected = [task.expected_answer]
            metrics_per_size[size] = self._compute_metrics(session_results, expected)

        return ScalingResult(
            context_sizes=context_sizes,
            results_per_size=results_per_size,
            metrics_per_size=metrics_per_size,
        )

    def _compute_metrics(
        self,
        results: List[SessionResult],
        expected: List[str],
    ) -> Dict[str, MetricResult]:
        return {
            "accuracy": self.metrics.accuracy(results, expected, exact=False),
            "cost_per_query": self.metrics.cost_per_query(results),
            "context_utilization": self.metrics.context_utilization(results),
            "strategy_distribution": self.metrics.strategy_distribution(results),
        }

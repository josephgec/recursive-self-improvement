"""Cost analysis: inference costs, latency, accuracy-cost tradeoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation.benchmark_suite import BenchmarkResults


@dataclass
class CostReport:
    """Report on inference costs."""
    system_costs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    most_efficient: str = ""
    cost_accuracy_tradeoff: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class CostAnalyzer:
    """Analyzes inference costs and accuracy-cost tradeoffs."""

    # Mock cost parameters
    COST_PER_LLM_CALL = 0.01  # dollars
    COST_PER_TOOL_CALL = 0.001  # dollars
    LATENCY_PER_LLM_CALL = 0.5  # seconds
    LATENCY_PER_TOOL_CALL = 0.1  # seconds

    def analyze(self, results: BenchmarkResults) -> CostReport:
        """Analyze costs for all systems.

        Args:
            results: BenchmarkResults from benchmark suite.

        Returns:
            CostReport with cost and efficiency metrics.
        """
        system_costs: Dict[str, Dict[str, float]] = {}
        cost_accuracy: Dict[str, Dict[str, float]] = {}

        for system in set(
            list(results.generalization.keys())
            + list(results.interpretability.keys())
            + list(results.robustness.keys())
        ):
            costs = self._compute_inference_costs(system, results)
            system_costs[system] = costs

            # Compute accuracy
            gen = results.generalization.get(system)
            accuracy = 0.0
            if gen:
                accuracy = (gen.in_domain_accuracy + gen.out_of_domain_accuracy) / 2

            cost_accuracy[system] = {
                "accuracy": accuracy,
                "cost_per_task": costs.get("cost_per_task", 0.0),
                "latency_per_task": costs.get("latency_per_task", 0.0),
                "efficiency": accuracy / max(costs.get("cost_per_task", 0.01), 0.001),
            }

        # Find most efficient
        most_efficient = ""
        best_efficiency = -1.0
        for system, ca in cost_accuracy.items():
            if ca["efficiency"] > best_efficiency:
                best_efficiency = ca["efficiency"]
                most_efficient = system

        return CostReport(
            system_costs=system_costs,
            most_efficient=most_efficient,
            cost_accuracy_tradeoff=cost_accuracy,
        )

    def _compute_inference_costs(
        self, system: str, results: BenchmarkResults
    ) -> Dict[str, float]:
        """Compute inference costs for a system.

        Args:
            system: System name.
            results: BenchmarkResults.

        Returns:
            Dict with cost metrics.
        """
        if system == "hybrid":
            # Hybrid: 1 LLM call + N tool calls per step
            avg_tool_calls = 2.0  # average from benchmark
            cost_per_task = (
                self.COST_PER_LLM_CALL * 2  # initial + follow-up
                + self.COST_PER_TOOL_CALL * avg_tool_calls
            )
            latency_per_task = (
                self.LATENCY_PER_LLM_CALL * 2
                + self.LATENCY_PER_TOOL_CALL * avg_tool_calls
            )
        elif system == "integrative":
            # Integrative: 1 LLM call with constraint overhead
            cost_per_task = self.COST_PER_LLM_CALL * 1.3  # 30% overhead for constraints
            latency_per_task = self.LATENCY_PER_LLM_CALL * 1.5  # constraint checking adds latency
        else:
            # Prose: 1 LLM call
            cost_per_task = self.COST_PER_LLM_CALL
            latency_per_task = self.LATENCY_PER_LLM_CALL

        return {
            "cost_per_task": cost_per_task,
            "latency_per_task": latency_per_task,
            "total_cost_100_tasks": cost_per_task * 100,
        }

    def plot_accuracy_latency_tradeoff(self, report: CostReport) -> Optional[Any]:
        """Plot accuracy vs latency tradeoff."""
        try:
            import matplotlib.pyplot as plt

            systems = list(report.cost_accuracy_tradeoff.keys())
            accuracies = [report.cost_accuracy_tradeoff[s]["accuracy"] for s in systems]
            latencies = [report.cost_accuracy_tradeoff[s]["latency_per_task"] for s in systems]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(latencies, accuracies, s=100)
            for i, system in enumerate(systems):
                ax.annotate(system, (latencies[i], accuracies[i]),
                           textcoords="offset points", xytext=(5, 5))
            ax.set_xlabel("Latency per task (s)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs Latency Tradeoff")
            plt.close(fig)
            return fig
        except ImportError:
            return None

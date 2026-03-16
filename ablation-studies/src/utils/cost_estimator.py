"""Cost estimation for ablation study runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.suites.base import AblationSuite


class CostEstimator:
    """Estimate computational cost of ablation runs."""

    def __init__(self, cost_per_run: float = 0.10,
                 time_per_run_seconds: float = 60.0):
        """Initialize the cost estimator.

        Args:
            cost_per_run: Estimated cost in USD per single run.
            time_per_run_seconds: Estimated wall-clock time per run.
        """
        self.cost_per_run = cost_per_run
        self.time_per_run_seconds = time_per_run_seconds

    def estimate_suite(
        self,
        suite: AblationSuite,
        repetitions: int = 5,
    ) -> Dict[str, Any]:
        """Estimate cost and time for a single suite."""
        n_conditions = len(suite.get_conditions())
        total_runs = n_conditions * repetitions
        total_cost = total_runs * self.cost_per_run
        total_time = total_runs * self.time_per_run_seconds

        return {
            "suite_name": suite.get_paper_name(),
            "n_conditions": n_conditions,
            "repetitions": repetitions,
            "total_runs": total_runs,
            "cost_per_run_usd": self.cost_per_run,
            "total_cost_usd": total_cost,
            "time_per_run_sec": self.time_per_run_seconds,
            "total_time_sec": total_time,
            "total_time_hours": total_time / 3600,
        }

    def estimate_all_suites(
        self,
        suites: List[AblationSuite],
        repetitions: int = 5,
    ) -> Dict[str, Any]:
        """Estimate cost and time for all suites combined."""
        suite_estimates = []
        total_cost = 0.0
        total_time = 0.0
        total_runs = 0

        for suite in suites:
            est = self.estimate_suite(suite, repetitions)
            suite_estimates.append(est)
            total_cost += est["total_cost_usd"]
            total_time += est["total_time_sec"]
            total_runs += est["total_runs"]

        return {
            "suites": suite_estimates,
            "grand_total_runs": total_runs,
            "grand_total_cost_usd": total_cost,
            "grand_total_time_sec": total_time,
            "grand_total_time_hours": total_time / 3600,
        }

    def format_estimate(self, estimate: Dict[str, Any]) -> str:
        """Format an estimate as a human-readable string."""
        lines = []

        if "suites" in estimate:
            lines.append("=== Cost Estimate: All Suites ===")
            for s in estimate["suites"]:
                lines.append(
                    f"  {s['suite_name']}: {s['total_runs']} runs, "
                    f"${s['total_cost_usd']:.2f}, "
                    f"{s['total_time_hours']:.1f}h"
                )
            lines.append(f"  TOTAL: {estimate['grand_total_runs']} runs, "
                        f"${estimate['grand_total_cost_usd']:.2f}, "
                        f"{estimate['grand_total_time_hours']:.1f}h")
        else:
            lines.append(f"=== Cost Estimate: {estimate['suite_name']} ===")
            lines.append(f"  Conditions: {estimate['n_conditions']}")
            lines.append(f"  Repetitions: {estimate['repetitions']}")
            lines.append(f"  Total runs: {estimate['total_runs']}")
            lines.append(f"  Cost: ${estimate['total_cost_usd']:.2f}")
            lines.append(f"  Time: {estimate['total_time_hours']:.1f}h")

        return "\n".join(lines)

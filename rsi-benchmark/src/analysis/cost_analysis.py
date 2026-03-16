"""Cost analysis: compute cost breakdowns and efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CostBreakdown:
    """Cost breakdown for an evaluation run."""
    total_cost: float
    cost_per_iteration: float
    cost_per_benchmark: Dict[str, float]
    cost_per_improvement_point: float
    iterations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def cost_breakdown(
    iterations: int,
    benchmarks: List[str],
    cost_per_task: float = 0.01,
    tasks_per_benchmark: int = 30,
    improvement: float = 0.1,
) -> CostBreakdown:
    """Compute cost breakdown for an evaluation run."""
    total_tasks = iterations * len(benchmarks) * tasks_per_benchmark
    total_cost = total_tasks * cost_per_task
    cost_per_iter = total_cost / max(iterations, 1)

    cost_per_bm: Dict[str, float] = {}
    bm_cost = (iterations * tasks_per_benchmark * cost_per_task)
    for bm in benchmarks:
        cost_per_bm[bm] = bm_cost

    cost_per_imp = total_cost / max(improvement, 1e-10)

    return CostBreakdown(
        total_cost=round(total_cost, 2),
        cost_per_iteration=round(cost_per_iter, 2),
        cost_per_benchmark=cost_per_bm,
        cost_per_improvement_point=round(cost_per_imp, 2),
        iterations=iterations,
    )


def cost_per_improvement_point(
    total_cost: float,
    improvement: float,
) -> float:
    """Compute cost per improvement point."""
    if improvement <= 0:
        return float("inf")
    return total_cost / improvement


def format_cost_table(breakdown: CostBreakdown) -> str:
    """Format cost breakdown as a table string."""
    lines = [
        "Cost Breakdown:",
        f"  Total Cost: ${breakdown.total_cost:.2f}",
        f"  Iterations: {breakdown.iterations}",
        f"  Cost/Iteration: ${breakdown.cost_per_iteration:.2f}",
        f"  Cost/Improvement Point: ${breakdown.cost_per_improvement_point:.2f}",
        "",
        "  Per Benchmark:",
    ]
    for bm, cost in breakdown.cost_per_benchmark.items():
        lines.append(f"    {bm}: ${cost:.2f}")
    return "\n".join(lines)

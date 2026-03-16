"""Cost breakdown analysis: where the money goes."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.benchmarks.task import EvalResult


class CostBreakdownAnalysis:
    """Analyze cost distribution across systems, strategies, and categories."""

    def by_strategy(self, results: List[EvalResult]) -> Dict[str, float]:
        """Cost breakdown by strategy."""
        costs: Dict[str, float] = defaultdict(float)
        for r in results:
            strat = r.strategy_detected or "unknown"
            costs[strat] += r.cost
        return dict(costs)

    def by_category(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> Dict[str, float]:
        """Cost breakdown by task category."""
        costs: Dict[str, float] = defaultdict(float)
        for r in results:
            cat = task_categories.get(r.task_id, "unknown")
            costs[cat] += r.cost
        return dict(costs)

    def by_benchmark(self, results: List[EvalResult]) -> Dict[str, float]:
        """Cost breakdown by benchmark."""
        costs: Dict[str, float] = defaultdict(float)
        for r in results:
            costs[r.benchmark] += r.cost
        return dict(costs)

    def io_ratio(self, results: List[EvalResult]) -> Dict[str, float]:
        """Compute input vs output token ratio."""
        total_input = sum(r.input_tokens for r in results)
        total_output = sum(r.output_tokens for r in results)
        total = total_input + total_output

        return {
            "input_ratio": total_input / total if total > 0 else 0,
            "output_ratio": total_output / total if total > 0 else 0,
            "total_input_tokens": float(total_input),
            "total_output_tokens": float(total_output),
        }

    def cost_summary_table(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> str:
        """Generate a formatted cost summary table."""
        by_strat = self.by_strategy(results)
        by_cat = self.by_category(results, task_categories)
        io = self.io_ratio(results)

        total_cost = sum(r.cost for r in results)

        lines = [
            "Cost Breakdown Summary",
            "=" * 50,
            "",
            "By Strategy:",
        ]
        for strat, cost in sorted(by_strat.items(), key=lambda x: -x[1]):
            pct = cost / total_cost * 100 if total_cost > 0 else 0
            lines.append(f"  {strat:<25} ${cost:.4f} ({pct:.1f}%)")

        lines.extend(["", "By Category:"])
        for cat, cost in sorted(by_cat.items(), key=lambda x: -x[1]):
            pct = cost / total_cost * 100 if total_cost > 0 else 0
            lines.append(f"  {cat:<25} ${cost:.4f} ({pct:.1f}%)")

        lines.extend([
            "",
            f"Input/Output ratio: {io['input_ratio']:.0%} / {io['output_ratio']:.0%}",
            f"Total cost: ${total_cost:.4f}",
        ])

        return "\n".join(lines)

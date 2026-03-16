"""Strategy distribution landscape analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.benchmarks.task import EvalResult


class StrategyLandscape:
    """Visualize strategy distribution as heatmaps and tables."""

    def distribution_table(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> str:
        """Generate a table of strategy distribution by category.

        Returns:
            Formatted ASCII table.
        """
        # Count strategies per category
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        all_strategies: set = set()

        for r in results:
            cat = task_categories.get(r.task_id, "unknown")
            strat = r.strategy_detected or "unknown"
            counts[cat][strat] += 1
            all_strategies.add(strat)

        strategies = sorted(all_strategies)
        categories = sorted(counts.keys())

        # Build table
        header = f"{'Category':<20}" + "".join(f"{s:<18}" for s in strategies)
        lines = [
            "Strategy Distribution by Category",
            "=" * len(header),
            header,
            "-" * len(header),
        ]

        for cat in categories:
            row = f"{cat:<20}"
            for strat in strategies:
                count = counts[cat].get(strat, 0)
                row += f"{count:<18}"
            lines.append(row)

        lines.append("-" * len(header))
        return "\n".join(lines)

    def heatmap_ascii(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> str:
        """Generate ASCII heatmap of strategy vs category.

        Uses characters to represent density:
        ' ' = 0, '.' = 1, ':' = 2-3, '#' = 4+
        """
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        all_strategies: set = set()

        for r in results:
            cat = task_categories.get(r.task_id, "unknown")
            strat = r.strategy_detected or "unknown"
            counts[cat][strat] += 1
            all_strategies.add(strat)

        strategies = sorted(all_strategies)
        categories = sorted(counts.keys())

        lines = ["Strategy Heatmap", ""]

        # Header
        header = f"{'':>15} " + " ".join(f"{s[:8]:>8}" for s in strategies)
        lines.append(header)
        lines.append("-" * len(header))

        for cat in categories:
            row = f"{cat[:15]:>15} "
            for strat in strategies:
                count = counts[cat].get(strat, 0)
                if count == 0:
                    char = "   .   "
                elif count <= 1:
                    char = "   o   "
                elif count <= 3:
                    char = "  oo   "
                else:
                    char = "  ###  "
                row += f"{char:>8}"
            lines.append(row)

        return "\n".join(lines)

    def strategy_summary(self, results: List[EvalResult]) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics per strategy.

        Returns:
            Dict[strategy] -> {count, accuracy, avg_cost, avg_latency}
        """
        stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {
            "correct": [], "cost": [], "latency": []
        })

        for r in results:
            strat = r.strategy_detected or "unknown"
            stats[strat]["correct"].append(float(r.correct))
            stats[strat]["cost"].append(r.cost)
            stats[strat]["latency"].append(r.latency_ms)

        summary: Dict[str, Dict[str, float]] = {}
        for strat, data in stats.items():
            n = len(data["correct"])
            summary[strat] = {
                "count": float(n),
                "accuracy": sum(data["correct"]) / n if n > 0 else 0.0,
                "avg_cost": sum(data["cost"]) / n if n > 0 else 0.0,
                "avg_latency": sum(data["latency"]) / n if n > 0 else 0.0,
            }
        return summary

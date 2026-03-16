"""Efficiency frontier analysis: Pareto front of accuracy vs cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.benchmarks.task import EvalResult


@dataclass
class FrontierPoint:
    """A point on the efficiency frontier."""
    system: str
    accuracy: float
    cost: float
    is_pareto_optimal: bool = False


class EfficiencyFrontierAnalyzer:
    """Compute the Pareto frontier of accuracy vs cost."""

    def compute_frontier(
        self,
        system_results: Dict[str, List[EvalResult]],
    ) -> List[FrontierPoint]:
        """Compute the efficiency frontier across multiple systems.

        Args:
            system_results: Dict mapping system name to its results.

        Returns:
            List of FrontierPoints, with is_pareto_optimal set.
        """
        points: List[FrontierPoint] = []

        for system, results in system_results.items():
            if not results:
                continue
            accuracy = sum(1 for r in results if r.correct) / len(results)
            total_cost = sum(r.cost for r in results)
            avg_cost = total_cost / len(results)
            points.append(FrontierPoint(
                system=system,
                accuracy=accuracy,
                cost=avg_cost,
            ))

        # Compute Pareto optimality
        # A point is Pareto optimal if no other point is both
        # cheaper and more accurate
        for p in points:
            p.is_pareto_optimal = True
            for q in points:
                if q.system != p.system:
                    if q.accuracy >= p.accuracy and q.cost <= p.cost:
                        if q.accuracy > p.accuracy or q.cost < p.cost:
                            p.is_pareto_optimal = False
                            break

        return points

    def plot(
        self,
        system_results: Dict[str, List[EvalResult]],
    ) -> str:
        """Generate an ASCII scatter plot of the efficiency frontier.

        X-axis: cost, Y-axis: accuracy.
        """
        points = self.compute_frontier(system_results)
        if not points:
            return "No data to plot."

        lines: List[str] = []
        lines.append("Efficiency Frontier: Accuracy vs Cost")
        lines.append("=" * 50)

        # Sort by cost
        points.sort(key=lambda p: p.cost)

        max_cost = max(p.cost for p in points) or 1.0
        width = 40

        for p in points:
            cost_pos = int(p.cost / max_cost * width)
            marker = "*" if p.is_pareto_optimal else "o"
            optimal_label = " [PARETO]" if p.is_pareto_optimal else ""
            bar = " " * cost_pos + marker
            lines.append(
                f"  {p.system:<15} acc={p.accuracy:.1%} cost=${p.cost:.4f} "
                f"{bar}{optimal_label}"
            )

        lines.append("")
        lines.append("Legend: * = Pareto optimal, o = dominated")

        return "\n".join(lines)

    def dominated_by(
        self,
        system_results: Dict[str, List[EvalResult]],
    ) -> Dict[str, List[str]]:
        """For each system, list which systems dominate it.

        Returns:
            Dict mapping system -> list of dominating systems.
        """
        points = self.compute_frontier(system_results)
        point_map = {p.system: p for p in points}

        dominators: Dict[str, List[str]] = {}
        for p in points:
            dominating = []
            for q in points:
                if q.system != p.system:
                    if q.accuracy >= p.accuracy and q.cost <= p.cost:
                        if q.accuracy > p.accuracy or q.cost < p.cost:
                            dominating.append(q.system)
            dominators[p.system] = dominating

        return dominators

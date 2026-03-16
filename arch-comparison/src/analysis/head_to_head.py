"""Head-to-head comparison analysis across systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.benchmark_suite import BenchmarkResults


@dataclass
class HeadToHeadReport:
    """Report from head-to-head comparison."""
    winner_table: Dict[str, str] = field(default_factory=dict)  # axis -> winner
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # system -> {axis: score}
    overall_winner: str = ""
    analysis_notes: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HeadToHeadAnalyzer:
    """Analyzes benchmark results for head-to-head comparison."""

    def analyze(self, results: BenchmarkResults) -> HeadToHeadReport:
        """Analyze benchmark results for head-to-head comparison.

        Args:
            results: BenchmarkResults from a full benchmark run.

        Returns:
            HeadToHeadReport with winner table and scores.
        """
        scores: Dict[str, Dict[str, float]] = {}
        winner_table: Dict[str, str] = {}
        notes: List[str] = []

        # Collect scores per system per axis
        all_systems = set()

        # Generalization scores
        for system, gen_result in results.generalization.items():
            all_systems.add(system)
            scores.setdefault(system, {})
            # Use average of in-domain and out-of-domain
            avg = (gen_result.in_domain_accuracy + gen_result.out_of_domain_accuracy) / 2
            scores[system]["generalization"] = avg

        # Interpretability scores
        for system, interp_result in results.interpretability.items():
            all_systems.add(system)
            scores.setdefault(system, {})
            scores[system]["interpretability"] = interp_result.overall_score

        # Robustness scores
        for system, robust_result in results.robustness.items():
            all_systems.add(system)
            scores.setdefault(system, {})
            # Use consistency as the robustness score
            scores[system]["robustness"] = robust_result.consistency

        # Determine winners per axis
        for axis in ["generalization", "interpretability", "robustness"]:
            best_system = ""
            best_score = -1.0
            for system in all_systems:
                s = scores.get(system, {}).get(axis, 0.0)
                if s > best_score:
                    best_score = s
                    best_system = system
            winner_table[axis] = best_system
            notes.append(f"{axis}: {best_system} wins with {best_score:.3f}")

        # Overall winner: most axes won
        win_counts: Dict[str, int] = {}
        for winner in winner_table.values():
            win_counts[winner] = win_counts.get(winner, 0) + 1
        overall_winner = max(win_counts, key=lambda s: win_counts[s]) if win_counts else ""

        return HeadToHeadReport(
            winner_table=winner_table,
            scores=scores,
            overall_winner=overall_winner,
            analysis_notes=notes,
        )

    def plot_radar_chart(self, report: HeadToHeadReport) -> Optional[Any]:
        """Generate a radar chart comparing systems.

        Returns a matplotlib figure or None if plotting fails.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            axes = ["generalization", "interpretability", "robustness"]
            systems = list(report.scores.keys())

            fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(8, 6))
            angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
            angles += angles[:1]

            for system in systems:
                values = [report.scores.get(system, {}).get(a, 0) for a in axes]
                values += values[:1]
                ax.plot(angles, values, label=system)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(axes)
            ax.set_title("System Comparison")
            ax.legend(loc="upper right")
            plt.close(fig)
            return fig
        except ImportError:
            return None

    def plot_per_axis_comparison(self, report: HeadToHeadReport) -> Optional[Any]:
        """Generate per-axis bar chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            axes = ["generalization", "interpretability", "robustness"]
            systems = list(report.scores.keys())

            fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 5))
            if len(axes) == 1:
                axs = [axs]

            for i, axis in enumerate(axes):
                vals = [report.scores.get(s, {}).get(axis, 0) for s in systems]
                axs[i].bar(systems, vals)
                axs[i].set_title(axis.capitalize())
                axs[i].set_ylim(0, 1)

            plt.tight_layout()
            plt.close(fig)
            return fig
        except ImportError:
            return None

    def generate_winner_table(self, report: HeadToHeadReport) -> str:
        """Generate a markdown winner table."""
        lines = ["| Axis | Winner | Score |", "|------|--------|-------|"]
        for axis, winner in report.winner_table.items():
            score = report.scores.get(winner, {}).get(axis, 0.0)
            lines.append(f"| {axis} | {winner} | {score:.3f} |")
        lines.append(f"\n**Overall Winner: {report.overall_winner}**")
        return "\n".join(lines)

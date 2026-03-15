"""Library growth analysis and visualization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.library.evolution import LibraryMetrics


class LibraryGrowthAnalyzer:
    """Analyzes how the rule library grows and evolves over time."""

    def __init__(self) -> None:
        self._snapshots: List[LibraryMetrics] = []
        self._pareto_history: List[List[Dict[str, float]]] = []

    def add_snapshot(self, metrics: LibraryMetrics) -> None:
        """Record a library metrics snapshot.

        Args:
            metrics: Current library metrics.
        """
        self._snapshots.append(metrics)

    def add_pareto_snapshot(
        self, pareto_front: List[Dict[str, float]]
    ) -> None:
        """Record a Pareto front snapshot.

        Args:
            pareto_front: List of dicts with 'accuracy' and 'complexity' keys.
        """
        self._pareto_history.append(pareto_front)

    def plot_library_growth(
        self, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot library growth metrics over time.

        Args:
            output_path: Path to save the plot.

        Returns:
            Path to saved plot, or None.
        """
        if not self._snapshots:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            iterations = list(range(len(self._snapshots)))
            sizes = [s.total_rules for s in self._snapshots]
            accuracies = [s.avg_accuracy for s in self._snapshots]
            qualities = [s.quality_score for s in self._snapshots]

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

            ax1.plot(iterations, sizes, "b-o", markersize=4)
            ax1.set_ylabel("Library Size")
            ax1.set_title("Library Growth")
            ax1.grid(True, alpha=0.3)

            ax2.plot(iterations, accuracies, "g-o", markersize=4)
            ax2.set_ylabel("Average Accuracy")
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3)

            ax3.plot(iterations, qualities, "r-o", markersize=4)
            ax3.set_ylabel("Quality Score")
            ax3.set_xlabel("Iteration")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                plt.close(fig)
                return None
        except ImportError:
            return None

    def plot_pareto_front_evolution(
        self, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot how the Pareto front evolves over iterations.

        Args:
            output_path: Path to save the plot.

        Returns:
            Path to saved plot, or None.
        """
        if not self._pareto_history:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            colors = plt.cm.viridis  # type: ignore
            n = len(self._pareto_history)

            for i, front in enumerate(self._pareto_history):
                if not front:
                    continue
                accs = [p.get("accuracy", 0) for p in front]
                comps = [p.get("complexity", 0) for p in front]
                color = colors(i / max(n - 1, 1))
                ax.scatter(comps, accs, c=[color], s=50, alpha=0.7,
                          label=f"Iter {i}")
                # Connect front
                paired = sorted(zip(comps, accs))
                ax.plot([p[0] for p in paired], [p[1] for p in paired],
                       c=color, alpha=0.3)

            ax.set_xlabel("BDM Complexity")
            ax.set_ylabel("Accuracy")
            ax.set_title("Pareto Front Evolution")
            if n <= 10:
                ax.legend()
            ax.grid(True, alpha=0.3)

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                plt.close(fig)
                return None
        except ImportError:
            return None

    def domain_coverage_over_time(self) -> List[int]:
        """Track number of unique domains covered over time.

        Returns:
            List of domain counts at each snapshot.
        """
        return [s.unique_domains for s in self._snapshots]

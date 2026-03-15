"""Collapse/escape analysis: tests whether the rule library resists collapse."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.library.evolution import LibraryEvolver, LibraryMetrics
from src.library.store import RuleStore


@dataclass
class CollapseEscapeResult:
    """Result of collapse/escape analysis."""

    metric_name: str
    values: List[float] = field(default_factory=list)
    is_collapsing: bool = False
    is_escaping: bool = False
    trend: float = 0.0  # positive = improving, negative = collapsing

    @property
    def status(self) -> str:
        if self.is_escaping:
            return "escaping"
        elif self.is_collapsing:
            return "collapsing"
        return "stable"


class CollapseEscapeAnalyzer:
    """Analyzes whether the rule library is collapsing or escaping.

    Collapse indicators:
    - Decreasing diversity (fewer unique domains)
    - Decreasing average accuracy
    - Increasing similarity between rules

    Escape indicators:
    - Growing library with maintained quality
    - Increasing domain coverage
    - Improving Pareto front
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the analyzer.

        Args:
            window_size: Number of recent snapshots to analyze.
        """
        self.window_size = window_size

    def analyze(
        self, metrics_history: List[LibraryMetrics]
    ) -> List[CollapseEscapeResult]:
        """Analyze metrics history for collapse/escape patterns.

        Args:
            metrics_history: List of library metrics over time.

        Returns:
            List of CollapseEscapeResult for different metrics.
        """
        if len(metrics_history) < 2:
            return []

        results = []

        # Analyze library size trend
        sizes = [m.total_rules for m in metrics_history]
        results.append(self._analyze_trend("library_size", sizes))

        # Analyze accuracy trend
        accuracies = [m.avg_accuracy for m in metrics_history]
        results.append(self._analyze_trend("avg_accuracy", accuracies))

        # Analyze domain diversity
        domains = [float(m.unique_domains) for m in metrics_history]
        results.append(self._analyze_trend("domain_diversity", domains))

        # Analyze quality score
        qualities = [m.quality_score for m in metrics_history]
        results.append(self._analyze_trend("quality_score", qualities))

        # Analyze coverage
        coverages = [m.coverage for m in metrics_history]
        results.append(self._analyze_trend("coverage", coverages))

        return results

    def _analyze_trend(
        self, name: str, values: List[float]
    ) -> CollapseEscapeResult:
        """Analyze the trend of a metric.

        Args:
            name: Metric name.
            values: Metric values over time.

        Returns:
            CollapseEscapeResult with trend analysis.
        """
        # Use recent window
        recent = values[-self.window_size:]

        # Compute trend as average of consecutive differences
        if len(recent) < 2:
            trend = 0.0
        else:
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            trend = sum(diffs) / len(diffs)

        # Determine collapse/escape
        threshold = 0.01  # small threshold for stability
        is_collapsing = trend < -threshold
        is_escaping = trend > threshold

        return CollapseEscapeResult(
            metric_name=name,
            values=values,
            is_collapsing=is_collapsing,
            is_escaping=is_escaping,
            trend=trend,
        )

    def plot_escape_vs_collapse(
        self,
        results: List[CollapseEscapeResult],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot escape vs. collapse analysis.

        Args:
            results: Analysis results.
            output_path: Path to save the plot.

        Returns:
            Path to saved plot, or None.
        """
        if not results:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_metrics = len(results)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
            if n_metrics == 1:
                axes = [axes]

            for ax, result in zip(axes, results):
                iterations = list(range(len(result.values)))
                color = (
                    "green" if result.is_escaping
                    else "red" if result.is_collapsing
                    else "blue"
                )
                ax.plot(iterations, result.values, f"-o", color=color, markersize=4)
                ax.set_ylabel(result.metric_name)
                ax.set_title(
                    f"{result.metric_name}: {result.status} "
                    f"(trend={result.trend:.4f})"
                )
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Iteration")
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

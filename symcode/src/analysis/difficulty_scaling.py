"""Difficulty scaling analysis: how accuracy varies with problem difficulty."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.verification.result_types import SolveResult
from src.utils.logging import get_logger

logger = get_logger("analysis.difficulty_scaling")


@dataclass
class ScalingCurve:
    """Data for a difficulty scaling curve."""

    difficulties: list[int] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    pipeline: str = ""


class DifficultyScaler:
    """Analyze how accuracy scales with problem difficulty."""

    def compute_scaling_curve(
        self,
        results: list[SolveResult],
        difficulties: list[int] | None = None,
        pipeline: str = "symcode",
    ) -> ScalingCurve:
        """Compute accuracy at each difficulty level.

        Args:
            results: Solve results.
            difficulties: Parallel list of difficulty values. If None,
                         reads ``_difficulty`` attribute from results.
            pipeline: Label for the pipeline.

        Returns:
            ScalingCurve with difficulty levels, accuracies, and counts.
        """
        groups: dict[int, list[bool]] = defaultdict(list)
        for i, r in enumerate(results):
            if difficulties is not None and i < len(difficulties):
                diff = difficulties[i]
            else:
                diff = getattr(r, "_difficulty", 0)
            groups[diff].append(r.correct)

        curve = ScalingCurve(pipeline=pipeline)
        for diff in sorted(groups.keys()):
            correct_list = groups[diff]
            curve.difficulties.append(diff)
            curve.accuracies.append(
                sum(correct_list) / len(correct_list) if correct_list else 0.0
            )
            curve.counts.append(len(correct_list))

        return curve

    def plot_scaling_curve(
        self,
        curves: list[ScalingCurve],
        output_path: str | None = None,
        title: str = "Accuracy vs Difficulty",
    ) -> Any:
        """Plot scaling curves. Returns the matplotlib figure.

        If matplotlib is not available, logs a warning and returns None.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        for curve in curves:
            ax.plot(
                curve.difficulties,
                curve.accuracies,
                marker="o",
                label=curve.pipeline,
                linewidth=2,
            )
            # Annotate counts
            for d, a, c in zip(curve.difficulties, curve.accuracies, curve.counts):
                ax.annotate(
                    f"n={c}",
                    (d, a),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=7,
                    ha="center",
                )

        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Scaling curve saved to %s", output_path)

        return fig

    def test_scaling_hypothesis(
        self,
        results: list[SolveResult],
        difficulties: list[int] | None = None,
    ) -> dict[str, Any]:
        """Test whether accuracy decreases with difficulty (Spearman rank correlation).

        Returns:
            - spearman_rho: Spearman correlation coefficient
            - p_value: p-value for the correlation
            - monotonically_decreasing: whether accuracy is strictly
              non-increasing with difficulty
        """
        curve = self.compute_scaling_curve(results, difficulties)

        if len(curve.difficulties) < 3:
            return {
                "spearman_rho": None,
                "p_value": None,
                "monotonically_decreasing": None,
                "message": "Not enough difficulty levels for correlation test",
            }

        # Check monotonically decreasing
        mono_dec = all(
            curve.accuracies[i] >= curve.accuracies[i + 1]
            for i in range(len(curve.accuracies) - 1)
        )

        # Spearman correlation
        rho, p_value = self._spearman(curve.difficulties, curve.accuracies)

        return {
            "spearman_rho": rho,
            "p_value": p_value,
            "monotonically_decreasing": mono_dec,
        }

    @staticmethod
    def _spearman(x: list[int | float], y: list[float]) -> tuple[float | None, float | None]:
        """Compute Spearman rank correlation.

        Tries scipy first; falls back to a simple implementation.
        """
        n = len(x)
        if n < 3:
            return None, None

        try:
            from scipy.stats import spearmanr

            res = spearmanr(x, y)
            return float(res.correlation), float(res.pvalue)
        except ImportError:
            pass

        # Manual Spearman computation
        def _rank(vals: list[float]) -> list[float]:
            sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
            ranks = [0.0] * len(vals)
            i = 0
            while i < len(sorted_vals):
                j = i
                while j < len(sorted_vals) and sorted_vals[j][1] == sorted_vals[i][1]:
                    j += 1
                avg_rank = (i + j - 1) / 2.0 + 1.0
                for k in range(i, j):
                    ranks[sorted_vals[k][0]] = avg_rank
                i = j
            return ranks

        rx = _rank([float(v) for v in x])
        ry = _rank(y)

        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        rho = 1.0 - 6.0 * d_sq / (n * (n * n - 1))

        # Approximate p-value using t-distribution
        import math

        if abs(rho) >= 1.0:
            p_value = 0.0
        else:
            t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))
            # Very rough p-value approximation
            p_value = 2.0 * (1.0 - min(0.9999, abs(t_stat) / (abs(t_stat) + n)))

        return rho, p_value

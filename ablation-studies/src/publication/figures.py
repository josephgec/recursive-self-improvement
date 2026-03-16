"""Publication-quality figure generation.

Generates figures using a mock/stub approach that works without matplotlib
being installed. When matplotlib is available, produces real figures.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.suites.base import AblationSuiteResult
from src.analysis.statistical_tests import PairwiseResult


# Colorblind-safe palette (Okabe-Ito)
COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # light blue
    "#D55E00",  # red-orange
    "#F0E442",  # yellow
    "#000000",  # black
]


class FigureData:
    """Data container for a figure (used when matplotlib is not available)."""

    def __init__(self, fig_type: str, data: Dict[str, Any],
                 title: str = "", xlabel: str = "", ylabel: str = ""):
        self.fig_type = fig_type
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.saved_path: Optional[str] = None

    def __repr__(self) -> str:
        return f"FigureData(type={self.fig_type!r}, title={self.title!r})"


class PublicationFigureGenerator:
    """Generate publication-quality figures.

    Works with or without matplotlib. When matplotlib is not available,
    returns FigureData objects that contain the data needed for plotting.
    """

    def __init__(self, font_family: str = "serif", font_size: int = 10,
                 colorblind_safe: bool = True, dpi: int = 300):
        self.font_family = font_family
        self.font_size = font_size
        self.colorblind_safe = colorblind_safe
        self.dpi = dpi
        self.palette = COLORBLIND_PALETTE if colorblind_safe else None
        self._has_matplotlib = False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self._plt = plt
            self._has_matplotlib = True
        except ImportError:
            self._plt = None

    def ablation_bar_chart(
        self,
        result: AblationSuiteResult,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> FigureData:
        """Generate a bar chart comparing all conditions."""
        conditions = result.get_all_condition_names()
        means = [result.get_mean_score(c) for c in conditions]
        stds = []
        for c in conditions:
            scores = result.get_scores(c)
            if len(scores) > 1:
                m = sum(scores) / len(scores)
                v = sum((x - m) ** 2 for x in scores) / (len(scores) - 1)
                stds.append(math.sqrt(v))
            else:
                stds.append(0.0)

        fig_title = title or f"Ablation Results: {result.suite_name}"

        fig_data = FigureData(
            fig_type="bar_chart",
            data={"conditions": conditions, "means": means, "stds": stds},
            title=fig_title,
            xlabel="Condition",
            ylabel="Accuracy",
        )

        if self._has_matplotlib and output_path:
            fig, ax = self._plt.subplots(figsize=(10, 6))
            self._apply_style(ax)
            colors = self._get_colors(len(conditions))
            bars = ax.bar(range(len(conditions)), means, yerr=stds,
                         color=colors, capsize=3, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Accuracy")
            ax.set_title(fig_title)
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            self._plt.close(fig)
            fig_data.saved_path = output_path

        return fig_data

    def improvement_curve_comparison(
        self,
        results: Dict[str, AblationSuiteResult],
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> FigureData:
        """Generate curves comparing improvement across suites."""
        suite_data = {}
        for suite_name, result in results.items():
            conditions = result.get_all_condition_names()
            means = sorted(
                [(c, result.get_mean_score(c)) for c in conditions],
                key=lambda x: x[1],
            )
            suite_data[suite_name] = means

        fig_title = title or "Improvement Curves Across Paradigms"

        fig_data = FigureData(
            fig_type="improvement_curve",
            data={"suites": suite_data},
            title=fig_title,
            xlabel="Condition (sorted by score)",
            ylabel="Accuracy",
        )

        if self._has_matplotlib and output_path:
            fig, ax = self._plt.subplots(figsize=(10, 6))
            self._apply_style(ax)
            colors = self._get_colors(len(results))
            for idx, (suite_name, means) in enumerate(suite_data.items()):
                x = list(range(len(means)))
                y = [m[1] for m in means]
                ax.plot(x, y, marker="o", label=suite_name, color=colors[idx])
            ax.set_xlabel("Condition (sorted)")
            ax.set_ylabel("Accuracy")
            ax.set_title(fig_title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            self._plt.close(fig)
            fig_data.saved_path = output_path

        return fig_data

    def contribution_waterfall(
        self,
        result: AblationSuiteResult,
        baseline: str = "full",
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> FigureData:
        """Generate a waterfall chart showing component contributions."""
        full_score = result.get_mean_score(baseline)
        conditions = [c for c in result.get_all_condition_names() if c != baseline]

        contributions = []
        for c in conditions:
            drop = full_score - result.get_mean_score(c)
            contributions.append((c, drop))

        # Sort by contribution magnitude
        contributions.sort(key=lambda x: x[1], reverse=True)

        fig_title = title or f"Component Contributions: {result.suite_name}"

        fig_data = FigureData(
            fig_type="waterfall",
            data={
                "baseline": baseline,
                "baseline_score": full_score,
                "contributions": contributions,
            },
            title=fig_title,
            xlabel="Ablated Component",
            ylabel="Performance Drop",
        )

        if self._has_matplotlib and output_path:
            fig, ax = self._plt.subplots(figsize=(10, 6))
            self._apply_style(ax)
            names = [c[0] for c in contributions]
            drops = [c[1] for c in contributions]
            colors = ["#D55E00" if d > 0 else "#009E73" for d in drops]
            ax.bar(range(len(names)), drops, color=colors, edgecolor="black",
                   linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_xlabel("Ablated Component")
            ax.set_ylabel("Performance Drop")
            ax.set_title(fig_title)
            ax.axhline(y=0, color="black", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            self._plt.close(fig)
            fig_data.saved_path = output_path

        return fig_data

    def pairwise_forest_plot(
        self,
        comparisons: List[PairwiseResult],
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> FigureData:
        """Generate a forest plot of pairwise comparisons."""
        plot_data = []
        for comp in comparisons:
            plot_data.append({
                "label": f"{comp.condition_a} vs {comp.condition_b}",
                "difference": comp.difference,
                "ci_lower": comp.ci_lower,
                "ci_upper": comp.ci_upper,
                "stars": comp.stars,
            })

        fig_title = title or "Pairwise Comparison Forest Plot"

        fig_data = FigureData(
            fig_type="forest_plot",
            data={"comparisons": plot_data},
            title=fig_title,
            xlabel="Difference in Accuracy",
            ylabel="Comparison",
        )

        if self._has_matplotlib and output_path:
            fig, ax = self._plt.subplots(figsize=(10, max(4, len(comparisons) * 0.8)))
            self._apply_style(ax)
            y_pos = list(range(len(plot_data)))
            diffs = [p["difference"] for p in plot_data]
            ci_lows = [p["ci_lower"] for p in plot_data]
            ci_highs = [p["ci_upper"] for p in plot_data]
            labels = [p["label"] + p["stars"] for p in plot_data]
            xerr_low = [d - cl for d, cl in zip(diffs, ci_lows)]
            xerr_high = [ch - d for d, ch in zip(diffs, ci_highs)]
            ax.errorbar(diffs, y_pos, xerr=[xerr_low, xerr_high],
                       fmt="o", color="#0072B2", capsize=4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Difference in Accuracy")
            ax.set_title(fig_title)
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            self._plt.close(fig)
            fig_data.saved_path = output_path

        return fig_data

    def _apply_style(self, ax: Any) -> None:
        """Apply publication style to an axis."""
        if not self._has_matplotlib:
            return
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._plt.rcParams["font.family"] = self.font_family
        self._plt.rcParams["font.size"] = self.font_size

    def _get_colors(self, n: int) -> List[str]:
        """Get n colors from the palette."""
        if self.palette:
            return [self.palette[i % len(self.palette)] for i in range(n)]
        return [f"C{i}" for i in range(n)]

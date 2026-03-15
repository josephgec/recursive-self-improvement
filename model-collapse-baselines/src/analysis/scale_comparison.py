"""Scale comparison analysis: 1B vs 7B model collapse dynamics.

Overlays metrics from two model scales to highlight how model size
affects the rate and character of collapse.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.collapse_curves import COLORS, _apply_style, _ensure_dir, _get_column

logger = logging.getLogger(__name__)


def plot_scale_comparison(
    metrics_1b: pd.DataFrame,
    metrics_7b: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
    ylabel: str | None = None,
) -> Path:
    """Overlay one metric from 1B and 7B runs on the same axes.

    Args:
        metrics_1b: Metrics DataFrame for the 1B model.
        metrics_7b: Metrics DataFrame for the 7B model.
        metric_name: Column name to plot.
        output_path: Where to save the figure.
        ylabel: Optional custom y-axis label.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    gens_1b = metrics_1b["generation"].values
    vals_1b = _get_column(metrics_1b, metric_name, default=0.0)
    ax.plot(gens_1b, vals_1b, marker="o", linewidth=2,
            label="1B", color="#1f77b4")

    gens_7b = metrics_7b["generation"].values
    vals_7b = _get_column(metrics_7b, metric_name, default=0.0)
    ax.plot(gens_7b, vals_7b, marker="s", linewidth=2,
            label="7B", color="#d62728", linestyle="--")

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title())
    ax.set_title(f"Scale Comparison: {metric_name.replace('_', ' ').title()}")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved scale comparison (%s) to %s", metric_name, output_path)
    return output_path


def plot_scale_comparison_panel(
    metrics_1b: pd.DataFrame,
    metrics_7b: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """2x3 panel comparing key metrics between 1B and 7B scales.

    Metrics plotted: entropy, KL divergence, self-BLEU, embedding variance,
    vocabulary usage, and distinct-2.

    Args:
        metrics_1b: Metrics DataFrame for the 1B model.
        metrics_7b: Metrics DataFrame for the 7B model.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    panel_metrics = [
        ("entropy", "Entropy"),
        ("kl_divergence", "KL Divergence"),
        ("self_bleu", "Self-BLEU"),
        ("embedding_variance", "Embedding Variance"),
        ("vocabulary_usage", "Vocab Usage"),
        ("distinct_2", "Distinct-2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (metric, label) in enumerate(panel_metrics):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        gens_1b = metrics_1b["generation"].values
        vals_1b = _get_column(metrics_1b, metric, default=0.0)
        ax.plot(gens_1b, vals_1b, marker="o", linewidth=1.5,
                label="1B", color="#1f77b4")

        gens_7b = metrics_7b["generation"].values
        vals_7b = _get_column(metrics_7b, metric, default=0.0)
        ax.plot(gens_7b, vals_7b, marker="s", linewidth=1.5,
                label="7B", color="#d62728", linestyle="--")

        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle("1B vs 7B Scale Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved scale comparison panel to %s", output_path)
    return output_path


def compute_scale_interaction_stats(
    metrics_1b: pd.DataFrame,
    metrics_7b: pd.DataFrame,
) -> dict[str, Any]:
    """Compute summary statistics comparing collapse dynamics across scales.

    Returns a dictionary with:
    - ``collapse_rate_ratio``: ratio of KL growth rates (1B / 7B)
    - ``fixed_point_generations``: dict with estimated fixed-point gen for each scale
    - ``entropy_floors``: dict with final entropy for each scale
    - ``variance_ratios``: dict with initial/final embedding variance ratios

    Args:
        metrics_1b: Metrics DataFrame for the 1B model.
        metrics_7b: Metrics DataFrame for the 7B model.

    Returns:
        Dictionary of scale interaction statistics.
    """
    stats: dict[str, Any] = {}

    # Collapse rate ratio: average KL growth per generation.
    kl_1b = _get_column(metrics_1b, "kl_divergence", default=0.0)
    kl_7b = _get_column(metrics_7b, "kl_divergence", default=0.0)

    rate_1b = _compute_growth_rate(kl_1b)
    rate_7b = _compute_growth_rate(kl_7b)

    stats["collapse_rate_ratio"] = (
        rate_1b / rate_7b if rate_7b != 0 else float("inf")
    )

    # Fixed point estimation: generation at which KL delta < threshold.
    stats["fixed_point_generations"] = {
        "1b": _estimate_fixed_point(kl_1b, tolerance=0.01),
        "7b": _estimate_fixed_point(kl_7b, tolerance=0.01),
    }

    # Entropy floors: final entropy value.
    entropy_1b = _get_column(metrics_1b, "entropy", default=0.0)
    entropy_7b = _get_column(metrics_7b, "entropy", default=0.0)
    stats["entropy_floors"] = {
        "1b": float(entropy_1b[-1]) if len(entropy_1b) > 0 else 0.0,
        "7b": float(entropy_7b[-1]) if len(entropy_7b) > 0 else 0.0,
    }

    # Variance ratios: initial vs final embedding variance.
    var_1b = _get_column(metrics_1b, "embedding_variance", default=0.0)
    var_7b = _get_column(metrics_7b, "embedding_variance", default=0.0)
    stats["variance_ratios"] = {
        "1b": _variance_ratio(var_1b),
        "7b": _variance_ratio(var_7b),
    }

    return stats


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _compute_growth_rate(values: np.ndarray) -> float:
    """Average per-step growth of an array."""
    if len(values) < 2:
        return 0.0
    deltas = np.diff(values)
    return float(np.mean(deltas))


def _estimate_fixed_point(
    kl_values: np.ndarray, tolerance: float = 0.01
) -> int | None:
    """Return the generation at which KL changes fall below tolerance,
    or ``None`` if no fixed point is reached."""
    if len(kl_values) < 2:
        return None
    for i in range(1, len(kl_values)):
        if abs(kl_values[i] - kl_values[i - 1]) < tolerance:
            return i
    return None


def _variance_ratio(values: np.ndarray) -> float:
    """Ratio of last to first value (final / initial), or 0 if no data."""
    if len(values) < 2:
        return 0.0
    if values[0] == 0:
        return 0.0
    return float(values[-1] / values[0])

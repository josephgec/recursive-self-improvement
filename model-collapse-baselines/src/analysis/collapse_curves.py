"""Visualization functions for model-collapse trajectory curves.

Generates publication-quality plots showing how key metrics evolve
across generations of recursive fine-tuning.
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

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Publication style
# ------------------------------------------------------------------

_STYLE_APPLIED = False

# Consistent colour palette.
COLORS = {
    "entropy": "#1f77b4",      # blue
    "kl": "#d62728",           # red
    "diversity": "#2ca02c",    # green
    "variance": "#9467bd",     # purple
    "js": "#ff7f0e",           # orange
    "alpha": "#8c564b",        # brown
    "tail_p01": "#d62728",     # red
    "tail_p05": "#ff7f0e",     # orange
    "tail_p10": "#bcbd22",     # yellow-green
}


def _apply_style() -> None:
    """Apply publication-quality plot style (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    _STYLE_APPLIED = True


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Individual curve functions
# ------------------------------------------------------------------


def plot_entropy_curve(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    fixed_point_gen: int | None = None,
) -> Path:
    """Plot entropy vs generation, optionally marking the fixed point.

    Expects ``metrics_df`` to have columns ``generation`` and ``entropy``.
    If ``entropy`` is missing, the column ``extra.entropy`` or a zero-filled
    series is used.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_path: Where to save the figure.
        fixed_point_gen: Generation index of the fixed point (if detected).

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    gens = metrics_df["generation"].values
    entropy = _get_column(metrics_df, "entropy", default=0.0)

    ax.plot(gens, entropy, marker="o", color=COLORS["entropy"],
            linewidth=2, label="Entropy")

    if fixed_point_gen is not None and fixed_point_gen in gens:
        idx = list(gens).index(fixed_point_gen)
        ax.axvline(x=fixed_point_gen, color="gray", linestyle="--",
                   alpha=0.7, label=f"Fixed point (gen {fixed_point_gen})")
        ax.scatter([fixed_point_gen], [entropy[idx]], color="red",
                   zorder=5, s=100, marker="*")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Entropy")
    ax.set_title("Predictive Entropy vs Generation")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved entropy curve to %s", output_path)
    return output_path


def plot_kl_curve(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot KL(P||Q_t) vs generation with JS divergence on secondary axis.

    Expects columns ``generation``, ``kl_divergence``, and optionally
    ``js_divergence``.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    gens = metrics_df["generation"].values
    kl = _get_column(metrics_df, "kl_divergence", default=0.0)

    ax1.plot(gens, kl, marker="o", color=COLORS["kl"],
             linewidth=2, label="KL(P||Q)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("KL Divergence", color=COLORS["kl"])
    ax1.tick_params(axis="y", labelcolor=COLORS["kl"])

    # JS on secondary axis.
    js = _get_column(metrics_df, "js_divergence", default=None)
    if js is not None:
        ax2 = ax1.twinx()
        ax2.plot(gens, js, marker="s", color=COLORS["js"],
                 linewidth=2, linestyle="--", label="JS Divergence")
        ax2.set_ylabel("JS Divergence", color=COLORS["js"])
        ax2.tick_params(axis="y", labelcolor=COLORS["js"])
        # Combined legend.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend()

    ax1.set_title("KL & JS Divergence vs Generation")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved KL curve to %s", output_path)
    return output_path


def plot_diversity_panel(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot a 2x2 diversity panel: distinct-n, self-BLEU, vocab usage, embedding variance.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    gens = metrics_df["generation"].values

    # Top-left: distinct-n
    ax = axes[0, 0]
    for n_val in [1, 2, 3, 4]:
        col = f"distinct_{n_val}"
        vals = _get_column(metrics_df, col, default=None)
        if vals is not None:
            ax.plot(gens, vals, marker="o", linewidth=1.5,
                    label=f"distinct-{n_val}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Distinct-N Ratio")
    ax.set_title("Distinct N-gram Ratios")
    ax.legend(fontsize=9)

    # Top-right: self-BLEU
    ax = axes[0, 1]
    self_bleu = _get_column(metrics_df, "self_bleu", default=0.0)
    ax.plot(gens, self_bleu, marker="o", color=COLORS["diversity"],
            linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Self-BLEU")
    ax.set_title("Self-BLEU (lower = more diverse)")

    # Bottom-left: vocabulary usage
    ax = axes[1, 0]
    vocab = _get_column(metrics_df, "vocabulary_usage",
                        alt_name="vocab_coverage", default=0.0)
    ax.plot(gens, vocab, marker="o", color=COLORS["diversity"],
            linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Unique Tokens")
    ax.set_title("Vocabulary Usage")

    # Bottom-right: embedding variance
    ax = axes[1, 1]
    emb_var = _get_column(metrics_df, "embedding_variance", default=0.0)
    ax.plot(gens, emb_var, marker="o", color=COLORS["variance"],
            linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Embedding Variance")
    ax.set_title("Embedding Variance (trace of cov)")

    fig.suptitle("Diversity Metrics vs Generation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved diversity panel to %s", output_path)
    return output_path


def plot_tail_erosion(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot stacked area chart of tail mass (p01, p05, p10) vs generation.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    gens = metrics_df["generation"].values
    p01 = _get_column(metrics_df, "tail_mass_p01", default=0.0)
    p05 = _get_column(metrics_df, "tail_mass_p05", default=0.0)
    p10 = _get_column(metrics_df, "tail_mass_p10", default=0.0)

    ax.stackplot(
        gens,
        p01,
        np.array(p05) - np.array(p01),
        np.array(p10) - np.array(p05),
        labels=["Bottom 1%", "1%-5%", "5%-10%"],
        colors=[COLORS["tail_p01"], COLORS["tail_p05"], COLORS["tail_p10"]],
        alpha=0.8,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Tail Probability Mass")
    ax.set_title("Tail Erosion: Rare-Token Mass vs Generation")
    ax.legend(loc="upper right")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved tail erosion plot to %s", output_path)
    return output_path


def plot_alpha_schedule_overlay(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Dual-axis plot: alpha_t schedule and KL divergence vs generation.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    gens = metrics_df["generation"].values
    alpha = _get_column(metrics_df, "alpha", default=0.0)
    kl = _get_column(metrics_df, "kl_divergence", default=0.0)

    ax1.fill_between(gens, 0, alpha, alpha=0.3, color=COLORS["alpha"],
                     label="alpha (real data fraction)")
    ax1.plot(gens, alpha, color=COLORS["alpha"], linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Alpha (real data fraction)", color=COLORS["alpha"])
    ax1.tick_params(axis="y", labelcolor=COLORS["alpha"])
    ax1.set_ylim(-0.05, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(gens, kl, marker="o", color=COLORS["kl"],
             linewidth=2, label="KL(P||Q)")
    ax2.set_ylabel("KL Divergence", color=COLORS["kl"])
    ax2.tick_params(axis="y", labelcolor=COLORS["kl"])

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Alpha Schedule & KL Divergence")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved alpha schedule overlay to %s", output_path)
    return output_path


def plot_all_curves(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
    fixed_point_gen: int | None = None,
) -> list[Path]:
    """Generate all standard collapse-curve plots.

    Args:
        metrics_df: DataFrame with per-generation metrics.
        output_dir: Directory to save all figures.
        fixed_point_gen: Generation index of the fixed point (if detected).

    Returns:
        List of paths to the saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    paths.append(plot_entropy_curve(
        metrics_df, output_dir / "entropy_curve.png", fixed_point_gen))
    paths.append(plot_kl_curve(
        metrics_df, output_dir / "kl_curve.png"))
    paths.append(plot_diversity_panel(
        metrics_df, output_dir / "diversity_panel.png"))
    paths.append(plot_tail_erosion(
        metrics_df, output_dir / "tail_erosion.png"))
    paths.append(plot_alpha_schedule_overlay(
        metrics_df, output_dir / "alpha_schedule_overlay.png"))

    logger.info("Generated %d plots in %s", len(paths), output_dir)
    return paths


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_column(
    df: pd.DataFrame,
    name: str,
    alt_name: str | None = None,
    default: Any = None,
) -> np.ndarray | None:
    """Safely extract a column from a DataFrame.

    Falls back to ``alt_name`` if ``name`` is absent, then to ``default``.
    If ``default`` is ``None``, returns ``None`` when the column is missing.
    """
    if name in df.columns:
        return df[name].values
    if alt_name and alt_name in df.columns:
        return df[alt_name].values
    # Try extracting from the 'extra' dict column.
    if "extra" in df.columns:
        try:
            vals = df["extra"].apply(
                lambda d: d.get(name, None) if isinstance(d, dict) else None
            )
            if vals.notna().any():
                return vals.fillna(0.0).values
        except Exception:
            pass
    if default is None:
        return None
    return np.full(len(df), default)

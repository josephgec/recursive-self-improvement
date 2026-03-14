"""Temporal curve and contamination-rate plots.

Produces publication-ready visualisations of how corpus similarity evolves
over time and what fraction of documents each time bin classifies as synthetic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_temporal_similarity_curve(
    curve_df: pd.DataFrame,
    output_path: Path,
    inflection_bin: str | None = None,
) -> None:
    """Plot the temporal similarity curve with optional inflection annotation.

    Parameters
    ----------
    curve_df:
        DataFrame as returned by
        :func:`src.embeddings.temporal_curves.compute_temporal_curve` with
        columns ``bin``, ``mean_similarity``, ``cross_similarity_to_reference``,
        ``similarity_p25``, ``similarity_p75``.
    output_path:
        Destination path for the PNG image (parent dirs created if needed).
    inflection_bin:
        If provided, a vertical dashed line is drawn at this bin and "Pre-LLM
        regime" / "Post-LLM regime" labels are added.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.15)

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = curve_df["bin"].astype(str).values
    x = np.arange(len(bins))
    mean_sim = curve_df["mean_similarity"].values
    p25 = curve_df["similarity_p25"].values
    p75 = curve_df["similarity_p75"].values
    cross_sim = curve_df["cross_similarity_to_reference"].values

    # Main similarity line with shaded IQR band
    ax.plot(x, mean_sim, marker="o", linewidth=2, label="Mean similarity (intra-bin)", color="#2c7bb6")
    ax.fill_between(x, p25, p75, alpha=0.2, color="#2c7bb6", label="p25\u2013p75 range")

    # Cross-similarity reference line
    ax.plot(
        x, cross_sim,
        marker="s", linewidth=2, linestyle="--",
        label="Cross-similarity to reference bin",
        color="#d7191c",
    )

    # Inflection point
    if inflection_bin is not None and inflection_bin in bins:
        idx = int(np.where(bins == inflection_bin)[0][0])
        ax.axvline(x=idx, color="#636363", linewidth=1.5, linestyle=":", zorder=1)

        # "Pre-LLM regime" and "Post-LLM regime" labels
        y_top = ax.get_ylim()[1]
        ax.text(
            idx - 0.3, y_top * 0.97, "Pre-LLM regime",
            ha="right", va="top", fontsize=11, fontstyle="italic", color="#636363",
        )
        ax.text(
            idx + 0.3, y_top * 0.97, "Post-LLM regime",
            ha="left", va="top", fontsize=11, fontstyle="italic", color="#636363",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_xlabel("Time bin (year)")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Temporal Similarity Curve")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved temporal similarity curve to %s", output_path)


def plot_contamination_rate(
    bin_labels: list[str],
    synthetic_fractions: list[float],
    output_path: Path,
) -> None:
    """Bar chart of per-bin contamination rate with a green-to-red gradient.

    Parameters
    ----------
    bin_labels:
        Time-bin labels (e.g. ``["2013", "2014", ..., "2024"]``).
    synthetic_fractions:
        Fraction of documents classified as synthetic in each bin, values
        in ``[0, 1]``.
    output_path:
        Destination path for the PNG image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.15)

    fig, ax = plt.subplots(figsize=(12, 6))

    fracs = np.asarray(synthetic_fractions, dtype=float)
    x = np.arange(len(bin_labels))

    # Build a green-to-red colour gradient based on fraction value
    cmap = plt.cm.RdYlGn_r  # Green at 0, red at 1
    norm = plt.Normalize(vmin=0.0, vmax=max(fracs.max(), 0.01))
    colours = [cmap(norm(f)) for f in fracs]

    bars = ax.bar(x, fracs, color=colours, edgecolor="white", linewidth=0.8)

    # Add value labels above bars
    for bar, frac in zip(bars, fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{frac:.1%}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Time bin (year)")
    ax.set_ylabel("Fraction classified as synthetic")
    ax.set_title("Contamination Rate by Time Bin")
    ax.set_ylim(0, min(max(fracs.max() * 1.25, 0.05), 1.0))

    # Colour-bar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Contamination level")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved contamination rate chart to %s", output_path)

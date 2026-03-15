"""Phase diagrams for the model-collapse experiment matrix.

Visualises how collapse metrics vary across the schedule x generation
parameter space, making it easy to identify safe vs collapsed regions.
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

from src.analysis.collapse_curves import _apply_style, _ensure_dir

logger = logging.getLogger(__name__)


def plot_phase_diagram(
    all_runs: dict[str, pd.DataFrame],
    metric_name: str,
    output_path: str | Path,
) -> Path:
    """Plot a heatmap: x=generation, y=schedule, colour=metric value.

    Args:
        all_runs: Mapping of schedule name to its metrics DataFrame.
            Each DataFrame must have ``generation`` and ``metric_name``
            columns.
        metric_name: Column name of the metric to visualise.
        output_path: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    # Build a matrix: rows = schedules, columns = generations.
    schedule_names = sorted(all_runs.keys())
    if not schedule_names:
        logger.warning("No runs provided for phase diagram")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # Determine the generation range across all runs.
    all_gens: set[int] = set()
    for df in all_runs.values():
        all_gens.update(df["generation"].tolist())
    gen_list = sorted(all_gens)

    matrix = np.full((len(schedule_names), len(gen_list)), np.nan)

    for row_idx, sched in enumerate(schedule_names):
        df = all_runs[sched]
        for _, record in df.iterrows():
            gen = int(record["generation"])
            if gen in gen_list:
                col_idx = gen_list.index(gen)
                val = record.get(metric_name, np.nan)
                if val is not None:
                    matrix[row_idx, col_idx] = float(val)

    fig, ax = plt.subplots(figsize=(max(8, len(gen_list) * 0.8),
                                     max(4, len(schedule_names) * 0.8)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(g) for g in gen_list],
        yticklabels=schedule_names,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric_name.replace("_", " ").title()},
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Schedule")
    ax.set_title(f"Phase Diagram: {metric_name.replace('_', ' ').title()}")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved phase diagram (%s) to %s", metric_name, output_path)
    return output_path


def plot_collapse_boundary(
    all_runs: dict[str, pd.DataFrame],
    collapse_threshold: float,
    output_path: str | Path,
    metric_name: str = "kl_divergence",
) -> Path:
    """Plot a contour showing safe vs collapsed regions.

    The boundary is defined as the contour where
    ``metric_name == collapse_threshold``.

    Args:
        all_runs: Mapping of schedule name to its metrics DataFrame.
        collapse_threshold: Value of the metric that defines collapse.
        output_path: Where to save the figure.
        metric_name: Metric used to define collapse (default: ``kl_divergence``).

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    output_path = Path(output_path)
    _ensure_dir(output_path)

    schedule_names = sorted(all_runs.keys())
    if not schedule_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    all_gens: set[int] = set()
    for df in all_runs.values():
        all_gens.update(df["generation"].tolist())
    gen_list = sorted(all_gens)

    matrix = np.full((len(schedule_names), len(gen_list)), np.nan)

    for row_idx, sched in enumerate(schedule_names):
        df = all_runs[sched]
        for _, record in df.iterrows():
            gen = int(record["generation"])
            if gen in gen_list:
                col_idx = gen_list.index(gen)
                val = record.get(metric_name, np.nan)
                if val is not None:
                    matrix[row_idx, col_idx] = float(val)

    fig, ax = plt.subplots(figsize=(max(8, len(gen_list) * 0.8),
                                     max(4, len(schedule_names) * 0.8)))

    # Use imshow for the background and overlay a contour.
    x = np.arange(len(gen_list))
    y = np.arange(len(schedule_names))
    X, Y = np.meshgrid(x, y)

    # Replace NaN with 0 for contouring.
    matrix_filled = np.nan_to_num(matrix, nan=0.0)

    im = ax.imshow(
        matrix_filled, aspect="auto", origin="lower",
        cmap="RdYlGn_r", interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label=metric_name.replace("_", " ").title())

    # Contour at the collapse threshold.
    try:
        contour = ax.contour(
            X, Y, matrix_filled,
            levels=[collapse_threshold],
            colors=["black"],
            linewidths=2,
        )
        ax.clabel(contour, inline=True, fontsize=10,
                  fmt=f"threshold={collapse_threshold:.2f}")
    except Exception:
        # Contour can fail if data is too sparse or uniform.
        logger.debug("Contour plotting skipped (data may be too sparse)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gen_list])
    ax.set_yticks(y)
    ax.set_yticklabels(schedule_names)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Schedule")
    ax.set_title("Collapse Boundary (Safe vs Collapsed Regions)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved collapse boundary to %s", output_path)
    return output_path

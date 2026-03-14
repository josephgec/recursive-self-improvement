"""Feature distribution plots: human vs. synthetic document comparison."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_feature_distributions(
    human_features: pd.DataFrame,
    synthetic_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grid of overlaid histograms comparing human and synthetic feature distributions.

    Parameters
    ----------
    human_features:
        Feature matrix for documents classified as human-authored.  Each
        column is a feature; metadata columns ``doc_id`` and ``timestamp``
        are silently ignored.
    synthetic_features:
        Feature matrix for documents classified as synthetic.  Must share
        the same feature columns as *human_features*.
    output_path:
        Destination path for the PNG image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _NON_FEATURE_COLS = {"doc_id", "timestamp"}

    feature_cols = [
        c for c in human_features.columns if c not in _NON_FEATURE_COLS
    ]

    if not feature_cols:
        logger.warning("No feature columns found; skipping distribution plot")
        return

    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False,
    )

    for idx, feature in enumerate(feature_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        if feature in human_features.columns:
            ax.hist(
                human_features[feature].dropna(),
                bins=30, alpha=0.55, label="Human", color="#2c7bb6",
                density=True, edgecolor="white", linewidth=0.5,
            )
        if feature in synthetic_features.columns:
            ax.hist(
                synthetic_features[feature].dropna(),
                bins=30, alpha=0.55, label="Synthetic", color="#d7191c",
                density=True, edgecolor="white", linewidth=0.5,
            )

        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_features, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Feature Distributions: Human vs. Synthetic", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature distribution plot to %s", output_path)

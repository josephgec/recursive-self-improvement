from __future__ import annotations

"""Analysis utilities for EPPO training results."""

import numpy as np

from ..eppo.trainer import EPPOEpochResult


def analyze_eppo_training(epoch_results: list[EPPOEpochResult]) -> dict:
    """Analyze EPPO training results.

    Args:
        epoch_results: List of epoch results from training.

    Returns:
        Dictionary with analysis metrics.
    """
    if not epoch_results:
        return {"status": "no_data"}

    entropies = [e.mean_entropy for e in epoch_results]
    losses = [e.mean_combined_loss for e in epoch_results]
    betas = [e.final_beta for e in epoch_results]

    # Entropy trend
    if len(entropies) >= 2:
        entropy_slope = float(np.polyfit(range(len(entropies)), entropies, 1)[0])
    else:
        entropy_slope = 0.0

    # Loss trend
    if len(losses) >= 2:
        loss_slope = float(np.polyfit(range(len(losses)), losses, 1)[0])
    else:
        loss_slope = 0.0

    # Entropy stability
    entropy_cv = float(np.std(entropies) / np.mean(entropies)) if np.mean(entropies) > 0 else 0.0

    return {
        "num_epochs": len(epoch_results),
        "entropy": {
            "initial": entropies[0],
            "final": entropies[-1],
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
            "mean": float(np.mean(entropies)),
            "slope": entropy_slope,
            "cv": entropy_cv,
        },
        "loss": {
            "initial": losses[0],
            "final": losses[-1],
            "mean": float(np.mean(losses)),
            "slope": loss_slope,
        },
        "beta": {
            "initial": betas[0],
            "final": betas[-1],
            "decay_ratio": betas[-1] / betas[0] if betas[0] > 0 else 0.0,
        },
        "healthy": entropy_slope > -0.1 and entropy_cv < 0.5,
    }

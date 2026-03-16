"""Threshold optimization for GDI calibration."""

from typing import Dict, List, Optional, Tuple

from .roc_analysis import compute_auc, compute_roc_curve, find_best_threshold


class ThresholdOptimizer:
    """Finds optimal thresholds for GDI alert levels.

    Uses ROC analysis to find thresholds that maximize detection
    of unhealthy states while minimizing false positives.
    """

    def find_optimal_threshold(
        self,
        gdi_values: List[float],
        health_labels: List[int],
    ) -> float:
        """Find the optimal binary threshold via ROC analysis.

        Args:
            gdi_values: List of GDI scores.
            health_labels: Binary labels (1 = unhealthy, 0 = healthy).

        Returns:
            Optimal threshold value.
        """
        fpr, tpr, thresholds = compute_roc_curve(gdi_values, health_labels)
        return find_best_threshold(fpr, tpr, thresholds)

    def plot_roc(
        self,
        gdi_values: List[float],
        health_labels: List[int],
    ) -> Dict[str, object]:
        """Compute ROC data (no actual plotting, returns data).

        Args:
            gdi_values: List of GDI scores.
            health_labels: Binary labels (1 = unhealthy, 0 = healthy).

        Returns:
            Dictionary with ROC curve data and AUC.
        """
        fpr, tpr, thresholds = compute_roc_curve(gdi_values, health_labels)
        auc = compute_auc(fpr, tpr)
        optimal = find_best_threshold(fpr, tpr, thresholds)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc,
            "optimal_threshold": optimal,
        }

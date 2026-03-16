"""Collapse-based calibration for GDI thresholds."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..composite.gdi import GoalDriftIndex
from .threshold_optimizer import ThresholdOptimizer
from .roc_analysis import compute_auc, compute_roc_curve


@dataclass
class CalibratedThresholds:
    """Thresholds calibrated from collapse data."""
    green_max: float
    yellow_max: float
    orange_max: float
    red_min: float
    auc: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollapseCalibrator:
    """Calibrates GDI thresholds from collapse trajectory data.

    Given data showing an agent's progression from healthy to collapsed,
    finds optimal thresholds for each alert level.
    """

    def __init__(self):
        self.optimizer = ThresholdOptimizer()

    def calibrate(
        self,
        gdi: GoalDriftIndex,
        collapse_data: List[Dict[str, Any]],
    ) -> CalibratedThresholds:
        """Calibrate thresholds from collapse trajectory data.

        Each entry in collapse_data should have:
            - "outputs": List[str] of agent outputs at this generation
            - "reference": List[str] of reference outputs
            - "health": str, one of "healthy", "degraded", "collapsed"

        Args:
            gdi: GoalDriftIndex instance for computing scores.
            collapse_data: List of generation snapshots.

        Returns:
            CalibratedThresholds optimized for this collapse trajectory.
        """
        gdi_scores = []
        health_labels = []

        for entry in collapse_data:
            outputs = entry["outputs"]
            reference = entry["reference"]
            health = entry["health"]

            result = gdi.compute(outputs, reference)
            gdi_scores.append(result.composite_score)

            # Binary labels for ROC
            if health == "healthy":
                health_labels.append(0)
            else:
                health_labels.append(1)

        if not gdi_scores:
            return CalibratedThresholds(
                green_max=0.15,
                yellow_max=0.40,
                orange_max=0.70,
                red_min=0.70,
            )

        # Find optimal binary threshold
        optimal = self.optimizer.find_optimal_threshold(gdi_scores, health_labels)

        # Compute AUC
        fpr, tpr, thresholds = compute_roc_curve(gdi_scores, health_labels)
        auc = compute_auc(fpr, tpr)

        # Derive multi-level thresholds from optimal binary threshold
        # Green: well below optimal, Yellow: approaching, Orange: at optimal, Red: above
        green_max = optimal * 0.4
        yellow_max = optimal * 0.75
        orange_max = optimal
        red_min = optimal

        return CalibratedThresholds(
            green_max=min(1.0, green_max),
            yellow_max=min(1.0, yellow_max),
            orange_max=min(1.0, orange_max),
            red_min=min(1.0, red_min),
            auc=auc,
            metadata={
                "optimal_binary_threshold": optimal,
                "num_samples": len(gdi_scores),
                "gdi_scores": gdi_scores,
                "health_labels": health_labels,
            },
        )

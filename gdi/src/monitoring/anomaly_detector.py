"""Anomaly detection for GDI time series."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Anomaly:
    """A detected anomaly in GDI history."""
    index: int
    score: float
    z_score: float
    direction: str  # "high" or "low"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Z-score based anomaly detection for GDI time series.

    Flags scores that deviate significantly from the rolling mean.
    """

    def __init__(self, z_threshold: float = 2.5, min_history: int = 5):
        """Initialize anomaly detector.

        Args:
            z_threshold: Z-score threshold for anomaly detection.
            min_history: Minimum history length before detection starts.
        """
        self.z_threshold = z_threshold
        self.min_history = min_history

    def detect(self, history: List[float]) -> List[Anomaly]:
        """Detect anomalies in GDI score history.

        Args:
            history: List of GDI composite scores.

        Returns:
            List of detected anomalies.
        """
        if len(history) < self.min_history:
            return []

        anomalies = []
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return []

        for i, score in enumerate(history):
            z = (score - mean) / std
            if abs(z) >= self.z_threshold:
                direction = "high" if z > 0 else "low"
                anomalies.append(
                    Anomaly(
                        index=i,
                        score=score,
                        z_score=z,
                        direction=direction,
                    )
                )

        return anomalies

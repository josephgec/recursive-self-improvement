"""Detect divergence between behavioral and internal representation changes."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

from src.probing.diff import ActivationDiffResult
from src.anomaly.ratio_monitor import RatioMonitor


@dataclass
class DivergenceCheckResult:
    """Result of a divergence check."""
    divergence_ratio: float
    internal_change: float
    behavioral_change: float
    z_score: float
    is_anomalous: bool
    safety_flag: bool
    safety_change_ratio: float
    iteration: int
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "divergence_ratio": self.divergence_ratio,
            "internal_change": self.internal_change,
            "behavioral_change": self.behavioral_change,
            "z_score": self.z_score,
            "is_anomalous": self.is_anomalous,
            "safety_flag": self.safety_flag,
            "safety_change_ratio": self.safety_change_ratio,
            "iteration": self.iteration,
            "details": self.details,
        }


class BehavioralInternalDivergenceDetector:
    """Detect divergence between behavioral outputs and internal representations.

    A high divergence ratio (internal >> behavioral) suggests the model is
    changing internally without visible behavioral change — a potential
    sign of deceptive alignment or silent reorganization.
    """

    def __init__(self, ratio_threshold: float = 3.0,
                 z_score_threshold: float = 2.5,
                 safety_factor: float = 2.0,
                 history_window: int = 20):
        self.ratio_threshold = ratio_threshold
        self.z_score_threshold = z_score_threshold
        self.safety_factor = safety_factor
        self.ratio_monitor = RatioMonitor(
            window_size=history_window,
            z_score_threshold=z_score_threshold,
        )
        self.iteration = 0

    def check(self, activation_diff: ActivationDiffResult,
              behavioral_change: float,
              iteration: Optional[int] = None) -> DivergenceCheckResult:
        """Check for behavioral/internal divergence.

        Args:
            activation_diff: Result from ActivationDiff.compute()
            behavioral_change: Magnitude of behavioral change [0, 1]
            iteration: Optional iteration number

        Returns:
            DivergenceCheckResult with anomaly flags
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1

        internal_change = activation_diff.overall_change_magnitude

        # Compute divergence ratio
        if behavioral_change > 1e-10:
            divergence_ratio = internal_change / behavioral_change
        elif internal_change > 1e-10:
            divergence_ratio = internal_change * 100  # Large ratio if no behavioral change
        else:
            divergence_ratio = 1.0  # Both near zero

        # Record and compute z-score
        self.ratio_monitor.record(divergence_ratio)
        z_score = self.ratio_monitor.compute_z_score()

        # Anomaly detection
        is_anomalous = (
            divergence_ratio > self.ratio_threshold or
            abs(z_score) > self.z_score_threshold
        )

        # Safety-specific flag
        safety_flag = activation_diff.safety_disproportionate
        safety_change_ratio = activation_diff.safety_change_ratio

        return DivergenceCheckResult(
            divergence_ratio=divergence_ratio,
            internal_change=internal_change,
            behavioral_change=behavioral_change,
            z_score=z_score,
            is_anomalous=is_anomalous,
            safety_flag=safety_flag,
            safety_change_ratio=safety_change_ratio,
            iteration=self.iteration,
            details={
                "ratio_threshold": self.ratio_threshold,
                "z_score_threshold": self.z_score_threshold,
                "ratio_history_len": len(self.ratio_monitor.history),
                "ratio_mean": self.ratio_monitor.get_mean(),
                "ratio_std": self.ratio_monitor.get_std(),
            },
        )

    def get_history(self) -> List[float]:
        """Return ratio history."""
        return self.ratio_monitor.get_history()

    def reset(self) -> None:
        """Reset detector state."""
        self.ratio_monitor = RatioMonitor(
            window_size=self.ratio_monitor.window_size,
            z_score_threshold=self.ratio_monitor.z_score_threshold,
        )
        self.iteration = 0

"""Fixed point detector: detect convergence and stability in RSI iterations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FixedPointStatus:
    """Status of fixed point detection."""
    is_fixed_point: bool
    iteration_detected: Optional[int]
    accuracy_at_detection: float
    entropy_at_detection: float
    convergence_type: str  # "stable", "oscillating", "diverging", "none"
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


class FixedPointDetector:
    """Detect fixed points in RSI improvement trajectories."""

    def __init__(
        self,
        accuracy_tolerance: float = 0.005,
        entropy_tolerance: float = 0.05,
        window_size: int = 3,
    ) -> None:
        self._acc_tol = accuracy_tolerance
        self._ent_tol = entropy_tolerance
        self._window = window_size

    def detect(
        self,
        accuracy: List[float],
        entropy: List[float],
    ) -> FixedPointStatus:
        """Detect fixed point from accuracy and entropy sequences."""
        if len(accuracy) < self._window or len(entropy) < self._window:
            return FixedPointStatus(
                is_fixed_point=False,
                iteration_detected=None,
                accuracy_at_detection=accuracy[-1] if accuracy else 0.0,
                entropy_at_detection=entropy[-1] if entropy else 0.0,
                convergence_type="none",
                confidence=0.0,
            )

        # Check for accuracy convergence
        acc_converged, acc_iter = self._check_convergence(
            accuracy, self._acc_tol
        )
        # Check for entropy convergence
        ent_converged, ent_iter = self._check_convergence(
            entropy, self._ent_tol
        )

        # Check for oscillation
        is_oscillating = self._check_oscillation(accuracy)

        # Determine type
        if acc_converged and ent_converged:
            convergence_type = "stable"
            is_fp = True
            detection_iter = max(acc_iter or 0, ent_iter or 0)
            confidence = 0.9
        elif is_oscillating:
            convergence_type = "oscillating"
            is_fp = False
            detection_iter = None
            confidence = 0.5
        elif acc_converged:
            convergence_type = "stable"
            is_fp = True
            detection_iter = acc_iter
            confidence = 0.7
        elif self._check_divergence(accuracy):
            convergence_type = "diverging"
            is_fp = False
            detection_iter = None
            confidence = 0.3
        else:
            convergence_type = "none"
            is_fp = False
            detection_iter = None
            confidence = 0.1

        return FixedPointStatus(
            is_fixed_point=is_fp,
            iteration_detected=detection_iter,
            accuracy_at_detection=accuracy[-1],
            entropy_at_detection=entropy[-1],
            convergence_type=convergence_type,
            confidence=confidence,
        )

    def _check_convergence(
        self, values: List[float], tolerance: float
    ) -> Tuple[bool, Optional[int]]:
        """Check if values have converged within tolerance."""
        for i in range(len(values) - self._window + 1):
            window = values[i : i + self._window]
            if max(window) - min(window) <= tolerance:
                return True, i
        return False, None

    def _check_oscillation(
        self, values: List[float], min_oscillations: int = 3
    ) -> bool:
        """Check if values are oscillating."""
        if len(values) < min_oscillations + 1:
            return False
        direction_changes = 0
        for i in range(2, len(values)):
            prev_dir = values[i - 1] - values[i - 2]
            curr_dir = values[i] - values[i - 1]
            if prev_dir * curr_dir < 0:
                direction_changes += 1
        return direction_changes >= min_oscillations

    def _check_divergence(self, values: List[float]) -> bool:
        """Check if values are monotonically decreasing (diverging from improvement)."""
        if len(values) < 3:
            return False
        recent = values[-3:]
        return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))

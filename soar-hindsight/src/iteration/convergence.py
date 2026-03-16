"""Convergence detection for the SOAR loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ConvergenceDetector:
    """Detect when the SOAR loop has converged (plateau in improvement).

    Convergence is detected when the metric hasn't improved by more than
    `min_improvement` for `patience` consecutive checks.
    """

    def __init__(self, patience: int = 3, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self._values: List[float] = []
        self._no_improve_count: int = 0
        self._best_value: float = float("-inf")
        self._converged: bool = False

    @property
    def is_converged(self) -> bool:
        return self._converged

    @property
    def best_value(self) -> float:
        return self._best_value if self._best_value > float("-inf") else 0.0

    @property
    def no_improve_count(self) -> int:
        return self._no_improve_count

    def check(self, value: float) -> bool:
        """Check a new metric value and return True if converged."""
        self._values.append(value)

        if value > self._best_value + self.min_improvement:
            self._best_value = value
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            self._converged = True

        return self._converged

    def reset(self) -> None:
        """Reset the convergence detector."""
        self._values.clear()
        self._no_improve_count = 0
        self._best_value = float("-inf")
        self._converged = False

    def summary(self) -> Dict[str, Any]:
        return {
            "converged": self._converged,
            "best_value": round(self.best_value, 4),
            "no_improve_count": self._no_improve_count,
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "n_checks": len(self._values),
        }

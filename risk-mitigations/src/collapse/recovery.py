"""Collapse recovery strategies.

Provides mechanisms to recover from detected or imminent model collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_STRATEGIES = ("increase_alpha", "rollback_n", "reset_to_baseline")


@dataclass
class RecoveryAction:
    """Record of a recovery action taken."""
    strategy: str
    params: Dict[str, Any]
    success: bool
    message: str


class CollapseRecovery:
    """Recovery mechanisms for model collapse.

    Strategies:
    - increase_alpha: Increase the clean data fraction
    - rollback_n: Roll back n checkpoints
    - reset_to_baseline: Reset to the initial baseline model
    """

    def __init__(self):
        self._recovery_history: List[RecoveryAction] = []
        self._current_alpha: float = 0.5
        self._checkpoint_index: int = 10  # Mock: assume we're at checkpoint 10
        self._baseline_checkpoint: int = 0

    def recover(
        self,
        strategy: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """Execute a recovery strategy.

        Args:
            strategy: One of 'increase_alpha', 'rollback_n', 'reset_to_baseline'.
            params: Strategy-specific parameters.

        Returns:
            RecoveryAction describing what was done.
        """
        params = params or {}

        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid: {VALID_STRATEGIES}"
            )

        if strategy == "increase_alpha":
            action = self._increase_alpha(params)
        elif strategy == "rollback_n":
            action = self._rollback_n(params)
        elif strategy == "reset_to_baseline":
            action = self._reset_to_baseline(params)
        else:
            raise ValueError(f"Unhandled strategy: {strategy}")

        self._recovery_history.append(action)
        return action

    def _increase_alpha(self, params: Dict[str, Any]) -> RecoveryAction:
        """Increase the clean data fraction."""
        increase_by = params.get("increase_by", 0.1)
        new_alpha = min(self._current_alpha + increase_by, 1.0)
        old_alpha = self._current_alpha
        self._current_alpha = new_alpha

        return RecoveryAction(
            strategy="increase_alpha",
            params={"old_alpha": old_alpha, "new_alpha": new_alpha},
            success=True,
            message=f"Alpha increased from {old_alpha:.2f} to {new_alpha:.2f}",
        )

    def _rollback_n(self, params: Dict[str, Any]) -> RecoveryAction:
        """Roll back n checkpoints."""
        n = params.get("n", 3)
        if n <= 0:
            return RecoveryAction(
                strategy="rollback_n",
                params={"n": n},
                success=False,
                message="Cannot roll back by 0 or fewer checkpoints",
            )

        old_index = self._checkpoint_index
        new_index = max(self._checkpoint_index - n, self._baseline_checkpoint)
        self._checkpoint_index = new_index

        return RecoveryAction(
            strategy="rollback_n",
            params={"n": n, "old_checkpoint": old_index, "new_checkpoint": new_index},
            success=True,
            message=f"Rolled back from checkpoint {old_index} to {new_index}",
        )

    def _reset_to_baseline(self, params: Dict[str, Any]) -> RecoveryAction:
        """Reset to the initial baseline model."""
        old_index = self._checkpoint_index
        self._checkpoint_index = self._baseline_checkpoint
        self._current_alpha = 0.5

        return RecoveryAction(
            strategy="reset_to_baseline",
            params={"old_checkpoint": old_index, "baseline": self._baseline_checkpoint},
            success=True,
            message=f"Reset from checkpoint {old_index} to baseline {self._baseline_checkpoint}",
        )

    def recommend_strategy(
        self, metrics: Dict[str, Any]
    ) -> str:
        """Recommend a recovery strategy based on current metrics.

        Args:
            metrics: Dict with keys like 'severity', 'iterations_declining', etc.

        Returns:
            Recommended strategy name.
        """
        severity = metrics.get("severity", "low")
        iterations_declining = metrics.get("iterations_declining", 0)

        if severity == "critical" or iterations_declining > 10:
            return "reset_to_baseline"
        elif severity == "high" or iterations_declining > 5:
            return "rollback_n"
        else:
            return "increase_alpha"

    def get_history(self) -> List[RecoveryAction]:
        """Return recovery action history."""
        return list(self._recovery_history)

    @property
    def current_alpha(self) -> float:
        return self._current_alpha

    @property
    def current_checkpoint(self) -> int:
        return self._checkpoint_index

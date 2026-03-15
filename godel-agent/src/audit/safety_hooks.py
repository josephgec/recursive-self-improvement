"""Safety hooks for monitoring and limiting self-modification."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.agent import IterationResult

logger = logging.getLogger(__name__)


class SafetyHooks:
    """Safety checks to prevent runaway self-modification."""

    def __init__(
        self,
        max_complexity_ratio: float = 5.0,
        max_modifications_in_window: int = 3,
        window_size: int = 5,
    ) -> None:
        self.max_complexity_ratio = max_complexity_ratio
        self.max_modifications_in_window = max_modifications_in_window
        self.window_size = window_size
        self._initial_complexity: float | None = None

    def set_initial_complexity(self, complexity: float) -> None:
        """Set the baseline complexity for ratio checks."""
        self._initial_complexity = complexity

    def check_complexity_bounds(self, current_complexity: float) -> bool:
        """Check if complexity is within bounds (5x initial).

        Returns True if within bounds, False if exceeded.
        """
        if self._initial_complexity is None or self._initial_complexity == 0:
            return True

        ratio = current_complexity / self._initial_complexity
        if ratio > self.max_complexity_ratio:
            logger.warning(
                f"Complexity ratio {ratio:.2f} exceeds limit {self.max_complexity_ratio}"
            )
            return False
        return True

    def check_modification_rate(
        self,
        iteration_results: list[Any],
        current_iteration: int,
    ) -> bool:
        """Check if modification rate is acceptable (>3 in 5 iterations = too fast).

        Returns True if rate is acceptable, False if too fast.
        """
        window_start = max(0, current_iteration - self.window_size)
        recent = iteration_results[window_start:]

        modification_count = sum(
            1 for r in recent
            if getattr(r, "modification_applied", False)
        )

        if modification_count >= self.max_modifications_in_window:
            logger.warning(
                f"Modification rate {modification_count}/{self.window_size} "
                f"exceeds limit {self.max_modifications_in_window}"
            )
            return False
        return True

    def check_all(
        self,
        current_complexity: float,
        iteration_results: list[Any],
        current_iteration: int,
    ) -> dict[str, bool]:
        """Run all safety checks.

        Returns dict of check_name -> passed.
        """
        return {
            "complexity_bounds": self.check_complexity_bounds(current_complexity),
            "modification_rate": self.check_modification_rate(
                iteration_results, current_iteration
            ),
        }

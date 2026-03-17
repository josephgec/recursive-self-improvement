"""Conservative alpha scheduler for controlling synthetic data mixing ratio.

Alpha controls the fraction of real (clean) data vs synthetic data in training.
Higher alpha = more real data = more conservative = slower but safer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AlphaScheduleConfig:
    """Configuration for alpha scheduling."""
    schedule_type: str = "exponential"
    gamma: float = 0.95
    initial_alpha: float = 0.5
    min_alpha: float = 0.01
    adaptive_entropy_threshold: float = 2.0
    adaptive_increase_factor: float = 1.2
    linear_decay_rate: float = 0.01


class ConservativeAlphaScheduler:
    """Schedules the alpha mixing parameter conservatively.

    Supports 4 schedule types:
    - exponential: alpha(t) = initial * gamma^t (default, gamma=0.95)
    - linear: alpha(t) = max(initial - decay_rate * t, min_alpha)
    - constant: alpha(t) = initial_alpha for all t
    - adaptive: like exponential, but increases alpha when entropy drops
    """

    def __init__(self, config: Optional[AlphaScheduleConfig] = None):
        self.config = config or AlphaScheduleConfig()
        self._history: List[Dict[str, Any]] = []
        self._adaptive_alpha: Optional[float] = None

    @property
    def schedule_type(self) -> str:
        return self.config.schedule_type

    def get_alpha(
        self,
        iteration: int,
        indicators: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute alpha for a given iteration and optional health indicators.

        Args:
            iteration: Current training iteration (0-indexed).
            indicators: Optional dict with keys like 'entropy', 'kl_divergence'.

        Returns:
            Alpha value in [min_alpha, initial_alpha].
        """
        indicators = indicators or {}

        if self.config.schedule_type == "exponential":
            alpha = self._exponential(iteration)
        elif self.config.schedule_type == "linear":
            alpha = self._linear(iteration)
        elif self.config.schedule_type == "constant":
            alpha = self._constant()
        elif self.config.schedule_type == "adaptive":
            alpha = self._adaptive(iteration, indicators)
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

        alpha = max(alpha, self.config.min_alpha)
        alpha = min(alpha, 1.0)

        self._history.append({
            "iteration": iteration,
            "alpha": alpha,
            "indicators": indicators,
            "schedule_type": self.config.schedule_type,
        })

        return alpha

    def _exponential(self, iteration: int) -> float:
        return self.config.initial_alpha * (self.config.gamma ** iteration)

    def _linear(self, iteration: int) -> float:
        return self.config.initial_alpha - self.config.linear_decay_rate * iteration

    def _constant(self) -> float:
        return self.config.initial_alpha

    def _adaptive(self, iteration: int, indicators: Dict[str, float]) -> float:
        """Adaptive scheduling: increase alpha when entropy drops below threshold."""
        base_alpha = self._exponential(iteration)

        if self._adaptive_alpha is None:
            self._adaptive_alpha = base_alpha

        entropy = indicators.get("entropy")
        if entropy is not None and entropy < self.config.adaptive_entropy_threshold:
            self._adaptive_alpha = min(
                self._adaptive_alpha * self.config.adaptive_increase_factor,
                1.0,
            )
        else:
            self._adaptive_alpha = base_alpha

        return self._adaptive_alpha

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full scheduling history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset scheduler state."""
        self._history.clear()
        self._adaptive_alpha = None

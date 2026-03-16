"""Early stopping based on stagnation detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StagnationState:
    """Current stagnation detection state."""

    best_fitness: float = 0.0
    generations_without_improvement: int = 0
    total_generations: int = 0
    fitness_history: List[float] = field(default_factory=list)
    improvement_threshold: float = 1e-6


class EarlyStopping:
    """Detects stagnation and signals when to stop search."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-6,
        min_generations: int = 5,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_generations = min_generations
        self._state = StagnationState(improvement_threshold=min_delta)

    @property
    def state(self) -> StagnationState:
        return self._state

    def should_stop(self, current_fitness: float) -> bool:
        """Check if search should stop due to stagnation."""
        self._state.total_generations += 1
        self._state.fitness_history.append(current_fitness)

        # Don't stop before minimum generations
        if self._state.total_generations < self.min_generations:
            if current_fitness > self._state.best_fitness + self.min_delta:
                self._state.best_fitness = current_fitness
                self._state.generations_without_improvement = 0
            else:
                self._state.generations_without_improvement += 1
            return False

        # Check for improvement
        if current_fitness > self._state.best_fitness + self.min_delta:
            self._state.best_fitness = current_fitness
            self._state.generations_without_improvement = 0
            return False

        self._state.generations_without_improvement += 1
        return self._state.generations_without_improvement >= self.patience

    def reset(self) -> None:
        """Reset stagnation detection state."""
        self._state = StagnationState(improvement_threshold=self.min_delta)

    def stagnation_ratio(self) -> float:
        """How close we are to triggering early stopping (0.0-1.0)."""
        if self.patience == 0:
            return 1.0
        return self._state.generations_without_improvement / self.patience

    def recent_trend(self, window: int = 5) -> float:
        """Compute recent fitness trend (positive = improving)."""
        history = self._state.fitness_history
        if len(history) < 2:
            return 0.0

        recent = history[-min(window, len(history)):]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def summary(self) -> dict:
        """Return stagnation summary."""
        return {
            "best_fitness": self._state.best_fitness,
            "stagnation_count": self._state.generations_without_improvement,
            "patience": self.patience,
            "stagnation_ratio": self.stagnation_ratio(),
            "total_generations": self._state.total_generations,
            "recent_trend": self.recent_trend(),
        }

from __future__ import annotations

"""Running reward normalization."""

import numpy as np


class RewardNormalizer:
    """Normalizes rewards using running mean and standard deviation.

    Uses Welford's online algorithm for numerically stable
    computation of running statistics.
    """

    def __init__(self, window: int = 100, epsilon: float = 1e-8):
        self._window = window
        self._epsilon = epsilon
        self._values: list[float] = []
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return float(np.sqrt(self._m2 / (self._count - 1)))

    @property
    def count(self) -> int:
        return self._count

    def normalize(self, reward: float) -> float:
        """Normalize a single reward value.

        Updates running statistics and returns normalized value.

        Args:
            reward: Raw reward value.

        Returns:
            Normalized reward (zero-mean, unit-variance).
        """
        self._update_stats(reward)

        if self._count < 2:
            return reward

        return (reward - self._mean) / (self.std + self._epsilon)

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize a batch of rewards.

        Args:
            rewards: Array of raw reward values.

        Returns:
            Array of normalized rewards.
        """
        result = np.empty_like(rewards, dtype=np.float64)
        for i, r in enumerate(rewards):
            result[i] = self.normalize(float(r))
        return result

    def _update_stats(self, value: float) -> None:
        """Update running statistics using Welford's algorithm."""
        self._count += 1
        self._values.append(value)

        # Keep only the window
        if len(self._values) > self._window:
            self._values = self._values[-self._window:]
            # Recompute from window
            self._count = len(self._values)
            self._mean = float(np.mean(self._values))
            self._m2 = float(np.sum((np.array(self._values) - self._mean) ** 2))
        else:
            delta = value - self._mean
            self._mean += delta / self._count
            delta2 = value - self._mean
            self._m2 += delta * delta2

    def reset(self) -> None:
        """Reset running statistics."""
        self._values.clear()
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

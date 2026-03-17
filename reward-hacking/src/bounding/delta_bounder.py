from __future__ import annotations

"""Delta bounding to smooth reward spikes."""

import numpy as np
from dataclasses import dataclass


@dataclass
class DeltaBoundResult:
    """Result from delta bounding a single reward."""

    original: float
    bounded: float
    was_bounded: bool
    delta: float
    max_delta: float


class DeltaBounder:
    """Bounds the change in reward between consecutive steps.

    Prevents sudden reward spikes that may indicate reward hacking.
    """

    def __init__(self, max_delta: float = 2.0):
        if max_delta <= 0:
            raise ValueError(f"max_delta must be positive, got {max_delta}")
        self._max_delta = max_delta
        self._last_reward: float | None = None
        self._bound_count = 0
        self._total_count = 0
        self._history: list[DeltaBoundResult] = []

    @property
    def max_delta(self) -> float:
        return self._max_delta

    @property
    def bound_count(self) -> int:
        return self._bound_count

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def last_reward(self) -> float | None:
        return self._last_reward

    @property
    def history(self) -> list[DeltaBoundResult]:
        return list(self._history)

    def bound(self, reward: float) -> tuple[float, bool]:
        """Bound a single reward value by maximum delta.

        Args:
            reward: Current reward value.

        Returns:
            Tuple of (bounded_reward, was_bounded).
        """
        self._total_count += 1

        if self._last_reward is None:
            self._last_reward = reward
            result = DeltaBoundResult(
                original=reward,
                bounded=reward,
                was_bounded=False,
                delta=0.0,
                max_delta=self._max_delta,
            )
            self._history.append(result)
            return reward, False

        delta = reward - self._last_reward
        was_bounded = abs(delta) > self._max_delta

        if was_bounded:
            self._bound_count += 1
            # Clamp delta to max_delta
            clamped_delta = np.clip(delta, -self._max_delta, self._max_delta)
            bounded = self._last_reward + clamped_delta
        else:
            bounded = reward

        result = DeltaBoundResult(
            original=reward,
            bounded=bounded,
            was_bounded=was_bounded,
            delta=delta,
            max_delta=self._max_delta,
        )
        self._history.append(result)
        self._last_reward = bounded
        return bounded, was_bounded

    def bound_batch(self, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bound a batch of sequential rewards.

        Args:
            rewards: Array of sequential reward values.

        Returns:
            Tuple of (bounded_rewards, was_bounded_mask).
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        bounded = np.empty_like(rewards)
        was_bounded = np.zeros(len(rewards), dtype=bool)

        for i, r in enumerate(rewards):
            bounded[i], was_bounded[i] = self.bound(float(r))

        return bounded, was_bounded

    def reset(self) -> None:
        """Reset the bounder state."""
        self._last_reward = None
        self._bound_count = 0
        self._total_count = 0
        self._history.clear()

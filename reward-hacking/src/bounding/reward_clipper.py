from __future__ import annotations

"""Reward clipping for bounding reward signals."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ClipStats:
    """Statistics from a clipping operation."""

    num_clipped_low: int
    num_clipped_high: int
    total: int
    fraction_clipped: float
    original_mean: float
    original_std: float
    clipped_mean: float
    clipped_std: float


class RewardClipper:
    """Clips rewards to a specified range.

    Prevents extreme reward values from destabilizing training.
    """

    def __init__(self, clip_min: float = -5.0, clip_max: float = 5.0):
        if clip_min >= clip_max:
            raise ValueError(f"clip_min ({clip_min}) must be < clip_max ({clip_max})")
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._total_clipped = 0
        self._total_seen = 0
        self._clip_history: list[ClipStats] = []

    @property
    def clip_min(self) -> float:
        return self._clip_min

    @property
    def clip_max(self) -> float:
        return self._clip_max

    @property
    def total_clipped(self) -> int:
        return self._total_clipped

    @property
    def total_seen(self) -> int:
        return self._total_seen

    @property
    def clip_history(self) -> list[ClipStats]:
        return list(self._clip_history)

    def clip(self, rewards: np.ndarray) -> tuple[np.ndarray, ClipStats]:
        """Clip rewards to configured range.

        Args:
            rewards: Array of reward values.

        Returns:
            Tuple of (clipped_rewards, ClipStats).
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        original_mean = float(np.mean(rewards))
        original_std = float(np.std(rewards))

        num_low = int(np.sum(rewards < self._clip_min))
        num_high = int(np.sum(rewards > self._clip_max))

        clipped = np.clip(rewards, self._clip_min, self._clip_max)

        total = len(rewards)
        self._total_clipped += num_low + num_high
        self._total_seen += total

        stats = ClipStats(
            num_clipped_low=num_low,
            num_clipped_high=num_high,
            total=total,
            fraction_clipped=(num_low + num_high) / max(total, 1),
            original_mean=original_mean,
            original_std=original_std,
            clipped_mean=float(np.mean(clipped)),
            clipped_std=float(np.std(clipped)),
        )
        self._clip_history.append(stats)
        return clipped, stats

    def clip_scalar(self, reward: float) -> float:
        """Clip a single scalar reward."""
        return float(np.clip(reward, self._clip_min, self._clip_max))

from __future__ import annotations

"""Process reward shaping pipeline: normalize -> clip -> delta-bound."""

import numpy as np
from dataclasses import dataclass

from .reward_normalizer import RewardNormalizer
from .reward_clipper import RewardClipper, ClipStats
from .delta_bounder import DeltaBounder


@dataclass
class ShapedReward:
    """Result of the full reward shaping pipeline."""

    raw: float
    normalized: float
    clipped: float
    bounded: float
    final: float
    was_clipped: bool
    was_delta_bounded: bool


class ProcessRewardShaper:
    """Full reward shaping pipeline: normalize -> clip -> delta-bound.

    Applies three transformations in sequence to produce a safe,
    bounded reward signal.
    """

    def __init__(
        self,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
        delta_max: float = 2.0,
        normalize: bool = True,
        window: int = 100,
    ):
        self._normalize_enabled = normalize
        self._normalizer = RewardNormalizer(window=window)
        self._clipper = RewardClipper(clip_min=clip_min, clip_max=clip_max)
        self._bounder = DeltaBounder(max_delta=delta_max)
        self._history: list[ShapedReward] = []

    @property
    def normalizer(self) -> RewardNormalizer:
        return self._normalizer

    @property
    def clipper(self) -> RewardClipper:
        return self._clipper

    @property
    def bounder(self) -> DeltaBounder:
        return self._bounder

    @property
    def history(self) -> list[ShapedReward]:
        return list(self._history)

    def shape(self, raw_reward: float) -> ShapedReward:
        """Apply full shaping pipeline to a raw reward.

        Pipeline: normalize -> clip -> delta-bound.

        Args:
            raw_reward: Raw reward value from the environment.

        Returns:
            ShapedReward with intermediate and final values.
        """
        # Step 1: Normalize
        if self._normalize_enabled:
            normalized = self._normalizer.normalize(raw_reward)
        else:
            normalized = raw_reward

        # Step 2: Clip
        clipped_arr, clip_stats = self._clipper.clip(np.array([normalized]))
        clipped = float(clipped_arr[0])
        was_clipped = clip_stats.num_clipped_low > 0 or clip_stats.num_clipped_high > 0

        # Step 3: Delta-bound
        bounded, was_delta_bounded = self._bounder.bound(clipped)

        result = ShapedReward(
            raw=raw_reward,
            normalized=normalized,
            clipped=clipped,
            bounded=bounded,
            final=bounded,
            was_clipped=was_clipped,
            was_delta_bounded=was_delta_bounded,
        )
        self._history.append(result)
        return result

    def shape_batch(self, raw_rewards: np.ndarray) -> list[ShapedReward]:
        """Shape a batch of sequential rewards.

        Args:
            raw_rewards: Array of raw reward values.

        Returns:
            List of ShapedReward results.
        """
        results = []
        for r in np.asarray(raw_rewards).flatten():
            results.append(self.shape(float(r)))
        return results

    def reset(self) -> None:
        """Reset all pipeline components."""
        self._normalizer.reset()
        self._bounder.reset()
        self._history.clear()

from __future__ import annotations

"""Detect divergence between reward and accuracy trends."""

import numpy as np
from dataclasses import dataclass


@dataclass
class DivergenceResult:
    """Result of reward-accuracy divergence check."""

    is_diverging: bool
    reward_trend: float
    accuracy_trend: float
    divergence_score: float
    description: str
    window_size: int


class RewardAccuracyDivergenceDetector:
    """Detects when reward increases while accuracy stays flat or declines.

    This is a key indicator of reward hacking: the model learns to
    game the reward signal without improving actual performance.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        window: int = 20,
    ):
        self._threshold = threshold
        self._window = window
        self._rewards: list[float] = []
        self._accuracies: list[float] = []
        self._results: list[DivergenceResult] = []

    @property
    def reward_history(self) -> list[float]:
        return list(self._rewards)

    @property
    def accuracy_history(self) -> list[float]:
        return list(self._accuracies)

    @property
    def results(self) -> list[DivergenceResult]:
        return list(self._results)

    def update(self, reward: float, accuracy: float) -> None:
        """Record a new reward and accuracy measurement.

        Args:
            reward: Current reward value.
            accuracy: Current accuracy value.
        """
        self._rewards.append(reward)
        self._accuracies.append(accuracy)

    def check(self) -> DivergenceResult:
        """Check for reward-accuracy divergence.

        Returns:
            DivergenceResult with analysis.
        """
        if len(self._rewards) < self._window:
            result = DivergenceResult(
                is_diverging=False,
                reward_trend=0.0,
                accuracy_trend=0.0,
                divergence_score=0.0,
                description="Insufficient data for divergence detection",
                window_size=len(self._rewards),
            )
            self._results.append(result)
            return result

        recent_rewards = np.array(self._rewards[-self._window:])
        recent_accuracy = np.array(self._accuracies[-self._window:])

        # Compute trends via linear regression slopes
        steps = np.arange(self._window)
        reward_slope = float(np.polyfit(steps, recent_rewards, 1)[0])
        accuracy_slope = float(np.polyfit(steps, recent_accuracy, 1)[0])

        # Normalize slopes by their respective ranges
        reward_range = float(np.ptp(recent_rewards))
        accuracy_range = float(np.ptp(recent_accuracy))

        norm_reward_slope = reward_slope / max(reward_range, 1e-8)
        norm_accuracy_slope = accuracy_slope / max(accuracy_range, 1e-8)

        # Divergence: reward going up while accuracy flat or down
        divergence_score = max(0.0, norm_reward_slope - norm_accuracy_slope)
        is_diverging = (
            reward_slope > 0
            and accuracy_slope <= 0.01
            and divergence_score > self._threshold
        )

        if is_diverging:
            description = (
                f"Reward-accuracy divergence detected: "
                f"reward slope={reward_slope:.4f}, "
                f"accuracy slope={accuracy_slope:.4f}, "
                f"divergence={divergence_score:.4f}"
            )
        else:
            description = "No divergence detected"

        result = DivergenceResult(
            is_diverging=is_diverging,
            reward_trend=reward_slope,
            accuracy_trend=accuracy_slope,
            divergence_score=divergence_score,
            description=description,
            window_size=self._window,
        )
        self._results.append(result)
        return result

    def reset(self) -> None:
        """Reset all history."""
        self._rewards.clear()
        self._accuracies.clear()
        self._results.clear()

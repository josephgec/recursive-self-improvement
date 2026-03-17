"""Reward sanity checking.

Validates that reward signals are within expected bounds and
exhibit reasonable statistical properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SanityResult:
    """Result of a reward sanity check."""
    sane: bool
    violations: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class RewardSanityChecker:
    """Checks reward signals for sanity.

    Validates:
    - Rewards are within bounds [min_reward, max_reward]
    - Standard deviation is reasonable
    - No NaN or infinite values
    - Distribution is not degenerate
    """

    def __init__(
        self,
        min_reward: float = -10.0,
        max_reward: float = 100.0,
        max_std_dev: float = 5.0,
    ):
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.max_std_dev = max_std_dev

    def check(self, rewards: List[float]) -> SanityResult:
        """Check a list of rewards for sanity.

        Args:
            rewards: List of reward values to check.

        Returns:
            SanityResult with pass/fail and violation details.
        """
        violations = []
        stats: Dict[str, float] = {}

        if not rewards:
            return SanityResult(
                sane=True,
                violations=[],
                stats={"count": 0},
            )

        n = len(rewards)
        stats["count"] = n

        # Check for non-finite values
        for i, r in enumerate(rewards):
            if r != r:  # NaN check
                violations.append(f"NaN reward at index {i}")
            elif abs(r) == float("inf"):
                violations.append(f"Infinite reward at index {i}")

        # Filter to finite values for stats
        finite_rewards = [r for r in rewards if r == r and abs(r) != float("inf")]
        if not finite_rewards:
            return SanityResult(
                sane=False,
                violations=violations or ["No finite rewards"],
                stats=stats,
            )

        mean_r = sum(finite_rewards) / len(finite_rewards)
        var_r = sum((r - mean_r) ** 2 for r in finite_rewards) / len(finite_rewards)
        std_r = var_r ** 0.5
        min_r = min(finite_rewards)
        max_r = max(finite_rewards)

        stats.update({
            "mean": mean_r,
            "std": std_r,
            "min": min_r,
            "max": max_r,
        })

        # Check bounds
        out_of_bounds = [r for r in finite_rewards if r < self.min_reward or r > self.max_reward]
        if out_of_bounds:
            violations.append(
                f"{len(out_of_bounds)} rewards out of bounds "
                f"[{self.min_reward}, {self.max_reward}]"
            )

        # Check std dev
        if std_r > self.max_std_dev:
            violations.append(
                f"Reward std dev {std_r:.2f} exceeds max {self.max_std_dev:.2f}"
            )

        # Check for degenerate distribution (all same value)
        if len(finite_rewards) >= 10 and std_r == 0.0:
            violations.append("Degenerate reward distribution (zero variance)")

        return SanityResult(
            sane=len(violations) == 0,
            violations=violations,
            stats=stats,
        )

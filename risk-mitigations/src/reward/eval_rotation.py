"""Evaluation set rotation to prevent overfitting to specific benchmarks.

Periodically rotates the evaluation set to ensure the agent genuinely
improves rather than memorizing specific test cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RotationRecord:
    """Record of an eval set rotation."""
    iteration: int
    retired_tasks: List[str]
    activated_tasks: List[str]
    reason: str


class EvalSetRotator:
    """Rotates evaluation sets periodically.

    Maintains active and reserve eval sets, swapping tasks to prevent
    overfitting to specific benchmarks.
    """

    def __init__(
        self,
        rotate_every_n: int = 10,
        reserve_fraction: float = 0.3,
        seed: int = 42,
    ):
        self.rotate_every_n = rotate_every_n
        self.reserve_fraction = reserve_fraction
        self._rng = random.Random(seed)
        self._last_rotation: int = 0
        self._rotation_history: List[RotationRecord] = []

    def should_rotate(self, iteration: int) -> bool:
        """Check if rotation should happen at this iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            True if enough iterations have passed since last rotation.
        """
        return (iteration - self._last_rotation) >= self.rotate_every_n

    def rotate(
        self,
        current_tasks: List[str],
        reserve_tasks: List[str],
        iteration: int = 0,
    ) -> Dict[str, List[str]]:
        """Rotate evaluation tasks between active and reserve sets.

        Args:
            current_tasks: Currently active task IDs.
            reserve_tasks: Reserve task IDs.
            iteration: Current iteration for record-keeping.

        Returns:
            Dict with 'active' and 'reserve' task lists.
        """
        if not reserve_tasks:
            return {"active": current_tasks, "reserve": reserve_tasks}

        # Determine how many to swap
        n_swap = max(1, int(len(current_tasks) * self.reserve_fraction))
        n_swap = min(n_swap, len(current_tasks), len(reserve_tasks))

        # Select tasks to retire and activate
        retired = self._rng.sample(current_tasks, n_swap)
        activated = self._rng.sample(reserve_tasks, n_swap)

        # Perform swap
        new_active = [t for t in current_tasks if t not in retired] + activated
        new_reserve = [t for t in reserve_tasks if t not in activated] + retired

        # Record rotation
        record = RotationRecord(
            iteration=iteration,
            retired_tasks=retired,
            activated_tasks=activated,
            reason=f"Scheduled rotation at iteration {iteration}",
        )
        self._rotation_history.append(record)
        self._last_rotation = iteration

        return {"active": new_active, "reserve": new_reserve}

    def get_rotation_history(self) -> List[RotationRecord]:
        """Return rotation history."""
        return list(self._rotation_history)

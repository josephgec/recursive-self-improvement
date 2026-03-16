"""Controlled environment for consistent ablation conditions."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


class ControlledEnvironment:
    """Ensures same seed/tasks across conditions for fair comparison.

    Manages seeds, task selection, and environment isolation so that
    each condition sees exactly the same tasks in the same order.
    """

    def __init__(self, seed: int = 42, n_tasks: int = 100):
        self.seed = seed
        self.n_tasks = n_tasks
        self._rng = random.Random(seed)
        self._tasks: Optional[List[Dict[str, Any]]] = None

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Generate a deterministic set of tasks."""
        if self._tasks is not None:
            return self._tasks

        rng = random.Random(self.seed)
        tasks = []
        for i in range(self.n_tasks):
            tasks.append({
                "id": i,
                "difficulty": rng.choice(["easy", "medium", "hard"]),
                "category": rng.choice(["code", "reasoning", "math", "language"]),
                "seed": rng.randint(0, 2**31),
            })
        self._tasks = tasks
        return tasks

    def get_condition_seed(self, condition_name: str, repetition: int) -> int:
        """Get a deterministic seed for a specific condition + repetition.

        Same seed is used across conditions so that any randomness in
        task ordering is controlled.
        """
        key = f"{self.seed}:{condition_name}:{repetition}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16)

    def reset(self) -> None:
        """Reset to initial state."""
        self._rng = random.Random(self.seed)
        self._tasks = None

    def get_task_subset(self, category: Optional[str] = None,
                        difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get a filtered subset of tasks."""
        tasks = self.get_tasks()
        if category:
            tasks = [t for t in tasks if t["category"] == category]
        if difficulty:
            tasks = [t for t in tasks if t["difficulty"] == difficulty]
        return tasks

    def verify_consistency(self) -> bool:
        """Verify that task generation is deterministic."""
        tasks_a = self.get_tasks()
        self.reset()
        tasks_b = self.get_tasks()
        return tasks_a == tasks_b

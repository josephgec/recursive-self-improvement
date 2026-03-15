"""Few-shot example selection."""

from __future__ import annotations

import random
from typing import Any

from src.core.executor import Task


class FewShotSelector:
    """Selects few-shot examples for a task with category-matching fallback."""

    def select(
        self, task: Task, pool: list[dict[str, Any]], n: int = 3
    ) -> list[dict[str, Any]]:
        """Select n examples from the pool for the given task.

        Strategy:
        1. Prefer examples from the same category
        2. Prefer examples from the same domain
        3. Fall back to random selection
        """
        if not pool:
            return []

        n = min(n, len(pool))

        # Category match
        same_category = [
            ex for ex in pool if ex.get("category", "") == task.category and task.category
        ]
        if len(same_category) >= n:
            return random.sample(same_category, n)

        # Domain match
        same_domain = [
            ex for ex in pool if ex.get("domain", "") == task.domain and task.domain
        ]
        selected = list(same_category)
        for ex in same_domain:
            if ex not in selected and len(selected) < n:
                selected.append(ex)

        # Random fill
        remaining = [ex for ex in pool if ex not in selected]
        while len(selected) < n and remaining:
            pick = random.choice(remaining)
            selected.append(pick)
            remaining.remove(pick)

        return selected[:n]

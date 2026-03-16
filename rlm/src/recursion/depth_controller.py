"""Depth controller: enforce recursion and iteration budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class BudgetExhaustedError(Exception):
    """Raised when the recursion or iteration budget is exhausted."""
    pass


@dataclass
class DepthController:
    """Track and enforce recursion depth and iteration budgets."""

    max_depth: int = 3
    max_iterations: int = 10
    max_sub_queries: int = 5
    budget_fraction: float = 0.5

    _current_depth: int = 0
    _total_calls: int = 0
    _sub_query_counts: Dict[int, int] = field(default_factory=dict)

    def can_recurse(self, depth: int | None = None) -> bool:
        """Check whether a new recursive call is allowed at the given depth."""
        d = depth if depth is not None else self._current_depth
        if d >= self.max_depth:
            return False
        count = self._sub_query_counts.get(d, 0)
        if count >= self.max_sub_queries:
            return False
        return True

    def register_call(self, depth: int | None = None) -> None:
        """Register a top-level call."""
        self._total_calls += 1
        if depth is not None:
            self._current_depth = depth

    def register_sub_query(self, parent_depth: int) -> int:
        """Register a sub-query and return the child depth.

        Raises BudgetExhaustedError if the budget is exceeded.
        """
        if not self.can_recurse(parent_depth):
            raise BudgetExhaustedError(
                f"Cannot recurse further: depth={parent_depth}, "
                f"max_depth={self.max_depth}, "
                f"sub_queries_at_depth={self._sub_query_counts.get(parent_depth, 0)}"
            )
        self._sub_query_counts[parent_depth] = self._sub_query_counts.get(parent_depth, 0) + 1
        child_depth = parent_depth + 1
        self._current_depth = child_depth
        return child_depth

    def remaining_budget(self, depth: int | None = None) -> int:
        """Return how many sub-queries are still allowed at the given depth."""
        d = depth if depth is not None else self._current_depth
        used = self._sub_query_counts.get(d, 0)
        return max(0, self.max_sub_queries - used)

    def allocate_sub_budget(self, parent_iterations: int) -> int:
        """Compute the iteration budget for a child session.

        The child gets ``budget_fraction`` of the parent's remaining iterations.
        """
        return max(1, int(parent_iterations * self.budget_fraction))

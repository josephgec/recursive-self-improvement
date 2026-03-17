"""Budget manager for controlling costs at multiple levels.

Manages budgets across 4 levels: query, session, iteration, and phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


BUDGET_LEVELS = ("query", "session", "iteration", "phase")


@dataclass
class BudgetState:
    """State of a single budget level."""
    level: str
    limit: float
    spent: float
    remaining: float
    utilization: float  # 0-1


class BudgetManager:
    """Manages budgets across 4 hierarchical levels.

    Levels (from most granular to broadest):
    - query: Per-query budget
    - session: Per-session budget
    - iteration: Per-iteration budget
    - phase: Per-phase budget
    """

    def __init__(
        self,
        query_limit: float = 1.00,
        session_limit: float = 50.00,
        iteration_limit: float = 500.00,
        phase_limit: float = 5000.00,
    ):
        self._limits = {
            "query": query_limit,
            "session": session_limit,
            "iteration": iteration_limit,
            "phase": phase_limit,
        }
        self._spent: Dict[str, float] = {level: 0.0 for level in BUDGET_LEVELS}
        self._spend_log: List[Dict[str, Any]] = []

    def can_spend(self, amount: float, level: str = "query") -> bool:
        """Check if an amount can be spent at the given level.

        Also checks all higher levels (session, iteration, phase).

        Args:
            amount: Amount to check.
            level: Budget level.

        Returns:
            True if the spend is within budget at all relevant levels.
        """
        if level not in BUDGET_LEVELS:
            raise ValueError(f"Invalid level: {level}. Must be one of {BUDGET_LEVELS}")

        level_idx = BUDGET_LEVELS.index(level)
        for i in range(level_idx, len(BUDGET_LEVELS)):
            check_level = BUDGET_LEVELS[i]
            if self._spent[check_level] + amount > self._limits[check_level]:
                return False
        return True

    def spend(self, amount: float, level: str = "query", description: str = "") -> bool:
        """Record a spend at the given level.

        Updates all higher levels as well.

        Args:
            amount: Amount spent.
            level: Budget level.
            description: Optional description.

        Returns:
            True if spend was recorded, False if it would exceed budget.
        """
        if not self.can_spend(amount, level):
            return False

        level_idx = BUDGET_LEVELS.index(level)
        for i in range(level_idx, len(BUDGET_LEVELS)):
            self._spent[BUDGET_LEVELS[i]] += amount

        self._spend_log.append({
            "amount": amount,
            "level": level,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "remaining": {
                lvl: self._limits[lvl] - self._spent[lvl]
                for lvl in BUDGET_LEVELS
            },
        })

        return True

    def remaining(self, level: str = "phase") -> float:
        """Return remaining budget at the given level."""
        if level not in BUDGET_LEVELS:
            raise ValueError(f"Invalid level: {level}")
        return self._limits[level] - self._spent[level]

    def burn_rate(self, level: str = "session") -> float:
        """Calculate the current burn rate (spend per log entry).

        Args:
            level: Budget level to calculate burn rate for.

        Returns:
            Average spend per entry at this level.
        """
        level_entries = [
            e for e in self._spend_log
            if BUDGET_LEVELS.index(e["level"]) <= BUDGET_LEVELS.index(level)
        ]
        if not level_entries:
            return 0.0
        total = sum(e["amount"] for e in level_entries)
        return total / len(level_entries)

    def get_state(self, level: str) -> BudgetState:
        """Get the current state for a budget level."""
        if level not in BUDGET_LEVELS:
            raise ValueError(f"Invalid level: {level}")
        limit = self._limits[level]
        spent = self._spent[level]
        return BudgetState(
            level=level,
            limit=limit,
            spent=spent,
            remaining=limit - spent,
            utilization=spent / limit if limit > 0 else 0.0,
        )

    def get_all_states(self) -> List[BudgetState]:
        """Get states for all budget levels."""
        return [self.get_state(level) for level in BUDGET_LEVELS]

    def reset_level(self, level: str) -> None:
        """Reset a specific budget level (e.g., at start of new session)."""
        if level not in BUDGET_LEVELS:
            raise ValueError(f"Invalid level: {level}")
        self._spent[level] = 0.0

    def get_spend_log(self) -> List[Dict[str, Any]]:
        """Return the full spend log."""
        return list(self._spend_log)

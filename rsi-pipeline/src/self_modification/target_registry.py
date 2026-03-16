"""Target registry: manages allowed and forbidden modification targets."""
from __future__ import annotations

from typing import List, Optional, Set


class TargetRegistry:
    """Registry of allowed and forbidden modification targets."""

    def __init__(
        self,
        allowed: Optional[List[str]] = None,
        forbidden: Optional[List[str]] = None,
    ):
        self._allowed: Set[str] = set(allowed or [
            "strategy_evolver", "candidate_pool",
            "empirical_gate", "compactness_gate", "pareto_filter",
        ])
        self._forbidden: Set[str] = set(forbidden or [
            "emergency_stop", "constraint_enforcer", "gdi_monitor",
        ])

    def is_allowed(self, target: str) -> bool:
        """Check if a target is allowed for modification."""
        return target in self._allowed and target not in self._forbidden

    def is_forbidden(self, target: str) -> bool:
        """Check if a target is explicitly forbidden."""
        return target in self._forbidden

    def list_allowed(self) -> List[str]:
        """List all allowed targets."""
        return sorted(self._allowed - self._forbidden)

    def list_forbidden(self) -> List[str]:
        """List all forbidden targets."""
        return sorted(self._forbidden)

    def add_allowed(self, target: str) -> None:
        """Add a target to the allowed list."""
        self._allowed.add(target)

    def add_forbidden(self, target: str) -> None:
        """Add a target to the forbidden list."""
        self._forbidden.add(target)

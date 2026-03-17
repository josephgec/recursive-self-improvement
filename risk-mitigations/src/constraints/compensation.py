"""Compensation monitoring for constraint relaxations.

When a constraint is relaxed, compensation mechanisms are activated
to maintain overall safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompensationEntry:
    """Record of an activated compensation mechanism."""
    constraint_name: str
    mechanism: str
    description: str
    active: bool = True
    activation_reason: str = ""


# Default compensation mechanisms for each type of relaxation
DEFAULT_COMPENSATIONS = {
    "quality_threshold": {
        "mechanism": "increased_monitoring",
        "description": "Increase quality monitoring frequency by 2x",
    },
    "diversity_threshold": {
        "mechanism": "diversity_audit",
        "description": "Run diversity audit after each iteration",
    },
    "latency_limit": {
        "mechanism": "batch_validation",
        "description": "Add batch validation step for latency-sensitive outputs",
    },
    "error_rate": {
        "mechanism": "error_sampling",
        "description": "Sample and review errors at higher rate",
    },
}


class CompensationMonitor:
    """Monitors and manages compensation mechanisms for relaxed constraints.

    When a constraint is relaxed, appropriate compensations are activated
    to maintain safety margins.
    """

    def __init__(
        self,
        compensations: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self._compensations_registry = compensations or DEFAULT_COMPENSATIONS
        self._active: Dict[str, CompensationEntry] = {}
        self._history: List[CompensationEntry] = []

    def activate_compensation(
        self,
        constraint_name: str,
        reason: str = "",
    ) -> CompensationEntry:
        """Activate compensation for a relaxed constraint.

        Args:
            constraint_name: The constraint that was relaxed.
            reason: Why compensation is needed.

        Returns:
            CompensationEntry for the activated mechanism.
        """
        comp_info = self._compensations_registry.get(constraint_name, {
            "mechanism": "generic_monitoring",
            "description": f"Enhanced monitoring for relaxed '{constraint_name}'",
        })

        entry = CompensationEntry(
            constraint_name=constraint_name,
            mechanism=comp_info["mechanism"],
            description=comp_info["description"],
            active=True,
            activation_reason=reason,
        )

        self._active[constraint_name] = entry
        self._history.append(entry)
        return entry

    def deactivate(self, constraint_name: str) -> bool:
        """Deactivate compensation for a constraint.

        Args:
            constraint_name: The constraint whose compensation to deactivate.

        Returns:
            True if deactivated, False if not found.
        """
        entry = self._active.get(constraint_name)
        if entry is None:
            return False

        entry.active = False
        del self._active[constraint_name]
        return True

    def get_active_compensations(self) -> List[CompensationEntry]:
        """Return all currently active compensations."""
        return [e for e in self._active.values() if e.active]

    def is_compensated(self, constraint_name: str) -> bool:
        """Check if a constraint has active compensation."""
        entry = self._active.get(constraint_name)
        return entry is not None and entry.active

    def get_history(self) -> List[CompensationEntry]:
        """Return full compensation history."""
        return list(self._history)

"""Base classes for the constraint system."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConstraintResult:
    """Result of evaluating a single constraint."""

    satisfied: bool
    measured_value: float
    threshold: float
    headroom: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # headroom is threshold-distance in the direction that matters
        pass


@dataclass
class CheckContext:
    """Context passed into every constraint check."""

    modification_type: str = "unknown"
    modification_description: str = ""
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Constraint(abc.ABC):
    """Abstract base class for all constraints.

    Every constraint is immutable once registered and must implement
    `check(agent_state, context) -> ConstraintResult`.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        threshold: float,
        is_immutable: bool = True,
    ) -> None:
        self._name = name
        self._description = description
        self._category = category
        self._threshold = threshold
        self._is_immutable = is_immutable

    # --- read-only properties ---------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_immutable(self) -> bool:
        return self._is_immutable

    # --- abstract interface -----------------------------------------------------

    @abc.abstractmethod
    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate this constraint against *agent_state*."""

    # --- helpers ----------------------------------------------------------------

    def headroom(self, measured_value: float) -> float:
        """Return headroom: positive means constraint is satisfied with room to spare."""
        return measured_value - self._threshold

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"category={self._category!r}, threshold={self._threshold})"
        )

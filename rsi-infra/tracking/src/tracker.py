"""Core experiment tracking abstraction.

Every backend (W&B, local JSONL, etc.) implements ``ExperimentTracker``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def init_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialise a new tracking run."""

    @abstractmethod
    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        """Log metrics for a given generation."""

    @abstractmethod
    def log_drift(self, generation: int, drift: dict[str, Any]) -> None:
        """Log goal-drift measurements."""

    @abstractmethod
    def log_constraints(self, generation: int, report: dict[str, Any]) -> None:
        """Log constraint-preservation results."""

    @abstractmethod
    def log_alert(self, alert: dict[str, Any]) -> None:
        """Log an alert event."""

    @abstractmethod
    def finish(self) -> None:
        """Finalise the run and flush all data."""

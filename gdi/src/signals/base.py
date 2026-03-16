"""Base classes for drift signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SignalResult:
    """Result from a drift signal computation."""
    signal_name: str
    raw_score: float
    normalized_score: float
    interpretation: str
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftSignal(ABC):
    """Abstract base class for drift signals.

    Each signal measures a different aspect of output drift between
    current outputs and reference outputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal name identifier."""
        ...

    @abstractmethod
    def compute(
        self, current: List[str], reference: List[str]
    ) -> SignalResult:
        """Compute drift signal between current and reference outputs.

        Args:
            current: List of current output strings.
            reference: List of reference output strings.

        Returns:
            SignalResult with raw and normalized scores.
        """
        ...

    @abstractmethod
    def normalize(self, raw: float) -> float:
        """Normalize a raw score to [0, 1] range.

        Args:
            raw: Raw signal score.

        Returns:
            Normalized score in [0, 1].
        """
        ...

    def interpret(self, normalized: float) -> str:
        """Interpret a normalized score as a human-readable string.

        Args:
            normalized: Normalized score in [0, 1].

        Returns:
            Interpretation string.
        """
        if normalized < 0.15:
            return "minimal_drift"
        elif normalized < 0.40:
            return "moderate_drift"
        elif normalized < 0.70:
            return "significant_drift"
        else:
            return "severe_drift"

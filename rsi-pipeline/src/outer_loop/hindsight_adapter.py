"""Hindsight adapter: feeds iteration outcomes back to SOAR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainingPair:
    """A training pair derived from hindsight analysis."""
    input_state: Dict[str, Any] = field(default_factory=dict)
    output_action: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HindsightAdapter:
    """Collects outcomes from iterations and generates training pairs for SOAR."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._training_pairs: List[TrainingPair] = []

    def collect_from_iteration(self, iteration_result: Any) -> None:
        """Collect data from an iteration result."""
        result_dict = iteration_result.to_dict() if hasattr(iteration_result, 'to_dict') else {}
        self._history.append(result_dict)

        # Generate training pair from the outcome
        pair = TrainingPair(
            input_state={"iteration": result_dict.get("iteration", 0)},
            output_action={"candidate": result_dict.get("candidate", {})},
            reward=1.0 if result_dict.get("improved", False) else -0.5,
            metadata={"safety_verdict": result_dict.get("safety_verdict", "unknown")},
        )
        self._training_pairs.append(pair)

    def feed_to_soar(self) -> List[TrainingPair]:
        """Return accumulated training pairs for SOAR population update."""
        pairs = list(self._training_pairs)
        return pairs

    def get_training_pairs(self) -> List[TrainingPair]:
        """Get all training pairs."""
        return list(self._training_pairs)

    def clear(self) -> None:
        """Clear all collected data."""
        self._history.clear()
        self._training_pairs.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def pair_count(self) -> int:
        return len(self._training_pairs)

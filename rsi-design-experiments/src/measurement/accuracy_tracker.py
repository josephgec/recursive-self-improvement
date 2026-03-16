"""Tracks accuracy metrics across iterations."""

from typing import Dict, List


class AccuracyTracker:
    """Records and reports accuracy measurements over time."""

    def __init__(self):
        self._records: List[Dict[str, float]] = []
        self._by_type: Dict[str, List[float]] = {}

    def record(self, accuracy: float, accuracy_type: str = "in_distribution"):
        """Record an accuracy measurement."""
        self._records.append({"accuracy": accuracy, "type": accuracy_type})
        if accuracy_type not in self._by_type:
            self._by_type[accuracy_type] = []
        self._by_type[accuracy_type].append(accuracy)

    def get_overall(self) -> float:
        """Get overall mean accuracy across all recorded values."""
        if not self._records:
            return 0.0
        return sum(r["accuracy"] for r in self._records) / len(self._records)

    def get_per_type(self, accuracy_type: str) -> float:
        """Get mean accuracy for a specific type (in_distribution or out_of_distribution)."""
        values = self._by_type.get(accuracy_type, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_trajectory(self) -> List[float]:
        """Get the full accuracy trajectory over time."""
        return [r["accuracy"] for r in self._records]

    def get_final(self) -> float:
        """Get the last recorded accuracy."""
        if not self._records:
            return 0.0
        return self._records[-1]["accuracy"]

    def count(self) -> int:
        """Number of recorded measurements."""
        return len(self._records)

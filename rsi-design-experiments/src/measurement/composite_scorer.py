"""Composite scoring: weighted combination of multiple metrics."""

from typing import Dict, Optional


class CompositeScorer:
    """Computes a weighted composite score from multiple metrics."""

    DEFAULT_WEIGHTS = {
        "accuracy": 0.4,
        "stability": 0.3,
        "efficiency": 0.15,
        "generalization": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def score(
        self,
        accuracy: float,
        stability: float,
        efficiency: float,
        generalization: float,
    ) -> float:
        """Compute weighted composite score.

        All inputs should be in [0, 1].
        Default: 0.4*accuracy + 0.3*stability + 0.15*efficiency + 0.15*generalization.
        """
        return (
            self.weights["accuracy"] * accuracy
            + self.weights["stability"] * stability
            + self.weights["efficiency"] * efficiency
            + self.weights["generalization"] * generalization
        )

    def score_from_dict(self, metrics: Dict[str, float]) -> float:
        """Compute composite score from a dict of metric values."""
        total = 0.0
        for key, weight in self.weights.items():
            total += weight * metrics.get(key, 0.0)
        return total

    def get_weights(self) -> Dict[str, float]:
        """Return the current weights."""
        return dict(self.weights)

    def set_weights(self, weights: Dict[str, float]):
        """Update weights."""
        self.weights = dict(weights)

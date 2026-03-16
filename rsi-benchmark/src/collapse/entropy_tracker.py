"""Entropy tracker: compute entropy curves and diversity metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class DiversityMetrics:
    """Metrics measuring output diversity."""
    unique_ratio: float  # Fraction of unique outputs
    entropy: float  # Shannon entropy of output distribution
    simpson_diversity: float  # Simpson's diversity index
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntropyTracker:
    """Track entropy and diversity of agent outputs over iterations."""

    def __init__(self) -> None:
        self._outputs: Dict[int, List[str]] = {}  # iteration -> outputs
        self._entropy_curve: List[Tuple[int, float]] = []

    def record_outputs(self, iteration: int, outputs: List[str]) -> None:
        """Record agent outputs for an iteration."""
        self._outputs[iteration] = outputs
        entropy = self._compute_shannon_entropy(outputs)
        self._entropy_curve.append((iteration, entropy))
        self._entropy_curve.sort(key=lambda x: x[0])

    def compute_entropy_curve(self) -> List[Tuple[int, float]]:
        """Get the entropy curve over iterations."""
        return list(self._entropy_curve)

    def compute_diversity_metrics(self, iteration: int) -> DiversityMetrics:
        """Compute diversity metrics for a specific iteration."""
        outputs = self._outputs.get(iteration, [])
        if not outputs:
            return DiversityMetrics(
                unique_ratio=0.0, entropy=0.0, simpson_diversity=0.0,
            )

        unique = set(outputs)
        unique_ratio = len(unique) / len(outputs)
        entropy = self._compute_shannon_entropy(outputs)
        simpson = self._compute_simpson_diversity(outputs)

        return DiversityMetrics(
            unique_ratio=unique_ratio,
            entropy=entropy,
            simpson_diversity=simpson,
        )

    def compute_all_diversity_metrics(self) -> Dict[int, DiversityMetrics]:
        """Compute diversity metrics for all recorded iterations."""
        return {it: self.compute_diversity_metrics(it) for it in self._outputs}

    def _compute_shannon_entropy(self, outputs: List[str]) -> float:
        """Compute Shannon entropy of output distribution."""
        if not outputs:
            return 0.0

        counts: Dict[str, int] = {}
        for o in outputs:
            counts[o] = counts.get(o, 0) + 1

        total = len(outputs)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _compute_simpson_diversity(self, outputs: List[str]) -> float:
        """Compute Simpson's diversity index: 1 - sum(p_i^2)."""
        if not outputs:
            return 0.0

        counts: Dict[str, int] = {}
        for o in outputs:
            counts[o] = counts.get(o, 0) + 1

        total = len(outputs)
        sum_p_sq = sum((c / total) ** 2 for c in counts.values())
        return 1.0 - sum_p_sq

    def is_entropy_declining(self, window: int = 3) -> bool:
        """Check if entropy is declining (sign of collapse)."""
        if len(self._entropy_curve) < window:
            return False
        recent = [e for _, e in self._entropy_curve[-window:]]
        declines = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
        return declines >= (window - 1) / 2

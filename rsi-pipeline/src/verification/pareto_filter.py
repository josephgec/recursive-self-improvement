"""Pareto filter: selects Pareto-optimal candidates."""
from __future__ import annotations

from typing import List, Tuple


class ParetoFilter:
    """Filters candidates to the Pareto-optimal set based on accuracy and compactness."""

    def __init__(self, objectives: Tuple[str, ...] = ("accuracy", "compactness")):
        self._objectives = objectives

    def filter(self, candidates: List[dict]) -> List[dict]:
        """Filter to Pareto-optimal candidates.

        Each candidate dict should have 'accuracy' (higher=better) and
        'bdm_score' (lower=better).
        """
        if not candidates:
            return []

        pareto: List[dict] = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is c:
                    continue
                if self._dominates(other, c):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)
        return pareto

    def _dominates(self, a: dict, b: dict) -> bool:
        """Check if a dominates b (a is at least as good in all objectives, strictly better in one)."""
        a_acc = a.get("accuracy", 0.0)
        b_acc = b.get("accuracy", 0.0)
        # Lower BDM is better, so invert for comparison
        a_compact = -a.get("bdm_score", 0.0)
        b_compact = -b.get("bdm_score", 0.0)

        a_vals = (a_acc, a_compact)
        b_vals = (b_acc, b_compact)

        at_least_as_good = all(av >= bv for av, bv in zip(a_vals, b_vals))
        strictly_better = any(av > bv for av, bv in zip(a_vals, b_vals))

        return at_least_as_good and strictly_better
